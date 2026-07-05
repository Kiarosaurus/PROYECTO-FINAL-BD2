from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from core.ports.buffer import BufferManager
from indices.inverted.text_preprocessor import DEFAULT_PREPROCESSOR, TextPreprocessor


Postings = dict[str, int]
Block = list[tuple[str, Postings]]

BLOCK_PAGE_SIZE = 4096


@dataclass
class SPIMIBlockBuilder:
    block_document_limit: int = 128
    preprocessor: TextPreprocessor = DEFAULT_PREPROCESSOR
    # Con buffer los bloques cerrados se van a páginas y no quedan en memoria
    buffer: BufferManager | None = None
    file_id: str = "spimi"
    _current: dict[str, Postings] = field(default_factory=dict)
    _blocks: list[Block] = field(default_factory=list)
    _block_pages: list[int] = field(default_factory=list)
    _documents_in_block: int = 0

    def add_document(self, doc_id: str, text: str) -> None:
        for term in self.preprocessor.tokenize(text):
            postings = self._current.setdefault(term, {})
            postings[doc_id] = postings.get(doc_id, 0) + 1
        self._documents_in_block += 1
        if self._documents_in_block >= self.block_document_limit:
            self.flush()

    def flush(self) -> None:
        if not self._current:
            self._documents_in_block = 0
            return
        block = [
            (term, dict(sorted(postings.items())))
            for term, postings in sorted(self._current.items())
        ]
        if self.buffer is None:
            self._blocks.append(block)
        else:
            self._spill_block(block)
        self._current = {}
        self._documents_in_block = 0

    def build(self, documents: Iterable[tuple[str, str]]) -> dict[str, Postings]:
        for doc_id, text in documents:
            self.add_document(doc_id, text)
        self.flush()
        return self.merge_blocks()

    # Serializa un bloque como líneas y lo reparte en páginas de tamaño fijo
    def _spill_block(self, block: Block) -> None:
        stream = b"".join(
            json.dumps({"term": term, "postings": postings}, separators=(",", ":")).encode("utf-8") + b"\n"
            for term, postings in block
        )
        block_file = self._block_file(len(self._block_pages))
        page_count = 0
        for start in range(0, len(stream), BLOCK_PAGE_SIZE):
            page = self.buffer.get(block_file, page_count)
            page.data[:] = stream[start:start + BLOCK_PAGE_SIZE]
            page.dirty = True
            page_count += 1
        self.buffer.flush(block_file)
        self._block_pages.append(page_count)

    # Recorre un bloque entrada por entrada sin cargarlo completo
    def _iter_block(self, block_no: int) -> Iterator[tuple[str, Postings]]:
        if self.buffer is None:
            yield from self._blocks[block_no]
            return
        block_file = self._block_file(block_no)
        carry = bytearray()
        for page_no in range(self._block_pages[block_no]):
            carry.extend(self.buffer.get(block_file, page_no).data)
            while True:
                cut = carry.find(b"\n")
                if cut < 0:
                    break
                line = bytes(carry[:cut])
                del carry[:cut + 1]
                if line:
                    item = json.loads(line.decode("utf-8"))
                    yield item["term"], item["postings"]
        if carry:
            item = json.loads(bytes(carry).decode("utf-8"))
            yield item["term"], item["postings"]

    def merge_blocks(self) -> dict[str, Postings]:
        iterators = [self._iter_block(block_no) for block_no in range(self._total_blocks())]
        heap: list[tuple[str, int, Postings]] = []
        for block_index, iterator in enumerate(iterators):
            first = next(iterator, None)
            if first is not None:
                heapq.heappush(heap, (first[0], block_index, first[1]))
        merged: dict[str, Postings] = {}
        while heap:
            term, block_index, postings = heapq.heappop(heap)
            accumulated = dict(postings)
            self._push_next(heap, iterators, block_index)
            while heap and heap[0][0] == term:
                _, same_block, same_postings = heapq.heappop(heap)
                for doc_id, frequency in same_postings.items():
                    accumulated[doc_id] = accumulated.get(doc_id, 0) + frequency
                self._push_next(heap, iterators, same_block)
            merged[term] = dict(sorted(accumulated.items()))
        return merged

    def _push_next(
        self,
        heap: list[tuple[str, int, Postings]],
        iterators: list[Iterator[tuple[str, Postings]]],
        block_index: int,
    ) -> None:
        entry = next(iterators[block_index], None)
        if entry is not None:
            heapq.heappush(heap, (entry[0], block_index, entry[1]))

    def blocks(self) -> list[Block]:
        self.flush()
        if self.buffer is None:
            return list(self._blocks)
        return [list(self._iter_block(block_no)) for block_no in range(len(self._block_pages))]

    def block_count(self) -> int:
        self.flush()
        return self._total_blocks()

    def _total_blocks(self) -> int:
        if self.buffer is None:
            return len(self._blocks)
        return len(self._block_pages)

    def _block_file(self, block_no: int) -> str:
        return f"{self.file_id}_block{block_no}"
