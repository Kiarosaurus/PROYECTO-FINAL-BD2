from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


Postings = dict[str, int]
Block = list[tuple[str, Postings]]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class SPIMIBlockBuilder:
    block_document_limit: int = 128
    _current: dict[str, Postings] = field(default_factory=dict)
    _blocks: list[Block] = field(default_factory=list)
    _documents_in_block: int = 0

    def add_document(self, doc_id: str, text: str) -> None:
        for term in tokenize(text):
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
        self._blocks.append(block)
        self._current = {}
        self._documents_in_block = 0

    def build(self, documents: Iterable[tuple[str, str]]) -> dict[str, Postings]:
        for doc_id, text in documents:
            self.add_document(doc_id, text)
        self.flush()
        return self.merge_blocks()

    def merge_blocks(self) -> dict[str, Postings]:
        heap: list[tuple[str, int, int]] = []
        for block_index, block in enumerate(self._blocks):
            if block:
                heapq.heappush(heap, (block[0][0], block_index, 0))
        merged: dict[str, Postings] = {}
        while heap:
            term, block_index, item_index = heapq.heappop(heap)
            postings = dict(self._blocks[block_index][item_index][1])
            next_index = item_index + 1
            if next_index < len(self._blocks[block_index]):
                next_term = self._blocks[block_index][next_index][0]
                heapq.heappush(heap, (next_term, block_index, next_index))
            while heap and heap[0][0] == term:
                _, same_block, same_index = heapq.heappop(heap)
                for doc_id, frequency in self._blocks[same_block][same_index][1].items():
                    postings[doc_id] = postings.get(doc_id, 0) + frequency
                next_same = same_index + 1
                if next_same < len(self._blocks[same_block]):
                    next_term = self._blocks[same_block][next_same][0]
                    heapq.heappush(heap, (next_term, same_block, next_same))
            merged[term] = dict(sorted(postings.items()))
        return merged

    def blocks(self) -> list[Block]:
        self.flush()
        return list(self._blocks)

    def block_count(self) -> int:
        return len(self.blocks())
