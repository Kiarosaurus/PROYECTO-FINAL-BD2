from __future__ import annotations

from pathlib import Path

from lark import Lark, Token, Tree

_GRAMMAR_PATH = Path(__file__).with_name("grammar.lark")


# Carga la grammar y arma los parsers de lark una sola vez
class SqlLexer:

    def __init__(self) -> None:
        grammar = _GRAMMAR_PATH.read_text(encoding="utf-8")
        self._parser = Lark(grammar, start="start", parser="earley")
        self._lexer = Lark(grammar, start="start", parser="earley", lexer="basic")

    # Parte el texto SQL en tokens
    def tokenize(self, sql: str) -> list[Token]:
        return list(self._lexer.lex(sql))

    # Devuelve el árbol crudo de lark
    def parse_tree(self, sql: str) -> Tree:
        return self._parser.parse(sql)
