from __future__ import annotations

from lark import Transformer

from query.ports import Parser
from query.parser.lexer import SqlLexer
from query.parser import ast as A


# Convierte el árbol de lark en nodos AST
class _AstBuilder(Transformer):

    def start(self, items):
        return items[0]

    def column_def(self, items):
        return A.ColumnDef(name=str(items[0]), type=str(items[1]).upper())

    def create_table(self, items):
        return A.CreateTable(table=str(items[0]), columns=list(items[1:]))

    def drop_table(self, items):
        return A.DropTable(table=str(items[0]))

    def create_index(self, items):
        return A.CreateIndex(
            table=str(items[0]),
            column=str(items[1]),
            index_type=str(items[2]),
        )

    def column_list(self, items):
        return [str(name) for name in items]

    def row(self, items):
        return tuple(items)

    def insert(self, items):
        columns: list[str] = []
        rows: list[tuple] = []
        for part in items[1:]:
            if isinstance(part, list):
                columns = part
            elif isinstance(part, tuple):
                rows.append(part)
        return A.Insert(table=str(items[0]), columns=columns, rows=rows)

    def delete(self, items):
        where = items[1] if len(items) > 1 else None
        return A.Delete(table=str(items[0]), where=where)

    def all_columns(self, items):
        return ["*"]

    def columns(self, items):
        return [str(name) for name in items]

    def where_clause(self, items):
        return items[0]

    def comparison(self, items):
        return A.Comparison(column=str(items[0]), op=str(items[1]), value=items[2])

    def between(self, items):
        return A.Between(column=str(items[0]), low=items[1], high=items[2])

    def vector(self, items):
        return list(items)

    def knn_file(self, items):
        return str(items[0])[1:-1]

    def knn_predicate(self, items):
        return A.KnnCondition(column=str(items[0]), query=items[1], k=int(items[2]))

    def spatial_predicate(self, items):
        return A.SpatialCondition(
            column=str(items[0]),
            min_corner=items[1],
            max_corner=items[2],
        )

    def match_predicate(self, items):
        terms = str(items[1])[1:-1]
        k = int(items[2]) if len(items) > 2 else None
        return A.MatchCondition(column=str(items[0]), terms=terms, k=k)

    def hybrid_predicate(self, items):
        return A.HybridCondition(
            column=str(items[0]),
            media_file=str(items[1])[1:-1],
            text_column=str(items[2]),
            terms=str(items[3])[1:-1],
            k=int(items[4]),
        )

    def limit_clause(self, items):
        return int(items[0])

    def select(self, items):
        columns: list[str] = []
        table = ""
        where = None
        limit = None
        for part in items:
            if isinstance(part, list):
                columns = part
            elif isinstance(part, A.Condition):
                where = part
            elif isinstance(part, int):
                limit = part
            else:
                table = str(part)
        return A.Select(table=table, columns=columns, where=where, limit=limit)

    # Quita el signo y decide entre entero y decimal
    def number(self, items):
        text = str(items[0])
        return float(text) if "." in text else int(text)

    # Saca las comillas del texto
    def string(self, items):
        return str(items[0])[1:-1]


# Lee texto SQL y devuelve el AST
class SqlParser(Parser):

    def __init__(self) -> None:
        self._lexer = SqlLexer()
        self._builder = _AstBuilder()

    def parse(self, sql: str):
        tree = self._lexer.parse_tree(sql)
        return self._builder.transform(tree)
