from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class FieldType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    VARCHAR = auto()
    BLOB = auto()


_FIXED_FORMATS: dict[FieldType, str] = {
    FieldType.INTEGER: "<q",
    FieldType.FLOAT: "<d",
    FieldType.BOOLEAN: "<?",
}

_NULL_FLAG = "<B"
_VARCHAR_LEN = "<H"
_BLOB_LEN = "<I"


@dataclass(frozen=True)
class Field:
    name: str
    type: FieldType
    # Solo aplica a VARCHAR, es el máximo de bytes en utf-8
    max_length: int | None = None

    def __post_init__(self) -> None:
        if self.type == FieldType.VARCHAR and not self.max_length:
            raise ValueError(f"el campo '{self.name}' es VARCHAR y necesita max_length")


@dataclass(frozen=True)
class Schema:
    name: str
    fields: tuple[Field, ...]

    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def get_field(self, name: str) -> Field:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"el schema '{self.name}' no tiene el campo '{name}'")

    # Calcula el tamaño máximo de una fila si los varchars vienen llenos
    def max_record_size(self) -> int:
        total = 0
        for field in self.fields:
            total += struct.calcsize(_NULL_FLAG)
            if field.type == FieldType.VARCHAR:
                total += struct.calcsize(_VARCHAR_LEN) + field.max_length
            elif field.type == FieldType.BLOB:
                total += struct.calcsize(_BLOB_LEN)
            else:
                total += struct.calcsize(_FIXED_FORMATS[field.type])
        return total


class DynamicRecord:

    __slots__ = ("schema", "values")

    def __init__(self, schema: Schema, values: dict[str, Any]) -> None:
        self.schema = schema
        self.values = values

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.values[name] = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DynamicRecord):
            return NotImplemented
        return self.schema == other.schema and self.values == other.values

    def __repr__(self) -> str:
        return f"DynamicRecord({self.values!r})"

    def pack(self) -> bytes:
        chunks: list[bytes] = [_pack_field(field, self.values.get(field.name)) for field in self.schema.fields]
        return b"".join(chunks)

    @classmethod
    def unpack(cls, schema: Schema, data: bytes, offset: int = 0) -> "DynamicRecord":
        values: dict[str, Any] = {}
        for field in schema.fields:
            value, offset = _unpack_field(field, data, offset)
            values[field.name] = value
        return cls(schema, values)


# Cada campo lleva un byte que dice si el valor es nulo
def _pack_field(field: Field, value: Any) -> bytes:
    if value is None:
        return struct.pack(_NULL_FLAG, 0)
    flag = struct.pack(_NULL_FLAG, 1)
    if field.type == FieldType.VARCHAR:
        encoded = str(value).encode("utf-8")
        if len(encoded) > field.max_length:
            raise ValueError(f"'{field.name}' excede max_length de {field.max_length} bytes")
        return flag + struct.pack(_VARCHAR_LEN, len(encoded)) + encoded
    if field.type == FieldType.BLOB:
        encoded = bytes(value)
        return flag + struct.pack(_BLOB_LEN, len(encoded)) + encoded
    return flag + struct.pack(_FIXED_FORMATS[field.type], value)


def _unpack_field(field: Field, data: bytes, offset: int) -> tuple[Any, int]:
    (is_present,) = struct.unpack_from(_NULL_FLAG, data, offset)
    offset += struct.calcsize(_NULL_FLAG)
    if not is_present:
        return None, offset
    if field.type == FieldType.VARCHAR:
        (length,) = struct.unpack_from(_VARCHAR_LEN, data, offset)
        offset += struct.calcsize(_VARCHAR_LEN)
        value = data[offset:offset + length].decode("utf-8")
        return value, offset + length
    if field.type == FieldType.BLOB:
        (length,) = struct.unpack_from(_BLOB_LEN, data, offset)
        offset += struct.calcsize(_BLOB_LEN)
        value = bytes(data[offset:offset + length])
        return value, offset + length
    fmt = _FIXED_FORMATS[field.type]
    (value,) = struct.unpack_from(fmt, data, offset)
    return value, offset + struct.calcsize(fmt)
