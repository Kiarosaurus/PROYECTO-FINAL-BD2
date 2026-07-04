from __future__ import annotations

import struct

# Una entrada de directorio: offset en el archivo de datos, capacidad reservada y largo real
DIR_ENTRY = struct.Struct("<QII")

# Una entrada libre: offset y capacidad de un hueco reusable
FREE_ENTRY = struct.Struct("<QI")
