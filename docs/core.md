# Storage y Buffer (Core)

Este documento resume lo implementado por Dev A: el `StorageEngine`, el `BufferManager` y el formato físico de página en disco. Todos los detalles de comportamiento están cubiertos por `tests/test_storage_contract.py` y `tests/test_postgres_storage_integration.py`.

## Storage Engine

Contrato: `core/ports/storage.py` (`read_page`, `write_page`, `allocate_page`, `stats`).

Implementaciones:

- `FileStorageEngine` (`core/storage/file_engine.py` + `core/storage/heap_file.py`): heap file paginado sobre archivos locales.
- `PostgresStorageEngine` (`core/storage/postgres_engine.py`): mismo contrato sobre la tabla `engine.page` de Postgres.

Invariantes que cumplen ambas implementaciones:

- Leer una página nunca escrita devuelve `b""`, nunca lanza error.
- Escribir una página sin haber llamado antes `allocate_page` funciona igual: `write_page` no exige asignación previa.
- `allocate_page` devuelve números crecientes por `file_id`.
- Las páginas de distintos `file_id` están completamente aisladas entre sí.
- `stats()` devuelve siempre el mismo objeto `IOStats`, actualizado en cada lectura, escritura y asignación reales.

### Cuándo usar cada uno

| Escenario | Motor recomendado |
| --- | --- |
| Desarrollo local, benchmarks de la Fase 4 contra el motor propio | `FileStorageEngine` |
| Persistir el estado del engine junto al schema `engine` que usa Postgres/pgvector | `PostgresStorageEngine` |

## Buffer Manager

Contrato: `core/ports/buffer.py` (`get`, `pin`, `flush`). Implementación: `LRUBufferManager` (`core/buffer/lru_buffer.py`), que además expone `unpin` y `stats`. Ninguno de los dos forma parte del ABC, pero son necesarios para que el pin/unpin sea real y para no duplicar contadores de `IOStats`.

Invariantes:

- `get` solo llama a `StorageEngine.read_page` en un cache miss real. Un hit de cache nunca incrementa `disk_reads`.
- Una página con `pin_count > 0` nunca se desaloja, sin importar qué tan vieja sea según el orden LRU.
- La página recién traída por `get` tampoco se puede desalojar en esa misma llamada, incluso si el buffer ya está lleno.
- Si no hay ninguna página desalojable (todas fijadas) y el buffer sigue por encima de su capacidad, se lanza `RuntimeError` en vez de crecer sin límite.
- `flush` solo escribe páginas con `dirty = True`.
- `stats()` del buffer devuelve el mismo `IOStats` que su `StorageEngine`, no lleva un contador aparte.

### Write-back por lotes

Al desalojar páginas para hacer espacio, primero se eligen todas las víctimas (respetando `pin` y la protección de la página recién traída) y recién después se escriben juntas, ordenadas por `(file_id, page_no)`. `flush()` usa el mismo mecanismo interno. Esto evita escribir una por una intercalado con la búsqueda de más candidatos, agrupando las escrituras a un mismo archivo.

## Formato de página en disco (FileStorageEngine)

Cada `file_id` usa tres archivos dentro del directorio base:

- `{file_id}.heap`: los bytes de las páginas, uno detrás de otro.
- `{file_id}.dir`: un arreglo de entradas de largo fijo (`core/storage/page_layout.py::DIR_ENTRY`, formato `<QII`) con offset en el heap, capacidad reservada y largo real. La entrada de la página número `N` vive en el byte `N * DIR_ENTRY.size` de este archivo.
- `{file_id}.free`: huecos reusables (`FREE_ENTRY`, formato `<QI`) con offset y capacidad. Se llena cuando una página crece y se muda a otro lugar del heap.

Regla de escritura: si el dato nuevo entra en la capacidad ya reservada, se sobreescribe en el mismo lugar. Si no entra, primero se busca en la free-list el hueco más ajustado que alcance (best-fit); si no hay ninguno, se agrega al final del archivo de datos.

## Formato de página en Postgres (PostgresStorageEngine)

Tabla `engine.page` (`docker/postgres/init.sql`): `(file_id TEXT, page_no INTEGER, data BYTEA)`, con `PRIMARY KEY (file_id, page_no)`. `allocate_page` calcula `COALESCE(MAX(page_no), -1) + 1` dentro del mismo `INSERT`, sin necesitar una tabla contadora aparte. `write_page` usa `INSERT ... ON CONFLICT DO UPDATE`.

## Formato de registros (DynamicRecord)

`core/record.py` define `Schema` (columnas tipadas: `INTEGER`, `FLOAT`, `BOOLEAN`, `VARCHAR`, `BLOB`) y `DynamicRecord`, que empaqueta cada fila con `struct`. Cada campo lleva un byte de nulo al inicio; si no es nulo, los campos de largo fijo usan su formato de `struct` y los de largo variable (`VARCHAR`, `BLOB`) llevan su largo antes del contenido. `VARCHAR` exige `max_length` en bytes utf-8 y lanza error si se excede.

## Nota para la integración

Hoy los índices (`BPlusTreeIndex`, etc.) llaman `storage.stats()` directamente, sin pasar por el `BufferManager`. Cuando se conecte todo en la integración final, los accesos a disco deberían venir del `BufferManager` para aprovechar el cache LRU y el write-back por lotes en vez de ir directo al `StorageEngine`.
