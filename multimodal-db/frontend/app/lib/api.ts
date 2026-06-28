export interface ExplainLine {
  depth: number;
  text: string;
}

export interface QueryResult {
  columns: string[];
  rows: unknown[][];
  io: { disk_reads: number; disk_writes: number; pages_allocated: number };
  index_type: string | null;
  predicate_kind: string | null;
  elapsed_ms: number;
  explain: ExplainLine[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Arma la URL para descargar un archivo subido
export function fileUrl(name: string): string {
  return `${API_URL}/files/${encodeURIComponent(name)}`;
}

// Manda el SQL al backend y devuelve los resultados
export async function runQuery(sql: string): Promise<QueryResult> {
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "error desconocido");
  }
  return res.json();
}
