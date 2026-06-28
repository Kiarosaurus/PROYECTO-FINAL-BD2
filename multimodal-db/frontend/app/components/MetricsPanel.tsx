import type { QueryResult } from "../lib/api";

export default function MetricsPanel({ result }: { result: QueryResult }) {
  const items: { label: string; value: string | number }[] = [
    { label: "Tiempo", value: `${result.elapsed_ms} ms` },
    { label: "Índice", value: result.index_type ?? "—" },
    { label: "Predicado", value: result.predicate_kind ?? "—" },
    { label: "Disk reads", value: result.io.disk_reads },
    { label: "Disk writes", value: result.io.disk_writes },
    { label: "Pages", value: result.io.pages_allocated },
  ];
  return (
    <div className="metrics">
      {items.map((item) => (
        <div key={item.label} className="metric">
          <span className="metric-label">{item.label}</span>
          <span className="metric-value">{item.value}</span>
        </div>
      ))}
    </div>
  );
}
