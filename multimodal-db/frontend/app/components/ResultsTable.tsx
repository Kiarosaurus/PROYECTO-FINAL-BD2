import type { QueryResult } from "../lib/api";

// Pasa cualquier valor a texto para mostrarlo
function formatCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export default function ResultsTable({ result }: { result: QueryResult }) {
  if (result.columns.length === 0) {
    return <p className="editor-info">La consulta no devolvió filas.</p>;
  }
  return (
    <table className="results">
      <thead>
        <tr>
          {result.columns.map((col) => (
            <th key={col}>{col}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {result.rows.map((row, i) => (
          <tr key={i}>
            {row.map((cell, j) => (
              <td key={j}>{formatCell(cell)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
