import type { QueryResult } from "../lib/api";

export default function PlanInspector({
  result,
}: {
  sql: string;
  result: QueryResult;
}) {
  if (result.explain.length === 0) {
    return null;
  }
  return (
    <details className="inspector">
      <summary className="inspector-title">Query plan</summary>
      <div className="explain">
        {result.explain.map((line, i) => (
          <span
            key={i}
            className="explain-line"
            style={{ paddingLeft: `${line.depth * 2}ch` }}
          >
            {line.text}
          </span>
        ))}
      </div>
    </details>
  );
}
