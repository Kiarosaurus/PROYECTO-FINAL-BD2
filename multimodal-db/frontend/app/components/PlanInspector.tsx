import type { QueryResult } from "../lib/api";

export default function PlanInspector({
  sql,
  result,
}: {
  sql: string;
  result: QueryResult;
}) {
  const plan = {
    sql,
    index_type: result.index_type,
    predicate_kind: result.predicate_kind,
    elapsed_ms: result.elapsed_ms,
    io: result.io,
    columns: result.columns,
    rows: result.rows.length,
  };
  return (
    <details className="inspector">
      <summary>Inspector del plan</summary>
      <pre className="inspector-body">{JSON.stringify(plan, null, 2)}</pre>
    </details>
  );
}
