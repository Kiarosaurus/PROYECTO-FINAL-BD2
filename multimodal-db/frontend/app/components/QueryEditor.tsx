"use client";

import { useState } from "react";
import { runQuery, type QueryResult } from "../lib/api";
import ResultsTable from "./ResultsTable";

export default function QueryEditor() {
  const [sql, setSql] = useState("SELECT * FROM img");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onRun() {
    setLoading(true);
    setError(null);
    try {
      const data = await runQuery(sql);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "error");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="editor">
      <textarea
        className="editor-input"
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        rows={6}
        spellCheck={false}
      />
      <button className="editor-run" onClick={onRun} disabled={loading}>
        {loading ? "Ejecutando..." : "Ejecutar"}
      </button>
      {error && <p className="editor-error">{error}</p>}
      {result && (
        <>
          <p className="editor-info">{result.rows.length} filas</p>
          <ResultsTable result={result} />
        </>
      )}
    </section>
  );
}
