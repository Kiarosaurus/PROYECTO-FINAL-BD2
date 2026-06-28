"use client";

import { useState } from "react";
import { runQuery, type QueryResult } from "../lib/api";
import ResultsTable from "./ResultsTable";
import MediaGallery from "./MediaGallery";
import AudioPlayer from "./AudioPlayer";
import MetricsPanel from "./MetricsPanel";
import SqlSnippets from "./SqlSnippets";

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

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      onRun();
    }
  }

  return (
    <section className="editor">
      <SqlSnippets onPick={setSql} />
      <textarea
        className="editor-input"
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        onKeyDown={onKeyDown}
        rows={6}
        spellCheck={false}
      />
      <div className="editor-actions">
        <button className="editor-run" onClick={onRun} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? "Ejecutando..." : "Ejecutar"}
        </button>
        <span className="editor-hint">Ctrl+Enter para ejecutar</span>
      </div>
      {error && <p className="editor-error">{error}</p>}
      {result && (
        <>
          <p className="editor-info">{result.rows.length} filas</p>
          <MetricsPanel result={result} />
          <ResultsTable result={result} />
          <MediaGallery result={result} />
          <AudioPlayer result={result} />
        </>
      )}
    </section>
  );
}
