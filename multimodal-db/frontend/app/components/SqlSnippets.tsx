const SNIPPETS: { label: string; sql: string }[] = [
  { label: "Crear tabla", sql: "CREATE TABLE media (id INT, path TEXT, feat VECTOR)" },
  { label: "Crear index", sql: "CREATE INDEX ON media (id) USING hash" },
  { label: "Insertar", sql: 'INSERT INTO media (id, path) VALUES (1, "a.jpg")' },
  { label: "Select todo", sql: "SELECT * FROM media" },
  { label: "Rango", sql: "SELECT * FROM media WHERE id BETWEEN 1 AND 9" },
  { label: "KNN", sql: "SELECT * FROM media WHERE KNN(feat, [0.1, 0.2, 0.3], 5)" },
  { label: "Espacial", sql: "SELECT * FROM media WHERE WITHIN(box, [0, 0], [10, 10])" },
];

export default function SqlSnippets({ onPick }: { onPick: (sql: string) => void }) {
  return (
    <div className="snippets">
      {SNIPPETS.map((snippet) => (
        <button
          key={snippet.label}
          className="snippet"
          onClick={() => onPick(snippet.sql)}
        >
          {snippet.label}
        </button>
      ))}
    </div>
  );
}
