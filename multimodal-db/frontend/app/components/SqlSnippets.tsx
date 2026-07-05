type Snippet = { label: string; sql: string; hint?: string };

const SNIPPETS: Snippet[] = [
  { label: "Crear tabla", sql: "CREATE TABLE media (id INT, path TEXT, feat VECTOR)" },
  { label: "Crear index", sql: "CREATE INDEX ON media (id) USING hash" },
  {
    label: "Insertar",
    sql: 'INSERT INTO media (id, path) VALUES (1, "a.jpg")',
    hint: "Si a.jpg no fue subido, la galería muestra una plantilla demo en su lugar",
  },
  { label: "Select todo", sql: "SELECT * FROM media" },
  { label: "Rango", sql: "SELECT * FROM media WHERE id BETWEEN 1 AND 9" },
  { label: "KNN", sql: "SELECT * FROM media WHERE KNN(feat, [0.1, 0.2, 0.3], 5)" },
  { label: "Espacial", sql: "SELECT * FROM media WHERE WITHIN(box, [0, 0], [10, 10])" },
  {
    label: "MATCH texto",
    sql: 'SELECT * FROM songs WHERE MATCH(lyrics, "corazón noche", 3)',
    hint: "Requiere correr el seed (tests/seed_demo.py)",
  },
  {
    label: "KNN imagen",
    sql: 'SELECT * FROM photos WHERE KNN(img, "demo_query.png", 5)',
    hint: "Requiere correr el seed (tests/seed_demo.py)",
  },
  {
    label: "KNN audio",
    sql: 'SELECT * FROM tracks WHERE KNN(audio, "demo_query.wav", 3)',
    hint: "Requiere correr el seed (tests/seed_demo.py)",
  },
];

export default function SqlSnippets({ onPick }: { onPick: (sql: string) => void }) {
  return (
    <div className="snippets">
      {SNIPPETS.map((snippet) => (
        <button
          key={snippet.label}
          className="snippet"
          title={snippet.hint}
          onClick={() => onPick(snippet.sql)}
        >
          {snippet.label}
        </button>
      ))}
    </div>
  );
}
