import { fileUrl, type QueryResult } from "../lib/api";

const AUDIO_EXT = [".mp3", ".wav", ".ogg", ".flac", ".m4a"];

// Dice si el valor parece el nombre de un audio
function isAudio(value: unknown): value is string {
  return (
    typeof value === "string" &&
    AUDIO_EXT.some((ext) => value.toLowerCase().endsWith(ext))
  );
}

// Deja solo el nombre del archivo sin la ruta
function basename(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1];
}

export default function AudioPlayer({ result }: { result: QueryResult }) {
  const items: string[] = [];
  for (const row of result.rows) {
    for (const cell of row) {
      if (isAudio(cell)) {
        items.push(cell);
      }
    }
  }
  if (items.length === 0) {
    return null;
  }
  return (
    <div className="audio-list">
      {items.map((path, i) => (
        <figure key={i} className="audio-item">
          <figcaption>{basename(path)}</figcaption>
          <audio controls src={fileUrl(basename(path))} />
        </figure>
      ))}
    </div>
  );
}
