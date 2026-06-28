"use client";

import { useEffect, useState } from "react";
import { fileUrl, type QueryResult } from "../lib/api";

const AUDIO_EXT = [".mp3", ".wav", ".ogg", ".flac", ".m4a"];

// Cuántos audios se muestran antes de pedir ver más
const PAGE = 50;

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
  const [shown, setShown] = useState(PAGE);

  const items: string[] = [];
  for (const row of result.rows) {
    for (const cell of row) {
      if (isAudio(cell)) {
        items.push(cell);
      }
    }
  }

  // Cada consulta nueva reinicia el cap
  useEffect(() => {
    setShown(PAGE);
  }, [result]);

  if (items.length === 0) {
    return null;
  }

  const visible = items.slice(0, shown);

  return (
    <div className="audio-list">
      {visible.map((path, i) => (
        <figure key={i} className="audio-item">
          <figcaption>{basename(path)}</figcaption>
          <audio controls src={fileUrl(basename(path))} />
        </figure>
      ))}
      {shown < items.length && (
        <button
          className="media-more"
          onClick={() => setShown((n) => n + PAGE)}
        >
          Ver más ({items.length - shown} restantes)
        </button>
      )}
    </div>
  );
}
