"use client";

import { useEffect, useState } from "react";
import { fileUrl, type QueryResult } from "../lib/api";

const IMAGE_EXT = [".jpg", ".jpeg", ".png", ".gif", ".webp"];

// Cuántas imágenes se muestran antes de pedir ver más
const PAGE = 50;

type View = "grid" | "list";

// Dice si el valor parece el nombre de una imagen
function isImage(value: unknown): value is string {
  return (
    typeof value === "string" &&
    IMAGE_EXT.some((ext) => value.toLowerCase().endsWith(ext))
  );
}

// Deja solo el nombre del archivo sin la ruta
function basename(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1];
}

export default function MediaGallery({ result }: { result: QueryResult }) {
  const [view, setView] = useState<View>("grid");
  const [shown, setShown] = useState(PAGE);
  const [selected, setSelected] = useState<string | null>(null);

  const items: string[] = [];
  for (const row of result.rows) {
    for (const cell of row) {
      if (isImage(cell)) {
        items.push(cell);
      }
    }
  }

  // Cada consulta nueva reinicia el cap y cierra el lightbox
  useEffect(() => {
    setShown(PAGE);
    setSelected(null);
  }, [result]);

  if (items.length === 0) {
    return null;
  }

  const visible = items.slice(0, shown);

  return (
    <div className="media">
      <div className="media-toolbar">
        <span className="media-count">
          {visible.length} de {items.length} imágenes
        </span>
        <div className="media-views">
          <button
            className={view === "grid" ? "media-view on" : "media-view"}
            onClick={() => setView("grid")}
          >
            Mosaico
          </button>
          <button
            className={view === "list" ? "media-view on" : "media-view"}
            onClick={() => setView("list")}
          >
            Lista
          </button>
        </div>
      </div>

      {view === "grid" ? (
        <div className="gallery">
          {visible.map((path, i) => (
            <figure
              key={i}
              className="gallery-item"
              onClick={() => setSelected(path)}
            >
              <img
                src={fileUrl(basename(path))}
                alt={basename(path)}
                loading="lazy"
              />
              <figcaption>{basename(path)}</figcaption>
            </figure>
          ))}
        </div>
      ) : (
        <ul className="media-rows">
          {visible.map((path, i) => (
            <li
              key={i}
              className="media-row"
              onClick={() => setSelected(path)}
            >
              <img
                src={fileUrl(basename(path))}
                alt={basename(path)}
                loading="lazy"
              />
              <span>{basename(path)}</span>
            </li>
          ))}
        </ul>
      )}

      {shown < items.length && (
        <button
          className="media-more"
          onClick={() => setShown((n) => n + PAGE)}
        >
          Ver más ({items.length - shown} restantes)
        </button>
      )}

      {selected && (
        <div className="lightbox" onClick={() => setSelected(null)}>
          <figure className="lightbox-item" onClick={(e) => e.stopPropagation()}>
            <img src={fileUrl(basename(selected))} alt={basename(selected)} />
            <figcaption>{basename(selected)}</figcaption>
          </figure>
          <button className="lightbox-close" onClick={() => setSelected(null)}>
            Cerrar
          </button>
        </div>
      )}
    </div>
  );
}
