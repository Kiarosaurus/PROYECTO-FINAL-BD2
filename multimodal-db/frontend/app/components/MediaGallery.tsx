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

// Scores que trae cada fila de una consulta HYBRID
type Scores = { fused: unknown; visual: unknown; text: unknown };

type GalleryItem = { path: string; scores?: Scores };

// Muestra un score con pocos decimales
// Usa un guion cuando el valor falta
function fmtScore(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function ScoreCaption({ scores }: { scores: Scores }) {
  return (
    <span className="gallery-scores">
      fusión {fmtScore(scores.fused)} · visual {fmtScore(scores.visual)} · texto{" "}
      {fmtScore(scores.text)}
    </span>
  );
}

export default function MediaGallery({ result }: { result: QueryResult }) {
  const [view, setView] = useState<View>("grid");
  const [shown, setShown] = useState(PAGE);
  const [selected, setSelected] = useState<string | null>(null);

  const fusedCol = result.columns.indexOf("fused_score");
  const visualCol = result.columns.indexOf("visual_score");
  const textCol = result.columns.indexOf("text_score");
  const hasScores = fusedCol !== -1 && visualCol !== -1 && textCol !== -1;

  const items: GalleryItem[] = [];
  for (const row of result.rows) {
    for (const cell of row) {
      // Una celda KNN trae una lista con el archivo y su score
      for (const value of Array.isArray(cell) ? cell : [cell]) {
        if (isImage(value)) {
          items.push(
            hasScores
              ? {
                  path: value,
                  scores: {
                    fused: row[fusedCol],
                    visual: row[visualCol],
                    text: row[textCol],
                  },
                }
              : { path: value },
          );
        }
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
          {visible.map((item, i) => (
            <figure
              key={i}
              className="gallery-item"
              onClick={() => setSelected(item.path)}
            >
              <img
                src={fileUrl(basename(item.path))}
                alt={basename(item.path)}
                loading="lazy"
              />
              <figcaption>
                {basename(item.path)}
                {item.scores && <ScoreCaption scores={item.scores} />}
              </figcaption>
            </figure>
          ))}
        </div>
      ) : (
        <ul className="media-rows">
          {visible.map((item, i) => (
            <li
              key={i}
              className="media-row"
              onClick={() => setSelected(item.path)}
            >
              <img
                src={fileUrl(basename(item.path))}
                alt={basename(item.path)}
                loading="lazy"
              />
              <span>
                {basename(item.path)}
                {item.scores && <ScoreCaption scores={item.scores} />}
              </span>
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
