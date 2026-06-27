import { fileUrl, type QueryResult } from "../lib/api";

const IMAGE_EXT = [".jpg", ".jpeg", ".png", ".gif", ".webp"];

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
  const items: string[] = [];
  for (const row of result.rows) {
    for (const cell of row) {
      if (isImage(cell)) {
        items.push(cell);
      }
    }
  }
  if (items.length === 0) {
    return null;
  }
  return (
    <div className="gallery">
      {items.map((path, i) => (
        <figure key={i} className="gallery-item">
          <img src={fileUrl(basename(path))} alt={basename(path)} />
          <figcaption>{basename(path)}</figcaption>
        </figure>
      ))}
    </div>
  );
}
