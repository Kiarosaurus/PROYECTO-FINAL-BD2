import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multimodal DB",
  description: "Interfaz del motor de búsqueda multimodal",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es">
      <body>
        <header className="topbar">
          <h1>Multimodal DB</h1>
        </header>
        <main className="content">{children}</main>
      </body>
    </html>
  );
}
