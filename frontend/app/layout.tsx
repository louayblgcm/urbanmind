import "leaflet/dist/leaflet.css";
import "./globals.css";

export const metadata = {
  title: "Urban Mind",
  description: "Map-based urban safety and activity intelligence",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
