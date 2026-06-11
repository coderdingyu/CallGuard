import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CallGuard",
  description: "Call risk awareness for speech pressure and fraud signals"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

