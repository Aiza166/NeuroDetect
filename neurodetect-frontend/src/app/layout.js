import '../styles/globals.css';

export const metadata = {
  title: 'NeuroDetect',
  description: 'AI-Powered Parkinson’s Detection',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
