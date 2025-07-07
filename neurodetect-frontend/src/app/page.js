import Link from 'next/link';
import Navbar from '../../components/Navbar';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-r from-blue-50 to-green-100 dark:from-gray-900 dark:to-gray-800">
      <Navbar />
      <section className="flex flex-col items-center justify-center h-[80vh] text-center px-4">
        <h1 className="text-5xl font-extrabold text-gray-800 dark:text-white mb-4">
          NeuroDetect
        </h1>
        <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
          AI-Powered Parkinson's Detection from Speech Features
        </p>
        <Link href="/predict" className="bg-blue-600 hover:bg-blue-700 text-white py-3 px-6 rounded-full shadow-lg transition">
          Start Diagnosis
        </Link>
      </section>
    </main>
  );
}

