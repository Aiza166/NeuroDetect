'use client';
import { useState } from 'react';
import Navbar from '@/components/Navbar';
import FeatureForm from '@/components/FeatureForm';
import ResultModal from '@/components/ResultModal';

export default function PredictPage() {
  const [result, setResult] = useState(null);

  const handlePrediction = (data) => {
    // Placeholder: Fake Prediction
    const detected = Math.random() > 0.5;
    setResult({
      detected,
      confidence: Math.floor(Math.random() * 20 + 80),
    });
  };

  return (
    <main className="min-h-screen bg-gradient-to-r from-blue-50 to-green-100 dark:from-gray-900 dark:to-gray-800">
      <Navbar />
      <section className="max-w-xl mx-auto py-12 px-6">
        <h2 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">
          Enter Speech Features
        </h2>
        <FeatureForm onPredict={handlePrediction} />
        {result && (
          <ResultModal result={result} onClose={() => setResult(null)} />
        )}
      </section>
    </main>
  );
}

