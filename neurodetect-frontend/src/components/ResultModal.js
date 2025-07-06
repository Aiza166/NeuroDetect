import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';

export default function ResultModal({ result, onClose }) {
  const { detected, confidence } = result;

  return (
    <motion.div
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      className="fixed inset-0 flex items-center justify-center bg-black/50 z-50"
    >
      <div className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg max-w-sm w-full text-center">
        <h2 className="text-2xl font-bold mb-4">
          {detected ? '⚠️ Parkinson Detected' : '✅ No Parkinson Detected'}
        </h2>
        <p className="text-lg mb-6">Confidence: {confidence}%</p>
        <Button onClick={onClose} className="w-full">Try Again</Button>
      </div>
    </motion.div>
  );
}

