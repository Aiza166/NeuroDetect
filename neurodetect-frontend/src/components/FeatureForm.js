import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

const defaultFeatures = {
  fo: '',
  fhi: '',
  flo: '',
};

export default function FeatureForm({ onPredict }) {
  const [features, setFeatures] = useState(defaultFeatures);

  const handleChange = (e) => {
    setFeatures({ ...features, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict(features);
  };

  return (
    <Card>
      <CardContent className="p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-1">MDVP:Fo(Hz)</label>
            <input type="number" name="fo" value={features.fo} onChange={handleChange}
              className="w-full px-3 py-2 rounded border bg-white dark:bg-gray-800 dark:border-gray-700 dark:text-white"
              required />
          </div>
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-1">MDVP:Fhi(Hz)</label>
            <input type="number" name="fhi" value={features.fhi} onChange={handleChange}
              className="w-full px-3 py-2 rounded border bg-white dark:bg-gray-800 dark:border-gray-700 dark:text-white"
              required />
          </div>
          <div>
            <label className="block text-gray-700 dark:text-gray-300 mb-1">MDVP:Flo(Hz)</label>
            <input type="number" name="flo" value={features.flo} onChange={handleChange}
              className="w-full px-3 py-2 rounded border bg-white dark:bg-gray-800 dark:border-gray-700 dark:text-white"
              required />
          </div>
          <Button type="submit" className="w-full">Predict</Button>
        </form>
      </CardContent>
    </Card>
  );
}

