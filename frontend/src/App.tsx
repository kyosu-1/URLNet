import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Model {
  name: string;
  description: string;
}

function App() {
  const [url, setUrl] = useState<string>('');
  const [prediction, setPrediction] = useState<string>('');
  const [models, setModels] = useState<Model[]>([]);

  useEffect(() => {
    const getModels = async () => {
      try {
        const response = await axios.get<Model[]>('http://localhost:8000/api/models');
        setModels(response.data);
      } catch (error) {
        console.error(error);
      }
    };
    getModels();
  }, []);

  const checkUrl = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/predict', { url: url, model: models[0].name });
      setPrediction(response.data.is_malicious ? 'Malicious' : 'Not malicious');
    } catch (error) {
      console.error(error);
      setPrediction('Error checking URL');
    }
  };

  return (
    <div className="App">
      <input value={url} onChange={e => setUrl(e.target.value)} placeholder="Enter a URL" />
      <button onClick={checkUrl}>Check URL</button>
      <p>{prediction}</p>
      <ul>
        {models.map(model => <li key={model.name}>{model.name}: {model.description}</li>)}
      </ul>
    </div>
  );
}

export default App;
