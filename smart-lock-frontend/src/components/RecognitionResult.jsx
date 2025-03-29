import React, { useState, useEffect } from 'react';

const RecognitionResult = () => {
  const [recognitionData, setRecognitionData] = useState({
    success: false,
    type: '',
    name: '',
    timestamp: ''
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchRecognitionData = async () => {
      try {
        setLoading(true);
        // Replace with your actual API endpoint
        const response = await fetch('http://localhost:5000/api/recognition');
        const data = await response.json();
        setRecognitionData(data);
      } catch (error) {
        console.error('Error fetching recognition data', error);
      } finally {
        setLoading(false);
      }
    };

    // Fetch initially
    fetchRecognitionData();

    // Set up interval to poll for updates (every 2 seconds)
    const intervalId = setInterval(fetchRecognitionData, 2000);

    // Cleanup function to clear interval when component unmounts
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="recognition-result">
      <h2>Recognition Result</h2>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <div className={`result-card ${recognitionData.success ? 'success' : 'failure'}`}>
          <p className="status">Status: {recognitionData.success ? 'Success' : 'Failed'}</p>
          {recognitionData.type && <p className="type">Method: {recognitionData.type}</p>}
          {recognitionData.name && <p className="name">User: {recognitionData.name}</p>}
          {recognitionData.timestamp && <p className="time">Time: {new Date(recognitionData.timestamp).toLocaleString()}</p>}
        </div>
      )}
    </div>
  );
};

export default RecognitionResult; 