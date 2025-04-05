import React, { useState } from 'react';
import RecognitionResult from './components/RecognitionResult';
import History from './components/History';
import UnlockButton from './components/UnlockButton';
import './index.css';

function App() {
  const [showHistory, setShowHistory] = useState(false);

  return (
    <>
      <div className="top-bar">
        <h1 className="project-title">Smart Lock System</h1>
        <button 
          className="history-button"
          onClick={() => setShowHistory(!showHistory)}
        >
          {showHistory ? 'Hide History' : 'Show History'}
        </button>
      </div>
      <div className={`app-container ${showHistory ? 'with-sidebar' : ''}`}>
        <UnlockButton />
        <div className="recognitions-container">
          <div className="recognition-section">
            <h2>RFID Recognition</h2>
            <div className="status-display">
              <RecognitionResult type="rfid" />
            </div>
          </div>
          <div className="recognition-section">
            <h2>Face Recognition</h2>
            <div className="status-display">
              <RecognitionResult type="face" />
            </div>
          </div>
          {/* <div className="recognition-section">
            <h2>Fingerprint Recognition</h2>
            <div className="status-display">
              <RecognitionResult type="fingerprint" />
            </div>
          </div> */}
          <div className="recognition-section">
            <h2>Remote Control</h2>
            <div className="status-display">
              <RecognitionResult type="remote" />
            </div>
          </div>
        </div>

        {showHistory && (
          <div className="sidebar">
            <History onClose={() => setShowHistory(false)} />
          </div>
        )}
      </div>
    </>
  );
}

export default App;