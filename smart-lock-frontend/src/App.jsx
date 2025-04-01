import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import RecognitionResult from './components/RecognitionResult';
import FacialRecognition from './components/FacialRecognition';
import FingerPrint from './components/FingerPrint';
import NFCReader from './components/NFCReader';
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
            <NFCReader />
          </div>
          <div className="recognition-section">
            <FacialRecognition />
          </div>
          <div className="recognition-section">
            <FingerPrint />
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