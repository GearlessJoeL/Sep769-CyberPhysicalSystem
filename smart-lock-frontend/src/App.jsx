import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import LiveVideo from './components/LiveVideo';
import FacialRecognition from './components/FacialRecognition';
import FingerPrint from './components/FingerPrint';
import NFCReader from './components/NFCReader';
import History from './components/History';
import UnlockButton from './components/UnlockButton';
import './index.css';

function App() {
  const [showHistory, setShowHistory] = useState(false);

  return (
    <div className={`app-container ${showHistory ? 'with-sidebar' : ''}`}>
      <div className="main-content">
        <div className="header-controls">
          <UnlockButton />
          <button 
            className="button history-button"
            onClick={() => setShowHistory(!showHistory)}
          >
            {showHistory ? 'Hide History' : 'Show History'}
          </button>
        </div>

        <div className="grid">
          <div className="video-container">
            <LiveVideo />
            <FacialRecognition />
          </div>

          <div className="status-container">
            <FingerPrint />
          </div>

          <div className="status-container">
            <NFCReader />
          </div>
        </div>
      </div>

      {showHistory && (
        <div className="sidebar">
          <History onClose={() => setShowHistory(false)} />
        </div>
      )}
    </div>
  );
}

export default App;