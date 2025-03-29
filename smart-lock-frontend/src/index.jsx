import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import './index.css';
import { BrowserRouter } from 'react-router-dom';
import PubNub from 'pubnub';
import { PubNubProvider } from 'pubnub-react';

// Define single channel for all communication
export const CHANNEL = 'MingyiHUO728';

const pubnub = new PubNub({
  publishKey: process.env.REACT_APP_PUBNUB_PUBLISH_KEY || 'your-pub-key',
  subscribeKey: process.env.REACT_APP_PUBNUB_SUBSCRIBE_KEY || 'your-sub-key',
  userId: "smart-lock-system",
  ssl: true
});

// Subscribe to channel
pubnub.subscribe({
  channels: [CHANNEL]
});

ReactDOM.render(
  <React.StrictMode>
    <PubNubProvider client={pubnub}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </PubNubProvider>
  </React.StrictMode>,
  document.getElementById('root')
);