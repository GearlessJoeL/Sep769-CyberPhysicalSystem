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
  publishKey: process.env.REACT_APP_PUBNUB_PUBLISH_KEY || 'pub-c-e478cfb1-92ef-4faa-93cc-d1c4022ecb19',
  subscribeKey: process.env.REACT_APP_PUBNUB_SUBSCRIBE_KEY || 'sub-c-a6797b99-e665-4db1-b0ec-2cb77ad995ed',
  userId: "321",
  ssl: true
});

// Subscribe to channel
pubnub.subscribe({
  channels: [CHANNEL]
});

// Add listener for remote control events
pubnub.addListener({
  message: (event) => {
    if (event.message.type === 'remote') {
      // Handle remote control events
      console.log('Remote control action:', event.message);
    }
  }
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