import React, { useState, useEffect } from 'react';
import PubNub from 'pubnub';

const RecognitionResult = ({ type }) => {
  const [lastActivity, setLastActivity] = useState(null);

  const formatMessage = (type, data) => {
    switch (type) {
      case 'remote':
        return {
          name: 'Remote Access',
          message: data.state === 1 ? 'Door unlocked remotely' : 'Door locked remotely',
          timestamp: new Date(data.time * 1000).toLocaleString()
        };
      // ... other cases
      default:
        return null;
    }
  };

  useEffect(() => {
    const pubnub = new PubNub({
      subscribeKey: 'sub-c-a6797b99-e665-4db1-b0ec-2cb77ad995ed',
      uuid: '321'
    });

    pubnub.subscribe({
      channels: ['MingyiHUO728']
    });

    pubnub.addListener({
      message: (msg) => {
        if (msg.message.message_type === 'status' && msg.message.type === type) {
          setLastActivity({
            name: msg.message.name,
            time: new Date(msg.message.time * 1000).toLocaleString(),
            state: msg.message.state
          });
        }
      }
    });

    return () => {
      pubnub.unsubscribe({
        channels: ['MingyiHUO728']
      });
    };
  }, [type]);

  return (
    <div className="recognition-result">
      {lastActivity ? (
        <>
          <p className="name">{lastActivity.name}</p>
          <p className="time">{lastActivity.time}</p>
          <p className={`status ${lastActivity.state === 1 ? 'success' : 'error'}`}>
            {lastActivity.state === 1 ? 'Unlock' : 'Block Access'}
          </p>
        </>
      ) : (
        <p>No recent activity</p>
      )}
    </div>
  );
};

export default RecognitionResult;