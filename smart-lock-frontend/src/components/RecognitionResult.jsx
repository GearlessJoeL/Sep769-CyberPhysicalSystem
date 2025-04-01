import React, { useState, useEffect } from 'react';
import { usePubNub } from 'pubnub-react';
import { CHANNEL } from '../index';

const RecognitionResult = ({ type }) => {
  const [status, setStatus] = useState(null);
  const pubnub = usePubNub();

  useEffect(() => {
    const handleMessage = (event) => {
      const message = event.message;
      if (message.type === type) {
        setStatus({
          success: message.state === 1,
          name: message.name,
          time: message.time
        });
      }
    };

    pubnub.addListener({ message: handleMessage });

    return () => {
      pubnub.removeListener({ message: handleMessage });
    };
  }, [pubnub, type]);

  return (
    <div className={`status-container ${status?.success ? 'status-success' : 'status-error'}`}>
      {status ? (
        <>
          <div className="status-text">
            {status.success ? 'Authentication Successful' : 'Waiting for authentication...'}
          </div>
          {status.name && <div className="user-name">{status.name}</div>}
        </>
      ) : (
        <div className="status-text">Waiting for authentication...</div>
      )}
    </div>
  );
};

export default RecognitionResult;