import React, { useState, useCallback } from 'react';
import { usePubNub } from 'pubnub-react';
import { CHANNEL } from '../index';
import Button from './Button';

const UnlockButton = () => {
  const pubnub = usePubNub();
  const [isUnlocking, setIsUnlocking] = useState(false);
  const [error, setError] = useState(null);

  const handleUnlock = useCallback(async () => {
    setIsUnlocking(true);
    setError(null);
    
    try {
      await pubnub.publish({
        channel: CHANNEL,
        message: {
          command: 'UNLOCK',
          timestamp: new Date().toISOString(),
          userId: 'web-user' // You might want to replace this with actual user ID
        }
      });
    } catch (error) {
      console.error('Failed to send unlock command:', error);
      setError('Failed to unlock door. Please try again.');
    } finally {
      setIsUnlocking(false);
    }
  }, [pubnub]);

  return (
    <div className="unlock-container">
      <Button
        onClick={handleUnlock}
        text={isUnlocking ? 'Unlocking...' : 'Unlock Door'}
        variant="primary"
        disabled={isUnlocking}
        className="unlock-button"
      />
      {error && (
        <div className="status-error">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default UnlockButton;