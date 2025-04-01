import React, { useState } from 'react';
import { usePubNub } from 'pubnub-react';
import { CHANNEL } from '../index';

const UnlockButton = () => {
  const pubnub = usePubNub();
  const [isUnlocking, setIsUnlocking] = useState(false);

  const handleUnlock = async () => {
    setIsUnlocking(true);
    try {
      await pubnub.publish({
        channel: CHANNEL,
        message: {
          message_type: "control",
          action: "unlock",
          timestamp: new Date().toISOString()
        }
      });
    } catch (error) {
      console.error('Failed to send unlock command:', error);
    } finally {
      setIsUnlocking(false);
    }
  };

  return (
    <button 
      onClick={handleUnlock}
      disabled={isUnlocking}
      className="unlock-button"
    >
      {isUnlocking ? 'Unlocking...' : 'Unlock Door'}
    </button>
  );
};

export default UnlockButton;