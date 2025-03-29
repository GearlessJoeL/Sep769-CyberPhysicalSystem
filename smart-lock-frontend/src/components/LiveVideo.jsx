// filepath: /smart-lock-frontend/smart-lock-frontend/src/components/LiveVideo.jsx
import React, { useEffect, useRef } from 'react';

const LiveVideo = () => {
  const videoRef = useRef(null);

  useEffect(() => {
    const getVideoStream = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Error accessing the camera', error);
      }
    };

    getVideoStream();

    // Cleanup function to stop video tracks when component unmounts
    return () => {
      if (videoRef.current) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div>
      <h2>Live Video Feed</h2>
      <video ref={videoRef} autoPlay playsInline />
    </div>
  );
};

export default LiveVideo;