import { useState, useCallback, useEffect } from 'react';
import { usePubNub } from 'pubnub-react';
import { CHANNEL } from '../index';

export const useHistory = () => {
    const pubnub = usePubNub();
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchLogs = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const result = await pubnub.history({
                channel: CHANNEL,
                count: 100,
                stringifiedTimeToken: true
            });

            const formattedLogs = result.messages.map(item => ({
                id: item.timetoken,
                timestamp: new Date(parseInt(item.timetoken / 10000)).toISOString(),
                type: item.entry?.type || item.message?.type,
                success: item.entry?.success || item.message?.success,
                message: item.entry?.message || item.message?.message,
                userId: item.entry?.userId || item.message?.userId,
                ...(item.entry || item.message)
            }));

            setLogs(formattedLogs);
        } catch (error) {
            setError('Failed to fetch access logs: ' + error.message);
            console.error('Failed to fetch access logs:', error);
        } finally {
            setLoading(false);
        }
    }, [pubnub]);

    useEffect(() => {
        const handleMessage = (event) => {
            if (event.channel === CHANNEL && event.message.type === 'ACCESS_LOG') {
                setLogs(prevLogs => [{
                    id: event.timetoken,
                    timestamp: new Date(parseInt(event.timetoken / 10000)).toISOString(),
                    ...event.message
                }, ...prevLogs]);
            }
        };

        pubnub.addListener({
            message: handleMessage
        });

        // Initial fetch
        fetchLogs();

        return () => {
            pubnub.removeListener({ message: handleMessage });
        };
    }, [pubnub, fetchLogs]);

    const clearLogs = useCallback(() => {
        setLogs([]);
    }, []);

    return {
        logs,
        loading,
        error,
        fetchLogs,
        clearLogs
    };
};

export default useHistory;