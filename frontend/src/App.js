import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [userId] = useState(() => 'user_' + Math.random().toString(36).substr(2, 9));
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [userStatus, setUserStatus] = useState('idle');
  const [partnerId, setPartnerId] = useState(null);
  const [roomId, setRoomId] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Ready to connect');
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  
  const wsRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const localAudioRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const iceCandidatesQueue = useRef([]);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://');
      wsRef.current = new WebSocket(`${wsUrl}/ws/${userId}`);
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        setStatusMessage('Connected to voice chat roulette');
        console.log('WebSocket connected');
      };
      
      wsRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        setStatusMessage('Disconnected from server');
        console.log('WebSocket disconnected');
        
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatusMessage('Connection error');
      };
    };
    
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [userId]);

  // Handle WebSocket messages
  const handleWebSocketMessage = (message) => {
    console.log('Received message:', message);
    
    switch (message.type) {
      case 'connected':
        setStatusMessage(message.message);
        break;
      case 'searching':
        setUserStatus('searching');
        setStatusMessage(message.message);
        break;
      case 'search_stopped':
        setUserStatus('idle');
        setStatusMessage(message.message);
        break;
      case 'match_found':
        setUserStatus('connected');
        setPartnerId(message.partner_id);
        setRoomId(message.room_id);
        setStatusMessage(message.message);
        initializeVoiceChat();
        break;
      case 'partner_disconnected':
        setUserStatus('idle');
        setPartnerId(null);
        setRoomId(null);
        setStatusMessage(message.message);
        cleanupVoiceChat();
        break;
      case 'webrtc_signal':
        handleWebRTCSignal(message.signal);
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  // Send message to WebSocket
  const sendMessage = (message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  };

  // Initialize voice chat
  const initializeVoiceChat = async () => {
    try {
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setLocalStream(stream);
      
      if (localAudioRef.current) {
        localAudioRef.current.srcObject = stream;
      }
      
      // Create peer connection
      const peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ]
      });
      
      peerConnectionRef.current = peerConnection;
      
      // Add local stream to peer connection
      stream.getTracks().forEach(track => {
        peerConnection.addTrack(track, stream);
      });
      
      // Handle remote stream
      peerConnection.ontrack = (event) => {
        const [remoteStream] = event.streams;
        setRemoteStream(remoteStream);
        
        if (remoteAudioRef.current) {
          remoteAudioRef.current.srcObject = remoteStream;
        }
      };
      
      // Handle ICE candidates
      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          sendMessage({
            type: 'webrtc_signal',
            signal: {
              type: 'ice-candidate',
              candidate: event.candidate
            }
          });
        }
      };
      
      // Process queued ICE candidates
      iceCandidatesQueue.current.forEach(candidate => {
        peerConnection.addIceCandidate(candidate);
      });
      iceCandidatesQueue.current = [];
      
      setIsVoiceEnabled(true);
      
    } catch (error) {
      console.error('Error initializing voice chat:', error);
      setStatusMessage('Error accessing microphone');
    }
  };

  // Handle WebRTC signaling
  const handleWebRTCSignal = async (signal) => {
    const peerConnection = peerConnectionRef.current;
    
    if (!peerConnection) return;
    
    try {
      if (signal.type === 'offer') {
        await peerConnection.setRemoteDescription(signal.offer);
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        
        sendMessage({
          type: 'webrtc_signal',
          signal: {
            type: 'answer',
            answer: answer
          }
        });
      } else if (signal.type === 'answer') {
        await peerConnection.setRemoteDescription(signal.answer);
      } else if (signal.type === 'ice-candidate') {
        if (peerConnection.remoteDescription) {
          await peerConnection.addIceCandidate(signal.candidate);
        } else {
          iceCandidatesQueue.current.push(signal.candidate);
        }
      }
    } catch (error) {
      console.error('Error handling WebRTC signal:', error);
    }
  };

  // Create offer for WebRTC
  const createOffer = async () => {
    const peerConnection = peerConnectionRef.current;
    
    if (!peerConnection) return;
    
    try {
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);
      
      sendMessage({
        type: 'webrtc_signal',
        signal: {
          type: 'offer',
          offer: offer
        }
      });
    } catch (error) {
      console.error('Error creating offer:', error);
    }
  };

  // Cleanup voice chat
  const cleanupVoiceChat = () => {
    if (localStream) {
      localStream.getTracks().forEach(track => track.stop());
      setLocalStream(null);
    }
    
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }
    
    setRemoteStream(null);
    setIsVoiceEnabled(false);
  };

  // Control functions
  const startSearch = () => {
    sendMessage({ type: 'start_search' });
  };

  const stopSearch = () => {
    sendMessage({ type: 'stop_search' });
  };

  const nextPartner = () => {
    sendMessage({ type: 'next_partner' });
  };

  // Get status color
  const getStatusColor = () => {
    switch (userStatus) {
      case 'searching': return 'text-yellow-400';
      case 'connected': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  // Get connection status color
  const getConnectionColor = () => {
    return connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-2xl shadow-2xl p-8 w-full max-w-md border border-gray-700">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Voice Chat Roulette</h1>
          <p className="text-gray-400">Connect with random people worldwide</p>
        </div>

        {/* Status Display */}
        <div className="bg-gray-700 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Connection:</span>
            <span className={`text-sm font-semibold ${getConnectionColor()}`}>
              {connectionStatus}
            </span>
          </div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Status:</span>
            <span className={`text-sm font-semibold ${getStatusColor()}`}>
              {userStatus}
            </span>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-300">{statusMessage}</p>
          </div>
        </div>

        {/* Voice Chat Indicators */}
        {isVoiceEnabled && (
          <div className="bg-gray-700 rounded-lg p-4 mb-6">
            <h3 className="text-white font-semibold mb-2">Voice Chat Active</h3>
            <div className="flex justify-between items-center">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-300">Your mic</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-300">Partner's voice</span>
              </div>
            </div>
          </div>
        )}

        {/* Control Buttons */}
        <div className="space-y-4">
          {userStatus === 'idle' && (
            <button
              onClick={startSearch}
              disabled={connectionStatus !== 'connected'}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
            >
              🔍 Start Search
            </button>
          )}

          {userStatus === 'searching' && (
            <button
              onClick={stopSearch}
              className="w-full bg-gradient-to-r from-red-500 to-pink-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-red-600 hover:to-pink-700 transition-all duration-200 transform hover:scale-105"
            >
              ⏹️ Stop Search
            </button>
          )}

          {userStatus === 'connected' && (
            <div className="space-y-3">
              <button
                onClick={createOffer}
                className="w-full bg-gradient-to-r from-green-500 to-emerald-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-green-600 hover:to-emerald-700 transition-all duration-200 transform hover:scale-105"
              >
                🎙️ Start Voice Chat
              </button>
              <button
                onClick={nextPartner}
                className="w-full bg-gradient-to-r from-orange-500 to-red-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-orange-600 hover:to-red-700 transition-all duration-200 transform hover:scale-105"
              >
                ⏭️ Next Partner
              </button>
            </div>
          )}
        </div>

        {/* Audio Elements */}
        <audio ref={localAudioRef} muted={true} />
        <audio ref={remoteAudioRef} autoPlay={true} />

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-xs text-gray-500">
            Anonymous • Secure • Global
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;