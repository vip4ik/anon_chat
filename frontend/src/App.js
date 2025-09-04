import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import "./favicon.ico";

// Защита от копирования
const disableCopy = (e) => {
  e.preventDefault();
  return false;
};

// Защита от контекстного меню
const disableContextMenu = (e) => {
  e.preventDefault();
  return false;
};

// Защита от перетаскивания
const disableDrag = (e) => {
  e.preventDefault();
  return false;
};

// Защита от выделения текста
const disableSelection = (e) => {
  e.preventDefault();
  return false;
};

// Обфускация констант
const getBackendUrl = () => process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const getConnectSound = () => "https://nekto.me/audiochat/sound/connect.mp3";

// Категории для выбора
const CATEGORIES = [
  { id: 'all', name: 'Все темы', icon: '🌐', description: 'Общение на любые темы' },
  { id: 'dating', name: 'Знакомства', icon: '💕', description: 'Найти интересных людей' },
  { id: 'friends', name: 'Дружба', icon: '👥', description: 'Пообщаться по душам' },
  { id: 'fun', name: 'Развлечения', icon: '🎭', description: 'Весело провести время' },
  { id: 'advice', name: 'Советы', icon: '💡', description: 'Получить или дать совет' },
  { id: 'games', name: 'Игры', icon: '🎮', description: 'Обсудить игры и хобби' }
];

// Компонент модального окна для системных сообщений
const SystemMessageModal = ({ message, isVisible, onClose }) => {
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4 transition-opacity duration-300">
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-purple-500/30 rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl animate-scale-in">
        <div className="text-center mb-6">
          <div className="w-16 h-16 bg-purple-600/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-white mb-2">Системное сообщение</h3>
          <p className="text-gray-300 text-sm leading-relaxed">{message}</p>
        </div>
        
        <button
          onClick={onClose}
          className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 transition-all duration-300 transform hover:-translate-y-0.5 active:translate-y-0"
        >
          Понятно
        </button>
        
        <div className="absolute top-4 right-4">
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors duration-200"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

// Компонент счетчика пользователей онлайн
const OnlineUsersCounter = () => {
  const [onlineCount, setOnlineCount] = useState(0);
  const [isGrowing, setIsGrowing] = useState(false);

  // Реальное получение данных о количестве онлайн-пользователей
  useEffect(() => {
    const fetchOnlineCount = async () => {
      try {
        const response = await fetch(`${getBackendUrl()}/api/stats`);
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        const newCount = data.online_users || 0;
        
        if (newCount > onlineCount) {
          setIsGrowing(true);
          setTimeout(() => setIsGrowing(false), 1000);
        }
        
        setOnlineCount(newCount);
      } catch (error) {
        console.error('Error fetching online count:', error);
        // Fallback: можно установить минимальное значение
        setOnlineCount(prev => prev > 0 ? prev : 1);
      }
    };

    // Первый запрос
    fetchOnlineCount();

    // Обновление каждые 10 секунд
    const interval = setInterval(fetchOnlineCount, 3000);

    return () => clearInterval(interval);
  }, [onlineCount]);

  // Анимация счетчика
  const formatCount = (count) => {
    return count.toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  };

  return (
    <div className="flex items-center justify-center mb-4 space-x--5">
      <div className="relative">
        <div className={`
          text-lg font-semibold transition-all duration-400
          ${isGrowing ? 'text-green-400 scale-110' : 'text-purple-300'}
        `}>
          <span className="inline-block min-w-[60px]">
            {formatCount(onlineCount)}
          </span>
        </div>
      </div>
      
      <div className="flex items-center space-x-2">
  <div className="relative">
    <div className="w-2 h-2 bg-green-500 rounded-full animate-ping absolute"></div>
    <div className="w-2 h-2 bg-green-500 rounded-full relative"></div>
  </div>
        <span className="text-sm text-gray-400 font-medium">онлайн сейчас</span>
      </div>
    </div>
  );
};

// Компонент таймера разговора
const ConversationTimer = ({ isActive, onReset }) => {
  const [time, setTime] = useState(0);
  
  useEffect(() => {
    let interval = null;
    
    if (isActive) {
      interval = setInterval(() => {
        setTime(prevTime => prevTime + 1);
      }, 1000);
    } else if (!isActive && time !== 0) {
      clearInterval(interval);
    }
    
    return () => clearInterval(interval);
  }, [isActive, time]);
  
  useEffect(() => {
    if (!isActive) {
      setTime(0);
    }
  }, [isActive]);
  
  // Форматирование времени в часы:минуты:секунды
  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  if (!isActive) return null;
  
  return (
    <div className="bg-gray-700/50 backdrop-blur-sm p-3 rounded-lg mb-4 text-center">
      <div className="text-sm text-gray-300 mb-1">Время разговора</div>
      <div className="text-xl font-mono font-bold text-purple-400">
        {formatTime(time)}
      </div>
    </div>
  );
};

function App() {
  // Добавляем обработчики защиты
  useEffect(() => {
    document.addEventListener('copy', disableCopy);
    document.addEventListener('contextmenu', disableContextMenu);
    document.addEventListener('dragstart', disableDrag);
    document.addEventListener('selectstart', disableSelection);
    
    return () => {
      document.removeEventListener('copy', disableCopy);
      document.removeEventListener('contextmenu', disableContextMenu);
      document.removeEventListener('dragstart', disableDrag);
      document.removeEventListener('selectstart', disableSelection);
    };
  }, []);

  const [userId] = useState(() => 'user_' + Math.random().toString(36).substr(2, 9));
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [userStatus, setUserStatus] = useState('idle');
  const [partnerId, setPartnerId] = useState(null);
  const [roomId, setRoomId] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Ready to connect');
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [showCategories, setShowCategories] = useState(false);
  const [userGender, setUserGender] = useState('any');
  const [searchGender, setSearchGender] = useState('any');
  const [isConversationActive, setIsConversationActive] = useState(false);
  const [systemMessage, setSystemMessage] = useState('');
  const [showSystemMessage, setShowSystemMessage] = useState(false);
  
  const wsRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const localAudioRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const iceCandidatesQueue = useRef([]);
  const audioRef = useRef(new Audio(getConnectSound()));

  const playSound = () => {
    audioRef.current.currentTime = 0;
    audioRef.current.play().catch(e => console.log("Audio play error:", e));
  };

  useEffect(() => {
    if (remoteAudioRef.current) {
      remoteAudioRef.current.volume = volume;
    }
  }, [volume]);

  const sendMessage = (message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      console.log('Sent message:', message);
    } else {
      console.error('WebSocket is not open');
    }
  };

  const handleWebSocketMessage = (message) => {
    console.log('Received message:', message);
    
    switch (message.type) {
      case 'connected':
        setConnectionStatus('connected');
        setStatusMessage(message.message);
        break;
      case 'search_started':
        setUserStatus('searching');
        setStatusMessage(message.message || 'Поиск собеседника...');
        break;
      case 'search_stopped':
        setUserStatus('idle');
        setStatusMessage(message.message || 'Поиск остановлен');
        break;
      case 'match_found':
        setUserStatus('connected');
        setPartnerId(message.partner_id);
        setRoomId(message.room_id);
        setStatusMessage('Собеседник найден! Идет подключение...');
        playSound();
        initializeVoiceChat();
        break;
      case 'partner_disconnected':
        handlePartnerDisconnected();
        break;
      case 'webrtc_signal':
        handleWebRTCSignal(message.signal);
        break;
      case 'force_disconnect':
        handlePartnerDisconnected('Соединение было разорвано');
        break;
      case 'search_status':
        setStatusMessage(message.message);
        break;
      case 'search_timeout':
        setUserStatus('idle');
        setStatusMessage(message.message || 'Не удалось найти собеседника');
        break;
      case 'system_message':
        // Обработка системного сообщения
        setSystemMessage(message.message || 'Системное сообщение');
        setShowSystemMessage(true);
        break;
      case 'error':
        setStatusMessage(message.message || 'Произошла ошибка');
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const handlePartnerDisconnected = (customMessage) => {
    setUserStatus('idle');
    setPartnerId(null);
    setRoomId(null);
    setIsConversationActive(false);
    setStatusMessage(customMessage || 'Собеседник отключился');
    playSound();
    cleanupVoiceChat();
  };

  const initializeVoiceChat = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      setLocalStream(stream);
      
      const peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ]
      });
      
      peerConnectionRef.current = peerConnection;
      
      stream.getTracks().forEach(track => {
        peerConnection.addTrack(track, stream);
      });
      
      peerConnection.ontrack = (event) => {
        const [remoteStream] = event.streams;
        setRemoteStream(remoteStream);
        setStatusMessage('Соединение установлено! Говорите свободно');
        setIsConversationActive(true);
      };
      
      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          sendMessage({
            type: 'webrtc_signal',
            signal: { type: 'ice-candidate', candidate: event.candidate }
          });
        }
      };
      
      peerConnection.onicecandidateerror = (error) => {
        console.error('ICE candidate error:', error);
      };
      
      const offer = await peerConnection.createOffer({
        offerToReceiveAudio: true
      });
      await peerConnection.setLocalDescription(offer);
      
      sendMessage({
        type: 'webrtc_signal',
        signal: { type: 'offer', offer: offer }
      });
      
      setIsVoiceEnabled(true);
      
    } catch (error) {
      console.error('Error initializing voice chat:', error);
      setStatusMessage('Ошибка доступа к микрофону');
    }
  };

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
            signal: { type: 'answer', answer: answer }
        });
      } 
      else if (signal.type === 'answer') {
        await peerConnection.setRemoteDescription(signal.answer);
      } 
      else if (signal.type === 'ice-candidate') {
        const candidate = new RTCIceCandidate(signal.candidate);
        if (peerConnection.remoteDescription) {
          await peerConnection.addIceCandidate(candidate);
        } else {
          iceCandidatesQueue.current.push(candidate);
        }
      }
    } catch (error) {
      console.error('Error handling WebRTC signal:', error);
    }
  };

  const toggleMicrophone = () => {
    if (localStream) {
      localStream.getAudioTracks().forEach(track => {
        track.enabled = !track.enabled;
      });
      setIsMicMuted(!isMicMuted);
    }
  };

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
    setIsMicMuted(false);
    setIsConversationActive(false);
  };

  const startSearch = () => {
    const searchData = {
      type: 'start_search',
      search_params: {
        category: selectedCategory,
        user_gender: userGender,
        search_gender: searchGender,
        topics: []
      }
    };
    sendMessage(searchData);
  };

  const stopSearch = () => {
    sendMessage({ type: 'stop_search' });
  };

  const stopConnection = () => {
    if (userStatus === 'connected') {
      sendMessage({ 
        type: 'disconnect',
        room_id: roomId,
        partner_id: partnerId
      });
      
      handlePartnerDisconnected('Вы разорвали соединение');
    }
  };

  const nextPartner = () => {
    sendMessage({ 
      type: 'next_partner',
      room_id: roomId,
      partner_id: partnerId
    });
    cleanupVoiceChat();
  };

  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = getBackendUrl().replace('https://', 'wss://').replace('http://', 'ws://');
      wsRef.current = new WebSocket(`${wsUrl}/ws/${userId}`);
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        setStatusMessage('Подключен к голосовому чату');
        console.log('WebSocket connected');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        setStatusMessage('Отключен от сервера');
        console.log('WebSocket disconnected');
        setTimeout(connectWebSocket, 5000);
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
      cleanupVoiceChat();
    };
  }, [userId]);

  useEffect(() => {
    if (remoteAudioRef.current && remoteStream) {
      remoteAudioRef.current.srcObject = remoteStream;
      remoteAudioRef.current.play().catch(e => 
        console.log("Remote audio play error:", e)
      );
    }
    
    if (localAudioRef.current && localStream) {
      localAudioRef.current.srcObject = localStream;
    }
  }, [remoteStream, localStream]);

  const getSelectedCategory = () => {
    return CATEGORIES.find(cat => cat.id === selectedCategory) || CATEGORIES[0];
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative">
      {/* Модальное окно для системных сообщений */}
      <SystemMessageModal
        message={systemMessage}
        isVisible={showSystemMessage}
        onClose={() => setShowSystemMessage(false)}
      />
      
      <div 
        className="bg-gray-800 rounded-2xl shadow-2xl p-8 w-full max-w-md border border-gray-700 z-10 relative"
        onCopy={disableCopy}
        onDragStart={disableDrag}
        onContextMenu={disableContextMenu}
      >
        <div 
          className="text-center mb-8"
          onCopy={disableCopy}
          onContextMenu={disableContextMenu}
        >
          <div className="flex justify-center mb-4">
            <svg 
              className="w-20 h-20" 
              viewBox="0 0 24 24" 
              fill="none" 
              xmlns="http://www.w3.org/2000/svg"
              onContextMenu={disableContextMenu}
            >
              <path 
                d="M12 15C13.6569 15 15 13.6569 15 12V6C15 4.34315 10.3431 3 12 3C10.3431 3 9 4.34315 9 6V12C9 13.6569 10.3431 15 12 15Z" 
                stroke="url(#gradient)" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              />
              <path 
                d="M18 12C18 15.3137 15.3137 18 12 18C8.68629 18 6 15.3137 6 12" 
                stroke="url(#gradient)" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              />
              <path 
                d="M19 12C19 15.866 15.866 19 12 19C8.13401 19 5 15.866 5 12" 
                stroke="url(#gradient)" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              />
              <path 
                d="M12 19V22M8 22H16" 
                stroke="url(#gradient)" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              />
              <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#a78bfa" />
                  <stop offset="100%" stopColor="#c084fc" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          
          <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-500 to-cyan-500 mb-2">
            ГОЛОСОВАЯ РУЛЕТКА
          </h1>
          <p 
            className="text-gray-400 mb-4"
            onCopy={disableCopy}
            onContextMenu={disableContextMenu}
          >
            
          </p>

          {/* РЕАЛЬНЫЙ СЧЕТЧИК ПОЛЬЗОВАТЕЛЕЙ ОНЛАЙН */}
          <OnlineUsersCounter />

        </div>

        {/* ТАЙМЕР РАЗГОВОРА */}
        <ConversationTimer isActive={isConversationActive} />

        {/* Блок выбора категорий и настроек */}
        {userStatus === 'idle' && (
          <div className="bg-gray-700/50 backdrop-blur-sm p-4 rounded-xl mb-6">
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Выберите тему:
              </label>
              <div className="relative">
                <button
                  onClick={() => setShowCategories(!showCategories)}
                  className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-left text-white flex items-center justify-between hover:bg-gray-500 transition-colors"
                >
                  <div className="flex items-center">
                    <span className="text-xl mr-3">{getSelectedCategory().icon}</span>
                    <span>{getSelectedCategory().name}</span>
                  </div>
                  <svg className="w-4 h-4 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                    style={{ transform: showCategories ? 'rotate(180deg)' : 'rotate(0deg)' }}>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                
                {showCategories && (
                  <div className="absolute z-50 w-full mt-1 bg-gray-700 border border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                    {CATEGORIES.map((category) => (
                      <button
                        key={category.id}
                        onClick={() => {
                          setSelectedCategory(category.id);
                          setShowCategories(false);
                        }}
                        className={`w-full text-left px-4 py-3 flex items-center hover:bg-gray-600 transition-colors ${
                          selectedCategory === category.id ? 'bg-purple-600/30 text-white' : 'text-gray-300'
                        }`}
                      >
                        <span className="text-xl mr-3">{category.icon}</span>
                        <div className="flex-1">
                          <div className="font-medium">{category.name}</div>
                          <div className="text-xs text-gray-400">{category.description}</div>
                        </div>
                        {selectedCategory === category.id && (
                          <svg className="w-4 h-4 text-green-400 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Ваш пол:
                </label>
                <select
                  value={userGender}
                  onChange={(e) => setUserGender(e.target.value)}
                  className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                >
                  <option value="any">Любой</option>
                  <option value="male">Мужской</option>
                  <option value="female">Женский</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Искать:
                </label>
                <select
                  value={searchGender}
                  onChange={(e) => setSearchGender(e.target.value)}
                  className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                >
                  <option value="any">Всех</option>
                  <option value="male">Мужчин</option>
                  <option value="female">Женщин</option>
                </select>
              </div>
            </div>

            <div className="bg-gray-600/50 rounded-lg p-3">
              <div className="flex items-center text-sm text-gray-300">
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Тема: {getSelectedCategory().name.toLowerCase()}</span>
              </div>
            </div>
          </div>
        )}

        <div 
          className="bg-gray-700 rounded-lg p-4 mb-6"
          onCopy={disableCopy}
          onContextMenu={disableContextMenu}
        >
          <div className="flex items-center justify-between mb-2">
  <span className="text-sm text-gray-400">Статус:</span>
  <div className="flex items-center">
    {!showCategories && (
      <span className={`relative flex h-3 w-3 mr-2 ${
        userStatus === 'searching' ? 'animate-ping' : ''
      }`}>
        <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${
          userStatus === 'searching' ? 'bg-yellow-400' : 
          userStatus === 'connected' ? 'bg-green-400' : 'bg-gray-400'
        } opacity-75`}></span>
        <span className={`relative inline-flex rounded-full h-3 w-3 ${
          userStatus === 'searching' ? 'bg-yellow-500' : 
          userStatus === 'connected' ? 'bg-green-500' : 'bg-gray-500'
        }`}></span>
      </span>
    )}
    <span className={`text-sm font-semibold ${
      userStatus === 'searching' ? 'text-yellow-400' : 
      userStatus === 'connected' ? 'text-green-400' : 'text-gray-400'
    }`}>
      {userStatus === 'searching' ? 'Поиск...' : 
       userStatus === 'connected' ? 'В разговоре' : 'Готов'}
    </span>
  </div>
</div>
          <div className="text-center">
            <p className="text-sm text-gray-300">{statusMessage}</p>
          </div>
        </div>

        <div className="space-y-3">
          {userStatus === 'idle' && (
            <button
              onClick={startSearch}
              disabled={connectionStatus !== 'connected'}
              className="w-full relative overflow-hidden bg-gradient-to-r from-blue-500 to-blue-600 text-white py-4 px-6 rounded-lg font-medium hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl active:shadow-md active:scale-[0.98] transform hover:-translate-y-0.5"
              onContextMenu={disableContextMenu}
            >
              <span className="relative z-10 flex items-center justify-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
                НАЧАТЬ ПОИСК
              </span>
              <span className="absolute inset-0 bg-white opacity-0 hover:opacity-10 transition-opacity duration-300"></span>
            </button>
          )}

          {userStatus === 'searching' && (
            <button
              onClick={stopSearch}
              className="w-full relative overflow-hidden bg-gradient-to-r from-red-500 to-red-600 text-white py-4 px-6 rounded-lg font-medium hover:from-red-600 hover:to-red-700 transition-all duration-300 shadow-lg hover:shadow-xl active:shadow-md active:scale-[0.98] transform hover:-translate-y-0.5"
              onContextMenu={disableContextMenu}
            >
              <span className="relative z-10 flex items-center justify-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                </svg>
                ОСТАНОВИТЬ ПОИСК
              </span>
              <span className="absolute inset-0 bg-white opacity-0 hover:opacity-10 transition-opacity duration-300"></span>
            </button>
          )}

          {userStatus === 'connected' && (
            <div className="bg-gray-700/50 backdrop-blur-sm p-4 rounded-xl shadow-inner">
              <div className="flex items-center space-x-3 mb-4">
                <span className="text-gray-300">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" clipRule="evenodd" />
                  </svg>
                </span>
                <div className="relative w-full flex items-center">
                  <input
  type="range"
  min="0"
  max="1"
  step="0.01"
  value={volume}
  onChange={(e) => setVolume(parseFloat(e.target.value))}
  className="w-full h-1.5 bg-gray-600 rounded-full appearance-none cursor-pointer accent-purple-500 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:shadow-lg"
  onContextMenu={disableContextMenu}
/>
                  <div 
                    className="absolute h-1.5 bg-purple-500 rounded-full pointer-events-none"
                    style={{ width: `${volume * 100}%` }}
                  ></div>
                </div>
                <span className="text-gray-300">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                  </svg>
                </span>
              </div>
              
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={toggleMicrophone}
                  className={`relative overflow-hidden py-3 px-2 rounded-lg font-medium transition-all duration-300 shadow hover:shadow-md active:scale-[0.98] ${
                    isMicMuted ? 'bg-gray-600/70 text-gray-300 hover:bg-gray-600' : 'bg-green-600/90 text-white hover:bg-green-600'
                  }`}
                  onContextMenu={disableContextMenu}
                >
                  <span className="flex flex-col items-center justify-center text-xs sm:text-sm">
                    {isMicMuted ? (
                      <>
                        <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                        </svg>
                        Вкл микрофон
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                        </svg>
                        Выкл микрофон
                      </>
                    )}
                  </span>
                  <span className="absolute inset-0 bg-white opacity-0 hover:opacity-10 transition-opacity duration-300"></span>
                </button>
                
                <button
                  onClick={stopConnection}
                  className="relative overflow-hidden bg-red-600/90 text-white py-3 px-2 rounded-lg font-medium hover:bg-red-600 transition-all duration-300 shadow hover:shadow-md active:scale-[0.98]"
                  onContextMenu={disableContextMenu}
                >
                  <span className="flex flex-col items-center justify-center text-xs sm:text-sm">
                    <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                    </svg>
                    Стоп
                  </span>
                  <span className="absolute inset-0 bg-white opacity-0 hover:opacity-10 transition-opacity duration-300"></span>
                </button>
                
                <button
                  onClick={nextPartner}
                  className="relative overflow-hidden bg-orange-600/90 text-white py-3 px-2 rounded-lg font-medium hover:bg-orange-600 transition-all duration-300 shadow hover:shadow-md active:scale-[0.98]"
                  onContextMenu={disableContextMenu}
                >
                  <span className="flex flex-col items-center justify-center text-xs sm:text-sm">
                    <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                    </svg>
                    Следующий
                  </span>
                  <span className="absolute inset-0 bg-white opacity-0 hover:opacity-10 transition-opacity duration-300"></span>
                </button>
              </div>
            </div>
          )}
        </div>

        <audio 
          ref={localAudioRef} 
          muted 
          playsInline
          style={{ display: 'none' }} 
        />
        <audio 
          ref={remoteAudioRef} 
          autoPlay 
          playsInline
          style={{ display: 'none' }}
        />
      </div>
    </div>
  );
}

export default App;