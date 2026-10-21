/*
============================================================
FULL APP.JS — LIVEKIT 1-1 + 6 КАМЕР, ЧАТ, ЖАЛОБЫ, UI
ОПТИМИЗИРОВАННАЯ ВЕРСИЯ + ВЫНЕСЕННЫЙ CHATCONTENT
============================================================
*/

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { createPortal } from 'react-dom';
import { FaTelegram, FaVk, FaTiktok, FaQuestionCircle } from 'react-icons/fa';
import { Room, RoomEvent, Track, VideoPresets } from 'livekit-client';
import { useVirtualizer } from '@tanstack/react-virtual';
import ConsoleGuard from './ConsoleGuard';
import { MetrikaCounter, ym } from 'react-metrika';
import "./App.css";

// ============= ПЕРЕВОД ПРИЧИН МОДЕРАЦИИ =============
const REASON_TRANSLATIONS = {
  'spam': 'Спам',
  'harassment': 'Харассмент',
  'inappropriate_content': 'Неуместный контент',
  'voice_abuse': 'Злоупотребление голосом',
  'ddos': 'DDOS атака',
  'multiple_accounts': 'Мультиаккаунт',
  'other': 'Другое',
  'no_action': 'Без действия',
  'manual_unmute': 'Ручное снятие заглушения',
  'manual_unban': 'Ручной разбан',
  'automatic': 'Автоматическое действие',
  'Manual unmute': 'Ручное снятие заглушения',
  'Manual unban': 'Ручной разбан',
};

const translateReason = (reason) => {
  if (!reason) return 'Нарушение правил';
  return REASON_TRANSLATIONS[reason] || reason;
};

// ============= УТИЛИТА ЛОГИРОВАНИЯ =============
const DEBUG = process.env.NODE_ENV !== 'production';
const log = DEBUG ? console.log.bind(console) : () => {};
const warn = DEBUG ? console.warn.bind(console) : () => {};
const info = DEBUG ? console.info.bind(console) : () => {};

// ============= PORTAL =============
const Portal = ({ children }) => {
  if (typeof document === 'undefined') return null;
  return createPortal(children, document.body);
};

// ============= SCROLL TO TOP (ракета с огнём) =============
const ScrollToTopButton = React.memo(({ isMobile }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isLaunching, setIsLaunching] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 400) {
        setIsVisible(true);
      } else {
        setIsVisible(false);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = useCallback(() => {
    setIsLaunching(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });

    const appEl = document.querySelector('.app');
    if (appEl && appEl.scrollTop > 0) {
      appEl.scrollTo({ top: 0, behavior: 'smooth' });
    }

    const chatCard = document.querySelector('.chat-card');
    if (chatCard && chatCard.scrollTop > 0) {
      chatCard.scrollTo({ top: 0, behavior: 'smooth' });
    }

    if (document.documentElement) document.documentElement.scrollTop = 0;
    if (document.body) document.body.scrollTop = 0;

    setTimeout(() => setIsLaunching(false), 800);
  }, []);

  const size = isMobile ? 52 : 62;

  return (
    <button
      onClick={scrollToTop}
      title="Наверх 🚀"
      aria-label="Наверх"
      style={{
        position: 'fixed',
        bottom: isMobile ? '80px' : '100px',
        right: isMobile ? '16px' : '24px',
        zIndex: 99998,
        width: size,
        height: size,
        borderRadius: '50%',
        background: 'radial-gradient(circle at 30% 30%, rgba(30,40,80,0.95), rgba(15,20,45,0.98))',
        border: '1.5px solid rgba(0,255,255,0.5)',
        boxShadow: '0 8px 24px rgba(0,210,255,0.5), 0 0 30px rgba(123,47,247,0.4)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        padding: 0,
        opacity: isVisible ? 1 : 0,
        transform: isLaunching
          ? 'translateY(-12px) scale(1.15)'
          : isVisible
            ? 'translateY(0) scale(1)'
            : 'translateY(20px) scale(0.8)',
        pointerEvents: isVisible ? 'auto' : 'none',
        transition: 'opacity 0.3s ease, transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.2s ease',
        WebkitTapHighlightColor: 'transparent',
        overflow: 'visible',
      }}
      onMouseEnter={(e) => {
        if (!isLaunching) {
          e.currentTarget.style.transform = 'translateY(-4px) scale(1.08)';
          e.currentTarget.style.boxShadow = '0 12px 32px rgba(0,210,255,0.7), 0 0 40px rgba(123,47,247,0.6), 0 0 60px rgba(255,100,0,0.4)';
        }
      }}
      onMouseLeave={(e) => {
        if (!isLaunching) {
          e.currentTarget.style.transform = 'translateY(0) scale(1)';
          e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,210,255,0.5), 0 0 30px rgba(123,47,247,0.4)';
        }
      }}
    >
      <svg
        width={size * 0.55}
        height={size * 0.55}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ display: 'block', margin: 0, position: 'relative', zIndex: 2 }}
      >
        <path
          d="M12 2C12 2 8 6 8 12V18L6 20H18L16 18V12C16 6 12 2 12 2Z"
          fill="url(#rocketBody)"
          stroke="rgba(0,255,255,0.7)"
          strokeWidth="1"
          strokeLinejoin="round"
        />
        <circle cx="12" cy="10" r="2" fill="#00d2ff" opacity="0.9" />
        <circle cx="12" cy="10" r="1" fill="#fff" opacity="0.8" />
        <path d="M8 16L5 20V22L8 19V16Z" fill="url(#rocketFin)" stroke="rgba(0,255,255,0.5)" strokeWidth="0.5" />
        <path d="M16 16L19 20V22L16 19V16Z" fill="url(#rocketFin)" stroke="rgba(0,255,255,0.5)" strokeWidth="0.5" />
        <path d="M10 20H14L13 22H11L10 20Z" fill="#333" stroke="rgba(0,255,255,0.4)" strokeWidth="0.5" />

        <defs>
          <linearGradient id="rocketBody" x1="12" y1="2" x2="12" y2="20" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#e0e8ff" />
            <stop offset="0.5" stopColor="#a0b8ff" />
            <stop offset="1" stopColor="#7088cc" />
          </linearGradient>
          <linearGradient id="rocketFin" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#ff6b6b" />
            <stop offset="1" stopColor="#c0392b" />
          </linearGradient>
        </defs>
      </svg>

      <div style={{
        position: 'absolute',
        top: '68%',
        left: '50%',
        transform: 'translateX(-50%)',
        width: isMobile ? '16px' : '20px',
        height: isMobile ? '24px' : '30px',
        zIndex: 1,
        pointerEvents: 'none',
      }}>
        <div style={{
          position: 'absolute',
          left: '50%',
          top: 0,
          transform: 'translateX(-50%)',
          width: '100%',
          height: '100%',
          background: 'radial-gradient(ellipse at 50% 0%, rgba(255,100,0,0.95) 0%, rgba(255,60,0,0.6) 40%, transparent 75%)',
          borderRadius: '50% 50% 40% 40%',
          filter: 'blur(1px)',
          animation: 'fireFlicker1 0.15s ease-in-out infinite alternate',
          transformOrigin: 'top center',
        }} />
        <div style={{
          position: 'absolute',
          left: '50%',
          top: '10%',
          transform: 'translateX(-50%)',
          width: '70%',
          height: '80%',
          background: 'radial-gradient(ellipse at 50% 0%, rgba(255,180,0,0.95) 0%, rgba(255,120,0,0.7) 50%, transparent 80%)',
          borderRadius: '50% 50% 40% 40%',
          filter: 'blur(0.5px)',
          animation: 'fireFlicker2 0.12s ease-in-out infinite alternate',
          transformOrigin: 'top center',
        }} />
        <div style={{
          position: 'absolute',
          left: '50%',
          top: '20%',
          transform: 'translateX(-50%)',
          width: '40%',
          height: '60%',
          background: 'radial-gradient(ellipse at 50% 0%, rgba(255,255,150,1) 0%, rgba(255,220,0,0.8) 50%, transparent 80%)',
          borderRadius: '50% 50% 40% 40%',
          filter: 'blur(0.5px)',
          animation: 'fireFlicker3 0.1s ease-in-out infinite alternate',
          transformOrigin: 'top center',
        }} />
        <div style={{
          position: 'absolute',
          left: '50%',
          top: '60%',
          width: '4px',
          height: '4px',
          borderRadius: '50%',
          background: '#ffcc00',
          boxShadow: '0 0 6px #ffcc00',
          animation: 'spark1 0.8s linear infinite',
        }} />
        <div style={{
          position: 'absolute',
          left: '40%',
          top: '60%',
          width: '3px',
          height: '3px',
          borderRadius: '50%',
          background: '#ff8800',
          boxShadow: '0 0 5px #ff8800',
          animation: 'spark2 1s linear infinite',
          animationDelay: '0.3s',
        }} />
      </div>

      <span style={{
        position: 'absolute',
        bottom: '-8px',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '40px',
        height: '20px',
        background: 'radial-gradient(ellipse, rgba(255,120,0,0.5) 0%, transparent 70%)',
        filter: 'blur(6px)',
        pointerEvents: 'none',
        animation: 'fireFlicker1 0.2s ease-in-out infinite alternate',
      }} />

      <span style={{
        position: 'absolute',
        inset: -4,
        borderRadius: '50%',
        border: '1.5px solid rgba(0,255,255,0.4)',
        animation: 'scrollTopPulse 2s ease-in-out infinite',
        pointerEvents: 'none',
      }} />
    </button>
  );
});

// ============= COPY EMAIL (копирование в буфер + всплывашка) =============
const CopyEmail = React.memo(({ email, color = '#00ffff', fontSize = '13px', style = {} }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(email);
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = email;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch (err) {
      console.error('❌ [COPY] Failed:', err);
    }
  }, [email]);

  return (
    <span
      onClick={handleCopy}
      title="Нажмите, чтобы скопировать"
      style={{
        color: color,
        cursor: 'pointer',
        fontWeight: '600',
        wordBreak: 'break-all',
        textDecoration: 'none',
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        fontSize: fontSize,
        transition: 'opacity 0.2s ease',
        ...style,
      }}
      onMouseEnter={(e) => { e.currentTarget.style.opacity = '0.8'; }}
      onMouseLeave={(e) => { e.currentTarget.style.opacity = '1'; }}
    >
      📧 {email}
      {copied && (
        <span
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: '6px',
            background: 'linear-gradient(135deg, #00ff88, #00cc66)',
            color: '#000',
            padding: '4px 10px',
            borderRadius: '8px',
            fontSize: '11px',
            fontWeight: '700',
            whiteSpace: 'nowrap',
            boxShadow: '0 4px 16px rgba(0,255,136,0.5)',
            animation: 'copyPop 0.3s ease',
            pointerEvents: 'none',
            zIndex: 100,
          }}
        >
          ✅ Скопировано
        </span>
      )}
    </span>
  );
});

// ============= CONTROL BUTTON (Cosmic Voice style) =============
const ControlButton = React.memo(({ icon, label, onClick, variant = 'primary', isActive = false, disabled = false, isMobile }) => {
  const colors = {
    primary: { border: '#00ffff', glow: '#00ffff', text: '#00ffff' },
    danger: { border: '#ff4757', glow: '#ff4757', text: '#ff6b6b' },
    success: { border: '#2ed573', glow: '#2ed573', text: '#2ed573' },
  };
  const c = colors[variant] || colors.primary;

  return (
    <button onClick={onClick} disabled={disabled} style={{
      display: 'flex', alignItems: 'center', gap: '10px',
      padding: isMobile ? '10px 14px' : '12px 20px',
      background: 'linear-gradient(135deg, rgba(10,20,45,0.9), rgba(5,10,25,0.95))',
      border: `2px solid ${disabled ? 'rgba(100,100,120,0.3)' : c.border}`,
      borderRadius: '12px',
      color: disabled ? '#555' : c.text,
      fontWeight: '800', letterSpacing: '0.08em', textTransform: 'uppercase',
      fontSize: isMobile ? '10px' : '11px',
      cursor: disabled ? 'not-allowed' : 'pointer',
      boxShadow: disabled ? 'none' : `0 0 20px ${c.glow}40, inset 0 0 20px ${c.glow}15`,
      transition: 'all 0.25s ease',
      opacity: disabled ? 0.5 : 1,
      position: 'relative',
    }}
    onMouseEnter={(e) => { if (!disabled) { e.currentTarget.style.boxShadow = `0 0 30px ${c.glow}, inset 0 0 30px ${c.glow}30`; e.currentTarget.style.transform = 'translateY(-2px)'; } }}
    onMouseLeave={(e) => { if (!disabled) { e.currentTarget.style.boxShadow = `0 0 20px ${c.glow}40, inset 0 0 20px ${c.glow}15`; e.currentTarget.style.transform = 'translateY(0)'; } }}
    >
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: isActive ? '#fff' : c.border,
        boxShadow: `0 0 8px ${c.glow}`,
        animation: isActive ? 'cosmicPulse 1.5s infinite' : 'none',
        flexShrink: 0,
      }} />
      <span style={{ fontSize: isMobile ? '16px' : '18px' }}>{icon}</span>
      {label && <span>{label}</span>}
    </button>
  );
});

// ============= КОНФИГУРАЦИЯ =============
const CONFIG = {
  BACKEND_URL: "http://192.168.0.103:8000",
  WS_URL: "ws://192.168.0.103:8000",
  SESSION_SOUND: "https://nekto.me/audiochat/sound/connect.mp3",
  WELCOME_SOUND: "/welcome.mp3",
  MODERATION_SOUNDS: {
    warn: "/sounds/warn.mp3",
    mute: "/sounds/mute.mp3",
    ban: "/sounds/ban.mp3",
    success: "/sounds/success.mp3",
    kick: "/sounds/kick.mp3",
    unmute: "/sounds/success.mp3",
    unban: "/sounds/success.mp3"
  },
  WS_CONNECTION_TIMEOUT: 10000,
  WS_RECONNECT_DELAY: 3000,
  WS_MAX_RECONNECT_ATTEMPTS: 10,
  WS_HEARTBEAT_INTERVAL: 30000,
  MODERATION_CHECK_INTERVAL: 60000,
  BAN_OVERLAY_AUTOCLOSE_SECONDS: 10,
  KICK_OVERLAY_AUTOCLOSE_SECONDS: 5,
  LIVEKIT_URL: "ws://192.168.0.103:7880"
};

const APP_VERSION = process.env.REACT_APP_VERSION || '0.0.2';
const YM_COUNTER_ID = 98765432;

const ICE_SERVERS = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ],
  iceCandidatePoolSize: 4,
  iceTransportPolicy: 'all',
  bundlePolicy: 'max-bundle',
  rtcpMuxPolicy: 'require'
};

const CATEGORIES = [
  { id: 'all', name: 'Все темы', icon: '🌐', description: 'Общение на любые темы' },
  { id: 'dating', name: 'Знакомства', icon: '💕', description: 'Найти интересных людей' },
  { id: 'friends', name: 'Дружба', icon: '👥', description: 'Пообщаться по душам' },
  { id: 'fun', name: 'Развлечения', icon: '🎭', description: 'Весело провести время' },
  { id: 'advice', name: 'Советы', icon: '💡', description: 'Получить или дать совет' },
  { id: 'games', name: 'Игры', icon: '🎮', description: 'Обсудить игры и хобби' }
];

const COUNTRIES = [
  { code: 'any', name: '🌐 Любая страна' },
  { code: 'RU', name: '🇷🇺 Россия' },
  { code: 'UA', name: '🇺🇦 Украина' },
  { code: 'BY', name: '🇧🇾 Беларусь' },
  { code: 'KZ', name: '🇰🇿 Казахстан' },
  { code: 'UZ', name: '🇺🇿 Узбекистан' },
  { code: 'KG', name: '🇰🇬 Кыргызстан' },
  { code: 'TJ', name: '🇹🇯 Таджикистан' },
  { code: 'AM', name: '🇦🇲 Армения' },
  { code: 'GE', name: '🇬🇪 Грузия' },
  { code: 'AZ', name: '🇦🇿 Азербайджан' },
  { code: 'MD', name: '🇲🇩 Молдова' },
  { code: 'LV', name: '🇱🇻 Латвия' },
  { code: 'LT', name: '🇱🇹 Литва' },
  { code: 'EE', name: '🇪🇪 Эстония' },
  { code: 'US', name: '🇺🇸 США' },
  { code: 'CA', name: '🇨🇦 Канада' },
  { code: 'GB', name: '🇬🇧 Великобритания' },
  { code: 'DE', name: '🇩🇪 Германия' },
  { code: 'FR', name: '🇫🇷 Франция' },
  { code: 'IT', name: '🇮🇹 Италия' },
  { code: 'ES', name: '🇪🇸 Испания' },
  { code: 'PL', name: '🇵🇱 Польша' },
  { code: 'CZ', name: '🇨🇿 Чехия' },
  { code: 'NL', name: '🇳🇱 Нидерланды' },
  { code: 'SE', name: '🇸🇪 Швеция' },
  { code: 'NO', name: '🇳🇴 Норвегия' },
  { code: 'FI', name: '🇫🇮 Финляндия' },
  { code: 'TR', name: '🇹🇷 Турция' },
  { code: 'IL', name: '🇮🇱 Израиль' },
  { code: 'AE', name: '🇦🇪 ОАЭ' },
  { code: 'CN', name: '🇨🇳 Китай' },
  { code: 'JP', name: '🇯🇵 Япония' },
  { code: 'KR', name: '🇰🇷 Корея' },
  { code: 'IN', name: '🇮🇳 Индия' },
  { code: 'BR', name: '🇧🇷 Бразилия' },
  { code: 'AR', name: '🇦🇷 Аргентина' },
  { code: 'AU', name: '🇦🇺 Австралия' },
  { code: 'other', name: '🌏 Другая' },
];

const AGE_RANGES = [
  { id: 'any', name: '🌐 Любой возраст', min: 0, max: 100 },
  { id: '18-25', name: '18-25 лет', min: 18, max: 25 },
  { id: '26-35', name: '26-35 лет', min: 26, max: 35 },
  { id: '36-45', name: '36-45 лет', min: 36, max: 45 },
  { id: '46-55', name: '46-55 лет', min: 46, max: 55 },
  { id: '56+', name: '56+ лет', min: 56, max: 100 },
];

let isModerationWebSocketConnecting = false;

const isMobileDevice = () => {
  const userAgent = navigator.userAgent || navigator.vendor || window.opera;
  const mobileRegex = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i;
  return mobileRegex.test(userAgent) || window.innerWidth <= 768;
};

// ============= WEBSOCKET МЕНЕДЖЕР =============
class WebSocketManager {
  constructor(userId, browserId, fingerprint, onStatusChange) {
    this.userId = userId;
    this.browserId = browserId;
    this.fingerprint = fingerprint;
    this.onStatusChange = onStatusChange;
    this.ws = null;
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.reconnectTimeout = null;
    this.heartbeatInterval = null;
    this.messageHandlers = new Set();
    this.isShuttingDown = false;
    this.blockedByBan = false;
  }
  addMessageHandler(h) { this.messageHandlers.add(h); }
  removeMessageHandler(h) { this.messageHandlers.delete(h); }

  async connect() {
    if (this.isShuttingDown || this.blockedByBan) return false;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      if (this.onStatusChange) this.onStatusChange('connected');
      return true;
    }
    if (this.isConnecting) return false;
    this.isConnecting = true;
    if (this.onStatusChange) this.onStatusChange('connecting');
    try {
      if (this.ws) {
        const old = this.ws; this.ws = null;
        old.onopen = old.onclose = old.onerror = old.onmessage = null;
        setTimeout(() => { try { if (old.readyState !== WebSocket.CLOSED) old.close(); } catch(e){} }, 0);
      }
      const url = `${CONFIG.WS_URL}/ws/${this.userId}?browser_id=${this.browserId}&fingerprint=${this.fingerprint}&t=${Date.now()}`;
      this.ws = new WebSocket(url);
      const connectionTimeout = setTimeout(() => {
        if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
          const w = this.ws; this.ws = null; w.close();
          this.isConnecting = false; this.scheduleReconnect();
        }
      }, CONFIG.WS_CONNECTION_TIMEOUT);
      this.ws.onopen = () => {
        clearTimeout(connectionTimeout);
        this.isConnecting = false; this.reconnectAttempts = 0;
        if (this.onStatusChange) this.onStatusChange('connected');
        this.startHeartbeat();
        setTimeout(() => { if (this.ws && this.ws.readyState === WebSocket.OPEN) this.sendMessage({ type: 'client_ready', user_id: this.userId }); }, 200);
      };
      this.ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === 'ping') { this.sendMessage({ type: 'pong' }); return; }
          if (m.type === 'blocked') {
            this.blockedByBan = true; this.isShuttingDown = true;
            if (this.reconnectTimeout) { clearTimeout(this.reconnectTimeout); this.reconnectTimeout = null; }
            if (this.onStatusChange) this.onStatusChange('blocked');
          }
          this.messageHandlers.forEach(h => { try { h(m); } catch(e){ console.error(e); } });
        } catch(e){}
      };
      this.ws.onclose = (ev) => {
        clearTimeout(connectionTimeout);
        this.isConnecting = false; this.stopHeartbeat();
        if (this.blockedByBan) { if (this.onStatusChange) this.onStatusChange('blocked'); return; }
        const policy = ev.code === 1008 || ev.code === 4000 || ev.code === 4001 || ev.code === 4003;
        const normal = ev.code === 1000 || ev.code === 1001;
        if (policy) { this.blockedByBan = true; if (this.onStatusChange) this.onStatusChange('blocked'); return; }
        if (this.onStatusChange) this.onStatusChange('disconnected');
        if (!normal && !this.isShuttingDown && !this.blockedByBan) this.scheduleReconnect();
      };
      this.ws.onerror = () => { clearTimeout(connectionTimeout); };
      return true;
    } catch(e) { this.isConnecting = false; this.scheduleReconnect(); return false; }
  }
  scheduleReconnect() {
    if (this.reconnectTimeout || this.isShuttingDown || this.blockedByBan) return;
    const base = this.reconnectAttempts === 0 ? 1000 : CONFIG.WS_RECONNECT_DELAY;
    const delay = Math.min(base * Math.pow(1.5, this.reconnectAttempts), 30000);
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null; this.reconnectAttempts++;
      if (this.blockedByBan || this.isShuttingDown) return;
      if (this.reconnectAttempts <= CONFIG.WS_MAX_RECONNECT_ATTEMPTS) this.connect();
    }, delay);
  }
  startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatInterval = setInterval(() => { if (this.ws && this.ws.readyState === WebSocket.OPEN) this.sendMessage({ type: 'ping' }); }, CONFIG.WS_HEARTBEAT_INTERVAL);
  }
  stopHeartbeat() { if (this.heartbeatInterval) { clearInterval(this.heartbeatInterval); this.heartbeatInterval = null; } }
  sendMessage(m) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    try { this.ws.send(JSON.stringify(m)); return true; } catch(e) { return false; }
  }
  disconnect() {
    this.isShuttingDown = true; this.stopHeartbeat();
    if (this.reconnectTimeout) { clearTimeout(this.reconnectTimeout); this.reconnectTimeout = null; }
    if (this.ws) { const w = this.ws; this.ws = null; try { w.close(1000, 'bye'); } catch(e){} }
    this.isConnecting = false;
    if (this.onStatusChange) this.onStatusChange('disconnected');
  }
  resetBlock() { this.blockedByBan = false; this.isShuttingDown = false; this.reconnectAttempts = 0; }
  isConnected() { return this.ws && this.ws.readyState === WebSocket.OPEN; }
  isConnectingState() { return this.isConnecting || (this.ws && this.ws.readyState === WebSocket.CONNECTING); }
}

// ============= BAN OVERLAY =============
const BanOverlay = React.memo(({ isVisible, type, reason, duration, banType, onClose, isMobile, autoCloseSeconds = 10 }) => {
  const [countdown, setCountdown] = useState(autoCloseSeconds);
  const isBan = type === 'ban';
  const isKick = type === 'kick';
  const isExtendedBan = type === 'extended_ban';
  const isPermanent = isBan || isExtendedBan;
  useEffect(() => {
    if (!isVisible) return;
    if (isPermanent) { setCountdown(0); return; }
    setCountdown(autoCloseSeconds);
    const interval = setInterval(() => {
      setCountdown(prev => { if (prev <= 1) { clearInterval(interval); onClose && onClose(); return 0; } return prev - 1; });
    }, 1000);
    return () => clearInterval(interval);
  }, [isVisible, autoCloseSeconds, onClose, isPermanent]);
  if (!isVisible) return null;
  const isMute = type === 'mute';
  const color = isBan || isExtendedBan ? '#ff3838' : (isKick ? '#ff9f43' : (isMute ? '#feca57' : '#54a0ff'));
  const icon = isBan || isExtendedBan ? '🚫' : (isKick ? '🚪' : (isMute ? '🔇' : '⚠️'));
  const title = isBan ? 'ВЫ ЗАБЛОКИРОВАНЫ' : (isExtendedBan ? 'ВЫ ЗАБЛОКИРОВАНЫ (РАСШИРЕННЫЙ БАН)' : (isKick ? 'ВЫ ОТКЛЮЧЕНЫ' : (isMute ? 'ВЫ ЗАГЛУШЕНЫ' : 'ПРЕДУПРЕЖДЕНИЕ')));
  const banTypeLabel = banType ? ({ 'user': 'по ID', 'ip': 'по IP', 'browser': 'по браузеру', 'fingerprint': 'по устройству', 'subnet': 'по подсети', 'all': 'по всем' }[banType] || '') : '';
  return (
    <div style={{ position: 'fixed', inset: 0, background: `radial-gradient(circle at 50% 50%, rgba(${isBan || isExtendedBan ? '255,0,0' : isKick ? '255,150,0' : '255,200,0'}, 0.15) 0%, rgba(0,0,0,0.95) 70%)`, backdropFilter: 'blur(15px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 999999, padding: isMobile ? '15px' : '30px' }}>
      <div style={{ maxWidth: isMobile ? '95%' : '520px', width: '100%', background: 'linear-gradient(145deg, rgba(20,10,30,0.98), rgba(10,5,20,0.99))', borderRadius: '24px', border: `2px solid ${color}`, boxShadow: `0 30px 80px ${color}40`, padding: isMobile ? '30px 20px' : '45px 35px', textAlign: 'center' }}>
        <div style={{ fontSize: isMobile ? '72px' : '96px', marginBottom: '20px', filter: `drop-shadow(0 0 30px ${color})` }}>{icon}</div>
        <h1 style={{ fontSize: isMobile ? '1.5rem' : '2rem', fontWeight: '900', color, marginBottom: '16px' }}>{title}</h1>
        {banTypeLabel && <div style={{ fontSize: '0.9rem', color: '#ffaa00', marginBottom: '12px', fontWeight: 'bold' }}>🔒 Заблокирован {banTypeLabel}</div>}
        <div style={{ background: 'rgba(0,0,0,0.4)', borderRadius: '14px', padding: '20px', marginBottom: '20px', border: `1px solid ${color}30` }}>
          <div style={{ color: '#ccc', marginBottom: '8px' }}><strong style={{ color: '#fff' }}>Причина:</strong></div>
          <div style={{ color: '#fff', fontWeight: '600', marginBottom: duration ? '12px' : 0 }}>{reason || 'Нарушение правил'}</div>
          {duration && <><div style={{ color: '#ccc', marginBottom: '8px' }}><strong style={{ color: '#fff' }}>Длительность:</strong></div><div style={{ color, fontWeight: 'bold' }}>{duration}</div></>}
        </div>
        {!isPermanent && countdown > 0 && <div style={{ color, fontWeight: 'bold', padding: '10px', background: `${color}15`, borderRadius: '10px', marginBottom: '16px' }}>⏱ Перезагрузка через {countdown} сек...</div>}
        {!isPermanent ? (
          <button onClick={onClose} style={{ width: '100%', padding: '16px', background: `linear-gradient(135deg, ${color}, ${color}cc)`, border: 'none', borderRadius: '14px', color: 'white', fontSize: '1.1rem', fontWeight: '700', cursor: 'pointer' }}>{isKick ? 'Перезагрузить' : 'Понятно'}</button>
        ) : (
          <div style={{ padding: '16px', background: 'rgba(255,255,255,0.05)', borderRadius: '14px', color: '#888', border: `1px solid ${color}30`, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
            <div>🔒 Обратитесь в поддержку:</div>
            <CopyEmail email="support@cosmic-voice.ru" color="#00ffff" fontSize="13px" />
          </div>
        )}
      </div>
    </div>
  );
});

// ============= NOTIFICATION =============
const Notification = React.memo(({ notification, onClose, isMobile }) => {
  useEffect(() => { const t = setTimeout(onClose, notification.timeout || 5000); return () => clearTimeout(t); }, [onClose, notification.timeout]);
  const typeStyles = {
    moderator: 'linear-gradient(135deg, #ff6b6b, #ee5a52)', admin: 'linear-gradient(135deg, #4834d4, #686de0)',
    system: 'linear-gradient(135deg, #00d2d3, #54a0ff)', warning: 'linear-gradient(135deg, #feca57, #ff9ff3)',
    info: 'linear-gradient(135deg, #48dbfb, #0abde3)', success: 'linear-gradient(135deg, #1dd1a1, #10ac84)',
    error: 'linear-gradient(135deg, #ff4757, #ff3838)', warn: 'linear-gradient(135deg, #feca57, #ff9f43)',
    mute: 'linear-gradient(135deg, #ff9f43, #ee5a52)', unmute: 'linear-gradient(135deg, #1dd1a1, #10ac84)',
    ban: 'linear-gradient(135deg, #ff4757, #c0392b)', unban: 'linear-gradient(135deg, #1dd1a1, #10ac84)',
    kick: 'linear-gradient(135deg, #ff6b6b, #c0392b)'
  };
  return (
    <div style={{ position: 'fixed', top: isMobile ? '10px' : '20px', left: '50%', transform: 'translateX(-50%)', zIndex: 10000, minWidth: isMobile ? '90vw' : '400px', maxWidth: '90vw', padding: isMobile ? '15px' : '20px', borderRadius: isMobile ? '14px' : '18px', boxShadow: '0 15px 40px rgba(0,0,0,0.4)', backdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.12)', display: 'flex', alignItems: 'flex-start', gap: isMobile ? '10px' : '15px', color: 'white', fontSize: isMobile ? '13px' : '14px', background: typeStyles[notification.type] || typeStyles.info }}>
      <div style={{ fontSize: isMobile ? '20px' : '24px' }}>{notification.icon || '💫'}</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 'bold', marginBottom: '5px', fontSize: isMobile ? '14px' : '16px' }}>{notification.title || 'Уведомление'}</div>
        <div style={{ opacity: 0.9 }}>{notification.message || ''}</div>
        {notification.submessage && <div style={{ opacity: 0.75, fontSize: '12px', marginTop: '5px' }}>{notification.submessage}</div>}
        {notification.duration && <div style={{ opacity: 0.8, fontSize: '12px', marginTop: '5px', fontWeight: 'bold' }}>⏱ {notification.duration}</div>}
      </div>
      <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'white', fontSize: '20px', cursor: 'pointer' }}>×</button>
    </div>
  );
});

// ============= MAINTENANCE PAGE =============
const MaintenancePage = React.memo(({ isMobile, systemMessage, muffleEnabled, onRetry }) => {
  const [checkState, setCheckState] = useState('maintenance');
  const [dots, setDots] = useState('');

  useEffect(() => { setCheckState('maintenance'); }, [muffleEnabled, systemMessage]);

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(interval);
  }, []);

  const handleRetry = async () => {
    setCheckState('checking');
    try {
      const result = await onRetry();
      if (result && result.available) {
        setCheckState('available');
        setTimeout(() => { if (onRetry.release) onRetry.release(); }, 1800);
      } else setCheckState('maintenance');
    } catch(e) { setCheckState('error'); setTimeout(() => setCheckState('maintenance'), 2500); }
  };

  const statusConfig = {
    maintenance: { icon: '🛠️', text: 'ТЕХНИЧЕСКИЕ РАБОТЫ', color: '#00ffff', sub: 'Скоро вернёмся' },
    checking:    { icon: '🔍', text: 'ПРОВЕРЯЕМ', color: '#feca57', sub: 'Диагностика систем' },
    available:   { icon: '🚀', text: 'ЛЕТИМ!', color: '#00ff88', sub: 'Всё в норме' },
    error:       { icon: '⚠️', text: 'ОШИБКА СВЯЗИ', color: '#ff4757', sub: 'Попробуйте позже' }
  };
  const sc = statusConfig[checkState] || statusConfig.maintenance;

  return (
    <div style={{
      position: 'fixed', inset: 0,
      background: 'radial-gradient(ellipse at 30% 40%, #0a0a2a 0%, #030318 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 9999,
      padding: isMobile ? '12px' : '24px',
      overflow: 'hidden',
      boxSizing: 'border-box',
    }}>

      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        background: `
          radial-gradient(2px 2px at 15% 20%, rgba(255,255,255,0.7), transparent),
          radial-gradient(1px 1px at 25% 60%, rgba(255,255,255,0.5), transparent),
          radial-gradient(1.5px 1.5px at 80% 30%, rgba(0,255,255,0.6), transparent),
          radial-gradient(1px 1px at 45% 85%, rgba(255,255,255,0.4), transparent),
          radial-gradient(2px 2px at 90% 70%, rgba(255,0,255,0.5), transparent),
          radial-gradient(1px 1px at 60% 15%, rgba(255,255,255,0.6), transparent)
        `,
        animation: 'maintenanceStars 20s linear infinite',
      }} />

      <div style={{
        position: 'absolute', top: '-20%', left: '-20%',
        width: '60%', height: '60%',
        background: 'radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 65%)',
        pointerEvents: 'none',
        animation: 'maintenanceGlow 6s ease-in-out infinite alternate',
      }} />
      <div style={{
        position: 'absolute', bottom: '-20%', right: '-20%',
        width: '60%', height: '60%',
        background: 'radial-gradient(circle, rgba(255,0,255,0.12) 0%, transparent 65%)',
        pointerEvents: 'none',
        animation: 'maintenanceGlow 6s ease-in-out infinite alternate-reverse',
      }} />

      <div style={{
        position: 'relative',
        zIndex: 1,
        width: '100%',
        maxWidth: isMobile ? '100%' : '460px',
        background: 'linear-gradient(145deg, rgba(12,8,30,0.97), rgba(5,3,18,0.99))',
        borderRadius: '20px',
        border: `2px solid ${sc.color}40`,
        padding: isMobile ? '22px 18px' : '28px 26px',
        textAlign: 'center',
        backdropFilter: 'blur(20px)',
        boxShadow: `0 20px 60px ${sc.color}25, inset 0 1px 0 rgba(255,255,255,0.05)`,
        boxSizing: 'border-box',
        overflow: 'hidden',
        transition: 'all 0.4s ease',
      }}>

        <div style={{
          position: 'absolute', top: '-50%', right: '-30%',
          width: '80%', height: '200%',
          background: `radial-gradient(circle, ${sc.color}10 0%, transparent 65%)`,
          pointerEvents: 'none',
          animation: 'maintenanceGlow 4s ease-in-out infinite alternate',
        }} />

        <div style={{
          position: 'relative', zIndex: 1,
          fontSize: isMobile ? '48px' : '58px',
          lineHeight: 1,
          marginBottom: '10px',
          filter: `drop-shadow(0 0 20px ${sc.color})`,
          animation: checkState === 'available' ? 'maintenanceRocket 1.5s ease-in-out infinite' : 'maintenanceIconFloat 3s ease-in-out infinite',
        }}>
          {sc.icon}
        </div>

        <h1 style={{
          position: 'relative', zIndex: 1,
          fontSize: isMobile ? '1.25rem' : '1.5rem',
          fontWeight: '900',
          letterSpacing: '0.06em',
          background: 'linear-gradient(135deg, #00ffff, #ff00ff, #00ffff)',
          backgroundSize: '300% 300%',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          margin: '0 0 10px 0',
          animation: 'maintenanceGradient 4s ease infinite',
        }}>COSMIC - VOICE</h1>

        <div style={{
          position: 'relative', zIndex: 1,
          color: '#c2c8d6',
          fontSize: isMobile ? '12px' : '13px',
          lineHeight: 1.55,
          marginBottom: '14px',
          textAlign: 'center',
        }}>
          <div style={{
            fontSize: isMobile ? '13px' : '14px',
            fontWeight: '700',
            color: '#fff',
            marginBottom: '8px',
            lineHeight: 1.4,
          }}>
            {systemMessage || '🚀 Улучшаем Cosmic Voice'}
          </div>

          <p style={{ margin: '0 0 10px 0' }}>
            Проводим <b style={{ color: '#00ffff' }}>плановое обновление</b> — делаем сервис{' '}
            <b style={{ color: '#00ff88' }}>быстрее</b> и{' '}
            <b style={{ color: '#ff00ff' }}>стабильнее</b>.
          </p>

          <div style={{
            background: 'rgba(0,255,255,0.06)',
            border: '1px solid rgba(0,255,255,0.2)',
            borderRadius: '10px',
            padding: isMobile ? '10px 12px' : '12px 14px',
            textAlign: 'left',
            fontSize: isMobile ? '11px' : '12px',
            lineHeight: 1.6,
            marginBottom: '10px',
          }}>
            <div style={{ color: '#00ffff', fontWeight: '700', marginBottom: '4px' }}>⚙️ Что делаем</div>
            <div>• Ускоряем поиск собеседника</div>
            <div>• Улучшаем качество видео и звука</div>
            <div>• Обновляем систему модерации</div>
          </div>

          <p style={{
            margin: 0,
            fontSize: isMobile ? '11px' : '12px',
            color: '#8a93a8',
            fontStyle: 'italic',
          }}>
            💖 Спасибо за терпение!
          </p>
        </div>

        <button
          onClick={handleRetry}
          disabled={checkState === 'checking'}
          style={{
            position: 'relative', zIndex: 1,
            width: '100%',
            padding: isMobile ? '11px' : '13px',
            background: checkState === 'checking'
              ? 'linear-gradient(135deg, #333, #222)'
              : `linear-gradient(135deg, ${sc.color}, ${sc.color}cc)`,
            border: 'none',
            borderRadius: '12px',
            color: checkState === 'checking' ? '#888' : '#0a0a1a',
            fontSize: isMobile ? '0.88rem' : '0.98rem',
            fontWeight: '900',
            letterSpacing: '0.04em',
            cursor: checkState === 'checking' ? 'not-allowed' : 'pointer',
            boxShadow: checkState === 'checking'
              ? 'none'
              : `0 8px 20px ${sc.color}50, 0 0 30px ${sc.color}30`,
            transition: 'all 0.3s ease',
            textTransform: 'uppercase',
            boxSizing: 'border-box',
          }}
          onMouseEnter={(e) => {
            if (checkState !== 'checking') {
              e.currentTarget.style.transform = 'translateY(-2px) scale(1.02)';
              e.currentTarget.style.boxShadow = `0 12px 28px ${sc.color}70, 0 0 50px ${sc.color}50`;
            }
          }}
          onMouseLeave={(e) => {
            if (checkState !== 'checking') {
              e.currentTarget.style.transform = 'translateY(0) scale(1)';
              e.currentTarget.style.boxShadow = `0 8px 20px ${sc.color}50, 0 0 30px ${sc.color}30`;
            }
          }}
        >
          {checkState === 'checking' ? `⏳ ПРОВЕРЯЕМ${dots}` : '🔄 ПРОВЕРИТЬ СНОВА'}
        </button>

        <div style={{
          position: 'relative', zIndex: 1,
          marginTop: '12px',
          fontSize: isMobile ? '10px' : '11px',
          color: '#666',
          letterSpacing: '0.03em',
        }}>
          Cosmic-Voice. All rights reserved 2026-2027
        </div>

      </div>
    </div>
  );
});

// ============= ONLINE COUNTER =============
const OnlineUsersCounter = React.memo(({ isMobile }) => {
  const [onlineCount, setOnlineCount] = useState(() => {
    try {
      const cached = sessionStorage.getItem('online_count_cache');
      if (cached) {
        const { count, ts } = JSON.parse(cached);
        if (Date.now() - ts < 60000) return count;
      }
    } catch (e) {}
    return 0;
  });
  const [isGrowing, setIsGrowing] = useState(false);
  const isMounted = useRef(true);
  const lastFetchRef = useRef(0);

  useEffect(() => {
    isMounted.current = true;
    let abortController = null;

    const fetchOnlineCount = async () => {
      const now = Date.now();
      if (now - lastFetchRef.current < 12000) return;
      lastFetchRef.current = now;

      if (abortController) abortController.abort();
      abortController = new AbortController();

      try {
        const response = await fetch(`${CONFIG.BACKEND_URL}/api/stats`, {
          signal: abortController.signal,
          cache: 'no-store',
        });
        if (response.ok && isMounted.current) {
          const data = await response.json();
          const newCount = data.online_users || 0;
          setOnlineCount(prev => {
            if (newCount > prev) {
              setIsGrowing(true);
              setTimeout(() => setIsGrowing(false), 1000);
            }
            try {
              sessionStorage.setItem('online_count_cache', JSON.stringify({ count: newCount, ts: Date.now() }));
            } catch (e) {}
            return newCount;
          });
        }
      } catch(e) {
        if (e.name !== 'AbortError') {}
      }
    };

    fetchOnlineCount();
    const interval = setInterval(fetchOnlineCount, 15000);
    return () => {
      isMounted.current = false;
      clearInterval(interval);
      if (abortController) abortController.abort();
    };
  }, []);

  return (
    <div
      className="online-counter-block"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: isMobile ? '6px' : '10px',
        padding: isMobile ? '4px 8px' : '8px 12px',
        background: 'rgba(0,255,136,0.08)',
        border: '1px solid rgba(0,255,136,0.35)',
        borderRadius: isMobile ? '10px' : '14px',
        backdropFilter: 'blur(10px)',
        cursor: 'default',
        willChange: 'transform',
      }}
    >
      <div style={{
        fontSize: isMobile ? '16px' : '22px',
        fontWeight: '900',
        color: '#00ff88',
        textShadow: '0 0 15px rgba(0,255,136,0.6)',
        lineHeight: 1,
        transition: 'transform 0.3s ease',
        transform: isGrowing ? 'scale(1.15)' : 'scale(1)',
      }}>
        {onlineCount.toLocaleString('ru-RU')}
      </div>

      {isMobile ? (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
        }}>
          <div style={{
            color: '#888',
            fontSize: '7px',
            lineHeight: 1,
          }}>Онлайн</div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            color: '#00ff88',
            fontSize: '10px',
            fontWeight: '700',
            letterSpacing: '0.05em',
          }}>
            <span style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: '#00ff88',
              boxShadow: '0 0 10px #00ff88',
              animation: 'pulse 1.5s ease-in-out infinite',
              flexShrink: 0,
            }} />
            LIVE
          </div>
          <div style={{ color: '#888', fontSize: '9px' }}>
            Космонавтов
          </div>
        </div>
      )}
    </div>
  );
});

// ============= CONVERSATION TIMER =============
const ConversationTimer = React.memo(({ isActive, isMobile }) => {
  const [time, setTime] = useState(0);
  useEffect(() => {
    let interval = null;
    if (isActive) interval = setInterval(() => setTime(t => t + 1), 1000);
    return () => { if (interval) clearInterval(interval); };
  }, [isActive]);
  useEffect(() => { if (!isActive) setTime(0); }, [isActive]);
  const formatTime = (s) => `${String(Math.floor(s/3600)).padStart(2,'0')}:${String(Math.floor((s%3600)/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  if (!isActive) return null;
  return (
    <div className="conversation-timer cosmic-timer">
      <div className="timer-content"><div className="timer-icon">⏱</div><div className="timer-info"><div className="timer-label">{isMobile ? 'Время' : 'Время связи'}</div><div className="timer-display">{formatTime(time)}</div></div></div>
    </div>
  );
});

// ============= REPORT MODAL =============
const ReportModal = React.memo(({ isVisible, onClose, onSendReport, isSending = false, isMobile }) => {
  const [reason, setReason] = useState('spam');
  const [description, setDescription] = useState('');
  useEffect(() => { if (!isVisible) { setReason('spam'); setDescription(''); } }, [isVisible]);
  const handleSend = () => { if (!description.trim()) { alert('Опишите проблему'); return; } onSendReport(reason, description); };
  if (!isVisible) return null;
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(5px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10001 }}>
      <div style={{ width: isMobile ? '90%' : '450px', padding: isMobile ? '20px' : '25px', background: 'linear-gradient(135deg, #1a1a2e, #0a0a1a)', borderRadius: '20px', border: '1px solid rgba(0,255,255,0.3)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2 style={{ fontSize: '1.3rem', color: '#fff', margin: 0 }}>🚨 Сообщить о проблеме</h2>
          <button onClick={onClose} disabled={isSending} style={{ background: 'none', border: 'none', color: '#fff', fontSize: '24px', cursor: 'pointer' }}>×</button>
        </div>
        <label style={{ display: 'block', marginBottom: '8px', color: '#aaa', fontSize: '13px' }}>Причина</label>
        <select value={reason} onChange={e => setReason(e.target.value)} disabled={isSending} style={{ width: '100%', padding: '12px 16px', background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(0,255,255,0.3)', borderRadius: '10px', color: '#fff', marginBottom: '15px' }}>
          <option value="spam">Спам</option>
          <option value="harassment">Оскорбления</option>
          <option value="inappropriate_content">Неуместный контент</option>
          <option value="voice_abuse">Проблемы с голосом</option>
          <option value="other">Другое</option>
        </select>
        <label style={{ display: 'block', marginBottom: '8px', color: '#aaa', fontSize: '13px' }}>Описание</label>
        <textarea value={description} onChange={e => setDescription(e.target.value)} rows={4} disabled={isSending} placeholder="Опишите ситуацию..." style={{ width: '100%', padding: '12px 16px', background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(0,255,255,0.3)', borderRadius: '10px', color: '#fff', resize: 'vertical', marginBottom: '20px' }} />
        <div style={{ display: 'flex', gap: '12px' }}>
          <button onClick={handleSend} disabled={isSending || !description.trim()} style={{ flex: 1, padding: '12px', background: 'linear-gradient(135deg, #00d2ff, #3a7bd5)', border: 'none', borderRadius: '10px', color: 'white', fontWeight: 'bold', cursor: isSending ? 'not-allowed' : 'pointer' }}>{isSending ? '⏳ Отправка...' : '🚀 Отправить'}</button>
          <button onClick={onClose} disabled={isSending} style={{ padding: '12px 20px', background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '10px', color: '#fff', cursor: 'pointer' }}>Отмена</button>
        </div>
      </div>
    </div>
  );
});

// ============= NICKNAME MODAL =============
const NicknameModal = React.memo(({ isVisible, onConfirm, isMobile }) => {
  const [nickname, setNickname] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  useEffect(() => { if (isVisible) { const saved = localStorage.getItem('cosmic_nickname'); if (saved) setNickname(saved); } }, [isVisible]);
  const handleConfirm = () => {
    const trimmed = nickname.trim();
    if (trimmed.length < 2) { alert('Минимум 2 символа'); return; }
    if (trimmed.length > 20) { alert('Максимум 20 символов'); return; }
    setIsLoading(true); localStorage.setItem('cosmic_nickname', trimmed);
    setTimeout(() => { setIsLoading(false); onConfirm(trimmed); }, 500);
  };
  if (!isVisible) return null;
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(20px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100000 }}>
      <div style={{ width: isMobile ? '92%' : '440px', padding: isMobile ? '30px 24px' : '40px 36px', background: 'linear-gradient(145deg, rgba(15,25,50,0.95), rgba(8,12,30,0.98))', borderRadius: '24px', border: '2px solid rgba(0,255,255,0.25)', textAlign: 'center' }}>
        <div style={{ fontSize: '64px', marginBottom: '16px' }}>🚀</div>
        <h2 style={{ fontSize: '1.8rem', fontWeight: '800', background: 'linear-gradient(135deg, #00ffff, #ff00ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '8px' }}>Добро пожаловать!</h2>
        <p style={{ color: '#9aa3b5', marginBottom: '24px' }}>Придумайте ник</p>
        <input type="text" value={nickname} onChange={e => setNickname(e.target.value)} placeholder="Ваш ник..." maxLength="20" autoFocus style={{ width: '100%', padding: '16px', background: 'rgba(0,0,0,0.4)', border: '2px solid rgba(0,255,255,0.2)', borderRadius: '14px', color: '#fff', fontSize: '1.1rem', outline: 'none', marginBottom: '20px' }} />
        <button onClick={handleConfirm} disabled={isLoading || nickname.trim().length < 2} style={{ width: '100%', padding: '16px', background: isLoading || nickname.trim().length < 2 ? '#444' : 'linear-gradient(135deg, #00d2ff, #7b2ff7, #ff00c8)', border: 'none', borderRadius: '14px', color: 'white', fontSize: '1.1rem', fontWeight: '700', cursor: isLoading || nickname.trim().length < 2 ? 'not-allowed' : 'pointer' }}>
          {isLoading ? '⏳ Установка...' : '⭐ Установить ник'}
        </button>
      </div>
    </div>
  );
});

// ============= VIDEO CAMERA =============
const VideoCameraInner = ({ participant, volume, onVolumeChange, onReport, isMuted, isMobile, userId, isDimmed: isDimmedProp }) => {
  const videoRef = useRef(null);
  const audioRef = useRef(null);
  const volumeWrapRef = useRef(null);
  const [showVolumeSlider, setShowVolumeSlider] = useState(false);
  const [localVolume, setLocalVolume] = useState(volume || 0.8);

  const attachedVideoTrackRef = useRef(null);
  const attachedAudioTrackRef = useRef(null);

  const isOwnCamera = participant?.id === userId || participant?.id === 'local';
  const isEmptySlot = participant?.id?.startsWith('empty_');
  const isActiveCamera = participant?.isActive !== false;
  const isDimmed = isDimmedProp !== undefined ? isDimmedProp : !!participant?.isDimmed;
  const hasVideo = (!!participant?.videoStream || !!participant?._videoTrack) && isActiveCamera;
  const hasAudio = !!participant?._audioTrack || !!participant?.audioStream;
  const showReportButton = !isDimmed && !isOwnCamera && !isEmptySlot && isActiveCamera && (hasVideo || hasAudio) && participant?.id;

  useEffect(() => {
    const videoEl = videoRef.current;
    if (!videoEl) return;

    const track = participant?._videoTrack;

    if (attachedVideoTrackRef.current === track) {
      return;
    }
    attachedVideoTrackRef.current = track;

    if (!track) {
      if (videoEl.srcObject) {
        try { videoEl.srcObject = null; } catch (e) {}
      }
      return;
    }

    const isLiveKitTrack = typeof track.attach === 'function';

    if (isLiveKitTrack) {
      try {
        track.attach(videoEl);
      } catch (e) {
        console.error('❌ [VideoCamera] attach error:', e);
        if (track.mediaStreamTrack) {
          try {
            videoEl.srcObject = new MediaStream([track.mediaStreamTrack]);
          } catch (e2) {}
        }
      }
    } else if (track instanceof MediaStream) {
      if (videoEl.srcObject !== track) {
        videoEl.srcObject = track;
      }
    } else if (track.mediaStreamTrack) {
      try {
        const stream = new MediaStream([track.mediaStreamTrack]);
        if (videoEl.srcObject !== stream) {
          videoEl.srcObject = stream;
        }
      } catch (e) {}
    }

    videoEl.muted = true;
    const p = videoEl.play();
    if (p && typeof p.then === 'function') {
      p.catch(e => { if (e.name !== 'AbortError') log('Video play error:', e); });
    }

    return () => {
      if (isLiveKitTrack && attachedVideoTrackRef.current === track) {
        try { track.detach(videoEl); } catch (e) {}
        attachedVideoTrackRef.current = null;
      }
    };
  }, [participant?._videoTrack]);

  useEffect(() => {
    const audioEl = audioRef.current;
    if (!audioEl || isOwnCamera) return;

    const track = participant?._audioTrack;

    if (attachedAudioTrackRef.current === track) {
      return;
    }
    attachedAudioTrackRef.current = track;

    if (!track) {
      if (audioEl.srcObject) {
        try { audioEl.srcObject = null; } catch (e) {}
      }
      return;
    }

    const isLiveKitTrack = typeof track.attach === 'function';

    if (isLiveKitTrack) {
      try {
        track.attach(audioEl);
      } catch (e) {
        if (track.mediaStreamTrack) {
          try {
            audioEl.srcObject = new MediaStream([track.mediaStreamTrack]);
          } catch (e2) {}
        }
      }
    } else if (track instanceof MediaStream) {
      if (audioEl.srcObject !== track) {
        audioEl.srcObject = track;
      }
    } else if (track.mediaStreamTrack) {
      try {
        const stream = new MediaStream([track.mediaStreamTrack]);
        if (audioEl.srcObject !== stream) {
          audioEl.srcObject = stream;
        }
      } catch (e) {}
    }

    audioEl.volume = localVolume;
    audioEl.play().catch(() => {});

    return () => {
      if (isLiveKitTrack && attachedAudioTrackRef.current === track) {
        try { track.detach(audioEl); } catch (e) {}
        attachedAudioTrackRef.current = null;
      }
    };
  }, [participant?._audioTrack, isOwnCamera]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = localVolume;
    }
  }, [localVolume]);

  useEffect(() => {
    if (!showVolumeSlider || !isMobile) return;
    const handler = (e) => {
      if (volumeWrapRef.current && !volumeWrapRef.current.contains(e.target)) {
        setShowVolumeSlider(false);
      }
    };
    document.addEventListener('touchstart', handler);
    document.addEventListener('mousedown', handler);
    return () => {
      document.removeEventListener('touchstart', handler);
      document.removeEventListener('mousedown', handler);
    };
  }, [showVolumeSlider, isMobile]);

  const handleVolumeChange = useCallback((e) => {
    const v = parseFloat(e.target.value);
    setLocalVolume(v);
    if (onVolumeChange) onVolumeChange(participant?.id, v);
  }, [onVolumeChange, participant?.id]);

  const handleReportClick = useCallback(() => {
    if (onReport) onReport(participant?.id, participant?.name);
  }, [onReport, participant?.id, participant?.name]);

  const getParticipantName = () => {
    if (!participant) return 'Ожидание';
    if (participant.name) return typeof participant.name === 'string' ? participant.name : 'Участник';
    return participant.isActive !== false ? 'Ожидание' : 'Свободно';
  };

  return (
    <div
      className="video-camera cosmic-camera"
      style={{
        position: 'relative',
        background: 'linear-gradient(135deg, #1a1a2e, #0a0a1a)',
        borderRadius: '10px',
        overflow: 'hidden',
        border: isActiveCamera ? '1.5px solid rgba(0,255,255,0.3)' : '1.5px solid rgba(100,100,100,0.3)',
        width: '100%',
        height: '100%',
        minHeight: isMobile ? '90px' : '120px',
        willChange: 'transform',
        transform: 'translateZ(0)',
        backfaceVisibility: 'hidden',
      }}
    >
      {isDimmed ? (
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          background: '#000',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 99999,
        }}>
          <span style={{ fontSize: isMobile ? '14px' : '18px', color: '#333' }}>👤</span>
        </div>
      ) : (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            backgroundColor: '#000',
            display: hasVideo ? 'block' : 'none',
          }}
        />
      )}

      {!hasVideo && !isDimmed && (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #0f0f23, #050510)',
        }}>
          <div style={{ fontSize: isMobile ? '20px' : '28px' }}>👤</div>
          <div style={{ fontSize: isMobile ? '7px' : '9px', color: isActiveCamera ? '#888' : '#444' }}>{getParticipantName()}</div>
        </div>
      )}

      <div style={{ position: 'absolute', bottom: '3px', left: '3px', background: 'rgba(0,0,0,0.6)', padding: '2px 6px', borderRadius: '8px', fontSize: isMobile ? '6px' : '8px', color: isActiveCamera ? '#00ffff' : '#666', zIndex: 10 }}>
        {getParticipantName()}
      </div>

      {showReportButton && (
        <button
          className="report-btn"
          onClick={handleReportClick}
          style={{
            position: 'absolute', top: '6px', right: '6px',
            background: 'linear-gradient(135deg, #ff4757, #ff3838)',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '12px',
            padding: isMobile ? '4px 8px' : '6px 12px',
            color: 'white',
            fontSize: isMobile ? '9px' : '11px',
            fontWeight: 'bold',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            zIndex: 999999,
          }}
        >
          <span style={{ fontSize: isMobile ? '10px' : '13px' }}>🚨</span>
          {!isMobile && <span>Жалоба</span>}
        </button>
      )}

      {!isDimmed && !isOwnCamera && !isEmptySlot && isActiveCamera && (hasVideo || hasAudio) && (
        <div
          ref={volumeWrapRef}
          onMouseEnter={() => { if (!isMobile) setShowVolumeSlider(true); }}
          onMouseLeave={() => { if (!isMobile) setShowVolumeSlider(false); }}
          onClick={(e) => {
            if (isMobile) {
              e.stopPropagation();
              setShowVolumeSlider(prev => !prev);
            }
          }}
          style={{
            position: 'absolute', bottom: '6px', right: '6px',
            background: 'rgba(0,0,0,0.75)',
            borderRadius: '14px',
            padding: showVolumeSlider ? '6px 12px' : '6px 10px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            zIndex: 999999,
            cursor: 'pointer',
            border: '1px solid rgba(0,255,255,0.3)',
            WebkitTapHighlightColor: 'transparent',
          }}
        >
          <span style={{ fontSize: isMobile ? '16px' : '18px', color: '#00ffff' }}>🔊</span>
          {showVolumeSlider && (
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={localVolume}
              onChange={handleVolumeChange}
              onClick={(e) => e.stopPropagation()}
              onTouchStart={(e) => e.stopPropagation()}
              style={{
                width: isMobile ? '90px' : '110px',
                height: isMobile ? '6px' : '4px',
                accentColor: '#00ffff',
                cursor: 'pointer',
              }}
            />
          )}
        </div>
      )}

      {participant?.isMuted && (
        <div style={{ position: 'absolute', top: '6px', left: '6px', background: 'rgba(255,0,0,0.7)', borderRadius: '50%', width: isMobile ? '18px' : '22px', height: isMobile ? '18px' : '22px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: isMobile ? '9px' : '11px', zIndex: 999999 }}>🔇</div>
      )}

      {!isOwnCamera && <audio ref={audioRef} autoPlay style={{ display: 'none' }} />}
    </div>
  );
};

const VideoCamera = React.memo(VideoCameraInner, (prev, next) => {
  if (prev.participant?.id !== next.participant?.id) return false;
  if (prev.participant?._videoTrack !== next.participant?._videoTrack) return false;
  if (prev.participant?._audioTrack !== next.participant?._audioTrack) return false;
  if (prev.participant?.isActive !== next.participant?.isActive) return false;
  if (prev.participant?.name !== next.participant?.name) return false;
  if (prev.participant?.isMuted !== next.participant?.isMuted) return false;
  if (prev.isDimmed !== next.isDimmed) return false;
  if (prev.isMobile !== next.isMobile) return false;
  if (prev.userId !== next.userId) return false;
  return true;
});

// ============= MEMOIZED CAMERA SLOT (1-на-1) =============
const MemoizedCameraSlot2 = React.memo(({ sp, volume, handleVolumeChange, handleReport, isMobile, userId }) => {
  return (
    <div
      style={{
        flex: 1,
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '16px',
        background: 'linear-gradient(135deg, #1a1a2e, #0a0a1a)',
        border: sp.isActive !== false
          ? '2px solid rgba(0,255,255,0.4)'
          : '2px solid rgba(100,100,100,0.2)',
        minHeight: '100px',
      }}
    >
      <VideoCamera
        participant={sp}
        volume={volume}
        onVolumeChange={handleVolumeChange}
        onReport={handleReport}
        isMuted={false}
        isMobile={isMobile}
        userId={userId}
      />
    </div>
  );
});

// ============= MEMOIZED CAMERA SLOT (6-режим) =============
const MemoizedCameraSlot = React.memo(({ sp, isThisMaximized, isHidden, volume, handleCameraClick, handleVolumeChange, handleReport, isMobile, userId }) => {
  return (
    <div
      onClick={() => handleCameraClick(sp.id)}
      style={{
        width: '100%',
        height: '100%',
        minHeight: 0,
        position: 'relative',
        overflow: 'hidden',
        borderRadius: '14px',
        background: '#000',
        border: isThisMaximized
          ? '2.5px solid rgba(0,210,255,0.9)'
          : (sp.isActive !== false
            ? '2.5px solid rgba(0,255,255,0.4)'
            : '2px solid rgba(100,100,100,0.25)'),
        cursor: sp.id.startsWith('empty_') ? 'default' : 'pointer',
        transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
        boxShadow: isThisMaximized ? '0 0 25px rgba(0,210,255,0.7), inset 0 0 15px rgba(0,210,255,0.2)' : 'none',
      }}
    >
      <VideoCamera
        participant={sp}
        isDimmed={isHidden}
        volume={volume}
        onVolumeChange={handleVolumeChange}
        onReport={handleReport}
        isMuted={false}
        isMobile={isMobile}
        userId={userId}
      />
    </div>
  );
});

// ============= VIDEO 2 MODE =============
const Video2Mode = React.memo(({ isActive, participants, messages, onSendMessage, onReport, onExit, onNext, isMobile, userId, onToggleMic, isMicMuted }) => {
  const [volumes, setVolumes] = useState({});
  const messagesEndRef = useRef(null);
  const [nickname, setNickname] = useState(() => localStorage.getItem('cosmic_nickname') || '');
  const [showNicknameModal, setShowNicknameModal] = useState(false);
  const [isSearchingNext, setIsSearchingNext] = useState(false);

  const messagesRef = useRef(messages);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if (isActive) {
      const saved = localStorage.getItem('cosmic_nickname');
      if (saved && saved.trim().length >= 2) { setNickname(saved); setShowNicknameModal(false); }
      else setShowNicknameModal(true);
    }
  }, [isActive]);

  const handleNicknameConfirm = useCallback((n) => { setNickname(n); setShowNicknameModal(false); }, []);

  useEffect(() => {
    const el = messagesEndRef.current;
    if (!el) return;
    const container = el.parentElement;
    if (container) {
      container.scrollTop = 0;
    }
  }, [messages]);

  const handleVolumeChange = useCallback((id, v) => setVolumes(prev => ({ ...prev, [id]: v })), []);
  const handleReport = useCallback((id, name) => onReport(id, name), [onReport]);

  const handleNextClick = useCallback(() => {
    if (isSearchingNext) { setIsSearchingNext(false); onNext && onNext(true); }
    else { setIsSearchingNext(true); onNext && onNext(false); }
  }, [isSearchingNext, onNext]);

  useEffect(() => {
    const hasPartner = participants.some(p => p && p.id !== userId && p.id !== 'local' && p.isActive !== false);
    if (hasPartner) setIsSearchingNext(false);
  }, [participants, userId]);

  const filled = useMemo(() => {
    const arr = [...participants];
    for (let i = arr.length; i < 2; i++) arr.push({ id: `empty_${i}`, name: null, videoStream: null, audioStream: null, isMuted: false, isActive: false });
    return arr.map(p => (p.id === userId || p.id === 'local') ? { ...p, name: nickname || p.name || 'Вы', audioStream: null } : p);
  }, [participants, userId, nickname]);

  if (!isActive) return null;

  return (
    <Portal>
      <NicknameModal isVisible={showNicknameModal} onConfirm={handleNicknameConfirm} isMobile={isMobile} />
      <div style={{ position: 'fixed', inset: 0, background: 'linear-gradient(135deg, #0a0a2a, #050518)', zIndex: 1000, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 20px', background: 'linear-gradient(180deg, rgba(0,0,0,0.9), rgba(0,0,0,0.6))', borderBottom: '1px solid rgba(0,255,255,0.2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ fontSize: '28px' }}>🎥</div>
            <div style={{ fontSize: '1.3rem', fontWeight: '800', background: 'linear-gradient(90deg, #00d2ff, #7b2ff7, #ff00c8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>✦ COSMIC-VOICE ✦</div>
          </div>
          <div style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', fontSize: '0.8rem', color: '#00ffff', padding: '4px 12px', background: 'rgba(0,0,0,0.5)', borderRadius: '20px', border: '1px solid rgba(0,255,255,0.2)' }}>◇ ВИДЕОЧАТ 1 НА 1 ◇</div>
          <button onClick={onExit} style={{ padding: '8px 18px', background: 'linear-gradient(135deg, #ff4757, #ff3838)', border: 'none', borderRadius: '10px', color: 'white', fontWeight: 'bold', cursor: 'pointer' }}>🚪 Выйти</button>
        </div>

        <div style={{ display: 'flex', flex: 1, flexDirection: isMobile ? 'column' : 'row', gap: '12px', padding: '12px', overflow: 'hidden', minHeight: 0 }}>
          <div style={{ flex: isMobile ? 'none' : '0 0 42%', display: 'flex', flexDirection: 'column', gap: '12px', minHeight: isMobile ? '220px' : '0' }}>
            {filled.slice().reverse().map((sp, i) => (
              <MemoizedCameraSlot2
                key={sp.id || i}
                sp={sp}
                volume={volumes[sp.id] || 0.8}
                handleVolumeChange={handleVolumeChange}
                handleReport={handleReport}
                isMobile={isMobile}
                userId={userId}
              />
            ))}
          </div>

          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', minHeight: isMobile ? '260px' : '0' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'linear-gradient(135deg, rgba(10,10,30,0.95), rgba(5,5,20,0.98))', borderRadius: '16px', border: '2px solid rgba(0,255,255,0.25)', overflow: 'hidden', minHeight: 0 }}>
              {/* 🆕 Проверяем, есть ли собеседник (кроме себя) */}
              {participants.filter(p => p.id !== userId && p.id !== 'local' && p.isActive !== false).length > 0 ? (
                <ChatPanel
                  messages={messages}
                  messagesEndRef={messagesEndRef}
                  onSend={onSendMessage}
                  nickname={nickname}
                  isMobile={isMobile}
                  title="Общий чат"
                />
              ) : (
                <div style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '20px',
                  textAlign: 'center',
                  color: '#8a93a8',
                }}>
                  <div style={{ fontSize: '48px', marginBottom: '16px', animation: 'pulse 2s ease-in-out infinite' }}>⏳</div>
                  <div style={{ fontSize: isMobile ? '14px' : '16px', fontWeight: '600', color: '#00ffff', marginBottom: '8px' }}>
                    Ожидание собеседника...
                  </div>
                  <div style={{ fontSize: isMobile ? '12px' : '13px', lineHeight: 1.5 }}>
                    Чат станет доступен, когда кто-то подключится.
                  </div>
                </div>
              )}
            </div>

                        <div style={{
              flexShrink: 0,
              display: 'flex',
              gap: isMobile ? '6px' : '10px',
              padding: isMobile ? '8px 10px' : '12px 16px',
              background: 'linear-gradient(135deg, rgba(10,15,40,0.85), rgba(5,8,25,0.95))',
              borderRadius: '14px',
              border: '1.5px solid rgba(0,255,255,0.3)',
              boxShadow: '0 4px 20px rgba(0,247,255,0.15)',
              justifyContent: 'center',
              alignItems: 'stretch',
            }}>
              {/* 🆕 ВЫЙТИ теперь первая (слева) */}
              <ControlButton
                icon="🚪"
                label="Выйти"
                onClick={onExit}
                variant="danger"
                isMobile={isMobile}
                compact={true}
              />
              <ControlButton
                icon={isMicMuted ? '🔇' : '🎤'}
                label={isMicMuted ? 'Вкл' : 'Выкл'}
                onClick={onToggleMic}
                variant={isMicMuted ? 'danger' : 'success'}
                isActive={!isMicMuted}
                isMobile={isMobile}
                compact={true}
              />
              {/* 🆕 ДАЛЬШЕ теперь последняя (справа) */}
              <ControlButton
                icon={isSearchingNext ? '⏹️' : '⏭️'}
                label={isSearchingNext ? 'Отмена' : 'Дальше'}
                onClick={handleNextClick}
                variant={isSearchingNext ? 'danger' : 'primary'}
                isMobile={isMobile}
                compact={true}
              />
            </div>
          </div>
        </div>
      </div>
    </Portal>
  );
}, (prev, next) => {
  if (prev.isActive !== next.isActive) return false;
  if (prev.participants !== next.participants) return false;
  
  if (prev.isMobile !== next.isMobile) return false;
  if (prev.userId !== next.userId) return false;
  if (prev.isMicMuted !== next.isMicMuted) return false;
  return true;
});

// ============= CHAT INPUT BAR =============
const ChatInputBar = React.memo(({ isRateLimited, cooldownSeconds, nickname, onSend, isMobile }) => {
  const [inputMessage, setInputMessage] = useState('');
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);

  const onSendRef = useRef(onSend);
  useEffect(() => {
    onSendRef.current = onSend;
  }, [onSend]);

  const emojis = useMemo(() => [
    '🚀','🛸','🛰️','🌌','🌠','⭐','🌟','✨','💫','☄️','🌙','🌕','🪐','🌍','🌏','🌎','👽','👾','🤖',
    '👨‍🚀','👩‍🚀','🧑‍🚀','🧑‍💻',
    '😀','😂','🤣','😊','😍','🥰','😘','😎','🤩','🥳','🤔','😅','😇','🙃','😴','🤯','🥺','😢','😭','😡','🤬','😱','😨',
    '👍','👎','👏','🙌','🤝','✌️','🤞','👋','🖖','💪','🙏','👌',
    '❤️','🧡','💛','💚','💙','💜','🖤','🤍','💖','💘','💝','🔥','⚡','💥','🎉','🎊','🎈','🎁','🏆','🥇',
    '🎵','🎶','🎤','🎧','🎸','🎹','🎮','🕹️','📹','📷','📸','💬','💭','🗨️','📢',
    '🍕','🍔','🍟','🌮','🍣','🍩','🍪','🍫','🍭','🍿','☕','🍵','🥤','🍺','🥂','🍾',
    '🐶','🐱','🦊','🐼','🐨','🦁','🐯','🦄','🐲','🐉','🌸','🌺','🌈','☀️','🌤️',
    '💯','✅','❌','⚠️','❓','❗','💤','💡','🔔','🔒','🔑','💎','🎯','🎲','🃏',
  ], []);

  const handleSend = useCallback(() => {
    const text = inputMessage.trim();
    if (!text || isRateLimited) return;
    const ok = onSendRef.current(text, nickname);
    if (ok !== false) {
      setInputMessage('');
      setShowEmojiPicker(false);
    }
  }, [inputMessage, isRateLimited, nickname]);

  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const addEmoji = useCallback((e) => {
    setInputMessage(p => p + e);
    setShowEmojiPicker(false);
  }, []);

  const toggleEmojiPicker = useCallback(() => {
    document.activeElement?.blur?.();
    setShowEmojiPicker(prev => !prev);
  }, []);

  return (
    <div style={{ padding: '6px 12px', borderTop: '2px solid rgba(0,255,255,0.12)', display: 'flex', gap: '4px', alignItems: 'center', flexShrink: 0, background: 'rgba(0,0,0,0.25)', minHeight: '40px' }}>
      <div style={{ position: 'relative' }}>
        <button
          onClick={toggleEmojiPicker}
          disabled={isRateLimited}
          title="Смайлы"
          style={{
            background: showEmojiPicker
              ? 'linear-gradient(135deg, #00d2ff, #7b2ff7)'
              : 'linear-gradient(135deg, #feca57, #ff9f43)',
            border: showEmojiPicker ? '1px solid rgba(0,255,255,0.5)' : 'none',
            borderRadius: '50%',
            width: '30px',
            height: '30px',
            fontSize: '15px',
            cursor: isRateLimited ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            WebkitTapHighlightColor: 'transparent',
            padding: 0,
            lineHeight: 0,
          }}
        >{showEmojiPicker ? '✕' : '🚀'}</button>

        {showEmojiPicker && (
          <div
            className="chat-scroll-area"
            style={{
              position: 'absolute',
              bottom: '44px',
              left: 0,
              background: 'linear-gradient(135deg, rgba(15,10,35,0.98), rgba(8,5,25,0.99))',
              borderRadius: '12px',
              padding: '8px',
              border: '1px solid rgba(0,255,255,0.35)',
              boxShadow: '0 10px 40px rgba(0,0,0,0.7), 0 0 20px rgba(0,255,255,0.15)',
              display: 'grid',
              gridTemplateColumns: 'repeat(7, 1fr)',
              gap: '2px',
              zIndex: 100,
              width: isMobile ? '260px' : '280px',
              maxHeight: '240px',
              overflowY: 'auto',
              overflowX: 'hidden',
              WebkitOverflowScrolling: 'touch',
              touchAction: 'pan-y',
              overscrollBehavior: 'contain',
            }}
          >
            {emojis.map((e, i) => (
              <button
                key={i}
                onClick={() => addEmoji(e)}
                title={e}
                style={{
                  background: 'transparent',
                  border: 'none',
                  fontSize: '20px',
                  cursor: 'pointer',
                  padding: '4px',
                  borderRadius: '6px',
                  lineHeight: 1,
                  WebkitTapHighlightColor: 'transparent',
                  transition: 'background 0.15s ease',
                }}
                onMouseEnter={(ev) => { ev.currentTarget.style.background = 'rgba(0,255,255,0.12)'; }}
                onMouseLeave={(ev) => { ev.currentTarget.style.background = 'transparent'; }}
              >
                {e}
              </button>
            ))}
          </div>
        )}
      </div>

      <input
        type="text"
        value={inputMessage}
        onChange={e => setInputMessage(e.target.value)}
        onKeyPress={handleKeyPress}
        disabled={isRateLimited}
        placeholder={isRateLimited ? `⛔ ${cooldownSeconds}с` : "Написать..."}
        style={{
          flex: 1,
          background: 'rgba(0,0,0,0.4)',
          border: `2px solid ${isRateLimited ? 'rgba(255,71,87,0.5)' : 'rgba(0,255,255,0.2)'}`,
          borderRadius: '12px',
          padding: '6px 12px',
          color: '#fff',
          fontSize: '12px',
          outline: 'none',
          minHeight: '34px',
        }}
      />

      <button
        onClick={handleSend}
        disabled={isRateLimited || !inputMessage.trim()}
        title="Отправить"
        style={{
          background: isRateLimited || !inputMessage.trim()
            ? 'rgba(60,60,80,0.4)'
            : 'linear-gradient(135deg, #00d2ff, #7b2ff7, #ff00c8)',
          border: isRateLimited || !inputMessage.trim()
            ? '1px solid rgba(255,255,255,0.08)'
            : '1px solid rgba(0,255,255,0.4)',
          borderRadius: '50%',
          width: '34px',
          height: '34px',
          minWidth: '34px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: isRateLimited || !inputMessage.trim() ? 'not-allowed' : 'pointer',
          boxShadow: isRateLimited || !inputMessage.trim()
            ? 'none'
            : '0 0 12px rgba(0,210,255,0.5)',
          padding: 0,
          lineHeight: 0,
          WebkitTapHighlightColor: 'transparent',
        }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', margin: 0 }}>
          <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </svg>
      </button>
    </div>
  );
}, (prev, next) => {
  if (prev.isRateLimited !== next.isRateLimited) return false;
  if (prev.cooldownSeconds !== next.cooldownSeconds) return false;
  if (prev.nickname !== next.nickname) return false;
  if (prev.isMobile !== next.isMobile) return false;
  return true;
});

// ============= CHAT PANEL =============
// ============= CHAT PANEL (С ВИРТУАЛИЗАЦИЕЙ, ПРИЖАТО К НИЗУ) =============
const ChatPanelInner = ({ messages, messagesEndRef, onSend, nickname, isMobile, title = 'Общий чат', isRateLimited = false, cooldownSeconds = 0 }) => {
  const parentRef = useRef(null);

  // Создаем перевернутый массив (новые в начале) для виртуализатора
  const reversedMessages = useMemo(() => {
    return messages ? [...messages].reverse() : [];
  }, [messages]);

  const rowVirtualizer = useVirtualizer({
    count: reversedMessages.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 60,
    overscan: 5,
  });

  // Скролл в самый низ (в column-reverse это scrollTop = 0) при новых сообщениях
  useEffect(() => {
    const container = parentRef.current;
    if (!container) return;
    container.scrollTop = 0; 
  }, [reversedMessages.length]);

  const renderMessage = useCallback((msg, idx) => {
    if (!msg || typeof msg !== 'object') return null;
    let text = '';
    if (msg.text) text = typeof msg.text === 'string' ? msg.text : String(msg.text);
    else if (msg.message) text = typeof msg.message === 'string' ? msg.message : String(msg.message);
    if (!text) return null;

    const senderName = msg.senderNickname || msg.senderName || msg.senderId?.slice(0, 6) || 'User';
    const isOwn = msg.isOwn || false;
    const ts = msg.timestamp
      ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : '';

    const avatarBg = isOwn
      ? 'linear-gradient(135deg, #8b5cf6, #6366f1)'
      : 'linear-gradient(135deg, #3b82f6, #0ea5e9)';
    const avatarIcon = isOwn ? '👨‍🚀' : '👽';

    return (
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '6px', width: '100%' }}>
        <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: avatarBg, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: '14px', boxShadow: '0 2px 8px rgba(0,0,0,0.3)' }}>{avatarIcon}</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: '10px', color: '#8a93a8', marginBottom: '3px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ color: isOwn ? '#a78bfa' : '#7dd3fc' }}>{senderName}{isOwn && ' (Вы)'}</span>
            <span style={{ color: '#555', fontWeight: '400' }}>• {ts}</span>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.08)', padding: '6px 12px', borderRadius: '12px', fontSize: '12px', color: '#e2e8f0', lineHeight: 1.4, wordBreak: 'break-word', display: 'inline-block', maxWidth: '100%' }}>{text}</div>
        </div>
      </div>
    );
  }, []);

  if (!messages || messages.length === 0) {
    return (
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div style={{ position: 'relative', padding: '10px 16px', borderBottom: '2px solid rgba(0,255,255,0.15)', background: 'rgba(0,0,0,0.35)', color: '#00ffff', fontWeight: 'bold', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', flexShrink: 0 }}>
          <span>💬</span><span>{title}</span>
          <span style={{ position: 'absolute', right: '16px', top: '50%', transform: 'translateY(-50%)', fontSize: '10px', color: '#666', fontWeight: 'normal' }}>0</span>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60px' }}>
          <div style={{ textAlign: 'center', color: '#666', fontSize: '12px' }}>Напишите первое сообщение...</div>
        </div>
        <ChatInputBar isRateLimited={isRateLimited} cooldownSeconds={cooldownSeconds} nickname={nickname} onSend={onSend} isMobile={isMobile} />
      </div>
    );
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      {/* Заголовок */}
      <div style={{ position: 'relative', padding: '10px 16px', borderBottom: '2px solid rgba(0,255,255,0.15)', background: 'rgba(0,0,0,0.35)', color: '#00ffff', fontWeight: 'bold', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', flexShrink: 0 }}>
        <span>💬</span><span>{title}</span>
        <span style={{ position: 'absolute', right: '16px', top: '50%', transform: 'translateY(-50%)', fontSize: '10px', color: '#666', fontWeight: 'normal' }}>{messages.length}</span>
      </div>

      {/* 🆕 ВИРТУАЛИЗИРОВАННЫЙ СПИСОК (column-reverse + scrollTop 0) */}
      <div
        ref={parentRef}
        className="chat-scroll-area"
        style={{
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
          padding: '12px 12px 0 12px',
          display: 'flex',
          flexDirection: 'column-reverse', // 🆕 Переворачиваем контейнер
          minHeight: '60px',
          WebkitOverflowScrolling: 'touch',
          touchAction: 'pan-y',
          overscrollBehavior: 'contain',
          position: 'relative',
        }}
      >
        <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, width: '100%', position: 'relative', flexShrink: 0 }}>
          {rowVirtualizer.getVirtualItems().map((virtualRow) => {
            const msg = reversedMessages[virtualRow.index];
            if (!msg) return null;

            return (
              <div
                key={virtualRow.key}
                ref={rowVirtualizer.measureElement}
                data-index={virtualRow.index}
                style={{
                  position: 'absolute',
                  // 🆕 Инвертируем позиционирование, так как контейнер перевернут
                  bottom: 0,
                  left: 0,
                  width: '100%',
                  transform: `translateY(-${virtualRow.start}px)`,
                }}
              >
                {renderMessage(msg, virtualRow.index)}
              </div>
            );
          })}
        </div>
        <div ref={messagesEndRef} />
      </div>

      <ChatInputBar isRateLimited={isRateLimited} cooldownSeconds={cooldownSeconds} nickname={nickname} onSend={onSend} isMobile={isMobile} />
    </div>
  );
};

const ChatPanel = React.memo(ChatPanelInner, (prev, next) => {
  if (prev.messages !== next.messages) return false;
  if (prev.isRateLimited !== next.isRateLimited) return false;
  if (prev.cooldownSeconds !== next.cooldownSeconds) return false;
  if (prev.isMobile !== next.isMobile) return false;
  if (prev.nickname !== next.nickname) return false;
  if (prev.title !== next.title) return false;
  return true;
});

// ============= ВЫНЕСЕННЫЙ ЧАТ ДЛЯ 6-РЕЖИМА =============
const Video6Chat = React.memo(({
  messages,
  messagesEndRef,
  onSendMessage,
  nickname,
  isMobile,
  isRateLimited,
  cooldownSeconds,
  isMaximized,
}) => {
  return (
    <div style={{
      flex: 1,
      minHeight: 0,
      display: 'flex',
      flexDirection: 'column',
      background: 'linear-gradient(135deg, rgba(10,10,30,0.95), rgba(5,5,20,0.98))',
      borderRadius: '16px',
      border: '2px solid rgba(0,255,255,0.25)',
      overflow: 'hidden',
      transition: 'flex 0.3s ease',
    }}>
      <div style={{
        flexShrink: 0,
        padding: isMobile ? '4px 4px' : '6px 8px',
        textAlign: 'center',
        borderBottom: '1px solid rgba(0,255,255,0.15)',
        background: 'rgba(0,0,0,0.25)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div style={{
          fontSize: isMobile ? '1rem' : '1.5rem',
          fontWeight: '900',
          letterSpacing: isMobile ? '0.08em' : '0.15em',
          background: 'linear-gradient(135deg, #00ffff, #ff00ff, #00ffff)',
          backgroundSize: '300% 300%',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          animation: 'neonPulse 3s ease-in-out infinite, gradientShift 4s ease infinite',
          filter: 'drop-shadow(0 0 12px rgba(0,255,255,0.4))',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          lineHeight: 1.1,
          textAlign: 'center',
          width: '100%',
        }}>✦ COSMIC-VOICE ✦</div>
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <ChatPanel
          messages={messages}
          messagesEndRef={messagesEndRef}
          onSend={onSendMessage}
          nickname={nickname}
          isMobile={isMobile}
          title="Общий чат"
          isRateLimited={isRateLimited}
          cooldownSeconds={cooldownSeconds}
        />
      </div>
    </div>
  );
});

// ============= VIDEO 6 MODE =============
const Video6Mode = React.memo(({ isActive, participants, messages, onSendMessage, onReport, onExit, onNextRoom, onCancelSearch, isSearchingNext, isMobile, userId, onToggleMic, isMicMuted, roomName }) => {
  const [volumes, setVolumes] = useState({});
  const messagesEndRef = useRef(null);
  const [nickname, setNickname] = useState(() => localStorage.getItem('cosmic_nickname') || '');

  const messagesRef = useRef(messages);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const [showNicknameModal, setShowNicknameModal] = useState(false);
  const [lastMessageTime, setLastMessageTime] = useState(0);
  const [messageCount, setMessageCount] = useState(0);
  const [messageWindowStart, setMessageWindowStart] = useState(Date.now());
  const [isRateLimited, setIsRateLimited] = useState(false);
  const [rateLimitTimer, setRateLimitTimer] = useState(null);
  const [cooldownSeconds, setCooldownSeconds] = useState(0);

  const [maximizedCameraId, setMaximizedCameraId] = useState(null);
  const [showCollapseBtn, setShowCollapseBtn] = useState(false);
  const isMaximized = !!maximizedCameraId;

  const SPAM_CONFIG = useMemo(() => ({
    MAX_MESSAGES_PER_WINDOW: 5,
    MESSAGE_WINDOW_MS: 10000,
    MIN_MESSAGE_INTERVAL_MS: 1000,
    COOLDOWN_MS: 30000,
    MAX_MESSAGE_LENGTH: 500,
    MIN_MESSAGE_LENGTH: 1,
    REPEAT_MESSAGE_LIMIT: 3,
  }), []);

  const checkSpam = useCallback((text) => {
    const now = Date.now();
    const t = text.trim();
    if (t.length < SPAM_CONFIG.MIN_MESSAGE_LENGTH) return { allowed: false, reason: 'Сообщение слишком короткое' };
    if (t.length > SPAM_CONFIG.MAX_MESSAGE_LENGTH) return { allowed: false, reason: `Максимум ${SPAM_CONFIG.MAX_MESSAGE_LENGTH} символов` };
    if (now - lastMessageTime < SPAM_CONFIG.MIN_MESSAGE_INTERVAL_MS && lastMessageTime > 0) {
      const w = Math.ceil((SPAM_CONFIG.MIN_MESSAGE_INTERVAL_MS - (now - lastMessageTime)) / 1000);
      return { allowed: false, reason: `Подождите ${w} сек.` };
    }
    const msgs = messagesRef.current;
    if (msgs.length > 0) {
      const last = msgs.slice(-SPAM_CONFIG.REPEAT_MESSAGE_LIMIT);
      const same = last.filter(m => m.text && typeof m.text === 'string' && m.text.toLowerCase() === t.toLowerCase());
      if (same.length >= SPAM_CONFIG.REPEAT_MESSAGE_LIMIT) return { allowed: false, reason: 'Нельзя одинаковые сообщения' };
    }
    if (now - messageWindowStart > SPAM_CONFIG.MESSAGE_WINDOW_MS) {
      setMessageCount(1);
      setMessageWindowStart(now);
    } else {
      if (messageCount >= SPAM_CONFIG.MAX_MESSAGES_PER_WINDOW) {
        return { allowed: false, reason: `Слишком много! Ждите ${Math.ceil(SPAM_CONFIG.COOLDOWN_MS/1000)} сек.`, isRateLimit: true };
      }
      setMessageCount(p => p + 1);
    }
    const urls = (t.match(/(https?:\/\/[^\s]+)|(www\.[^\s]+)/g) || []).length;
    if (urls > 1) return { allowed: false, reason: 'Слишком много ссылок' };
    const upper = (t.match(/[A-ZА-Я]/g) || []).length;
    const letters = (t.match(/[a-zA-Zа-яА-Я]/g) || []).length;
    if (letters > 0 && upper / letters > 0.7 && t.length > 10) return { allowed: false, reason: 'Слишком много капса' };
    return { allowed: true };
  }, [lastMessageTime, messageCount, messageWindowStart, SPAM_CONFIG]);

  const activateRateLimit = useCallback(() => {
    setIsRateLimited(true);
    setCooldownSeconds(Math.ceil(SPAM_CONFIG.COOLDOWN_MS / 1000));
    setMessageCount(0);
    setMessageWindowStart(Date.now());
    if (rateLimitTimer) clearInterval(rateLimitTimer);
    const timer = setInterval(() => {
      setCooldownSeconds(prev => {
        if (prev <= 1) { clearInterval(timer); setIsRateLimited(false); return 0; }
        return prev - 1;
      });
    }, 1000);
    setRateLimitTimer(timer);
  }, [rateLimitTimer, SPAM_CONFIG]);

  const handleSendWithSpamCheck = useCallback((text, nick) => {
    const check = checkSpam(text);
    if (!check.allowed) {
      if (check.isRateLimit) activateRateLimit();
      return false;
    }
    setLastMessageTime(Date.now());
    onSendMessage(text, nick);
    return true;
  }, [checkSpam, activateRateLimit, onSendMessage]);

  useEffect(() => {
    if (isActive) {
      const saved = localStorage.getItem('cosmic_nickname');
      if (saved && saved.trim().length >= 2) {
        setNickname(saved);
        setShowNicknameModal(false);
      } else {
        setShowNicknameModal(true);
      }
    }
  }, [isActive]);

  const handleNicknameConfirm = useCallback((n) => {
    setNickname(n);
    setShowNicknameModal(false);
  }, []);

  useEffect(() => {
    const el = messagesEndRef.current;
    if (!el) return;
    const container = el.parentElement;
    if (container) {
      container.scrollTop = 0;
    }
  }, [messages]);

  const handleVolumeChange = useCallback((id, v) => {
    setVolumes(prev => ({ ...prev, [id]: v }));
  }, []);

  const handleReport = useCallback((id, name) => onReport(id, name), [onReport]);

  const handleNextRoomClick = useCallback(() => {
    if (isSearchingNext) onCancelSearch && onCancelSearch();
    else onNextRoom && onNextRoom();
  }, [isSearchingNext, onCancelSearch, onNextRoom]);

  const handleCameraClick = useCallback((participantId) => {
    if (!participantId || participantId.startsWith('empty_')) return;
    setMaximizedCameraId(prev => prev === participantId ? null : participantId);
    setShowCollapseBtn(false);
  }, []);

  const handleMaximizedClick = useCallback((e) => {
    if (!isMobile) return;
    if (e.target.closest('.camera-collapse-btn')) return;
    setShowCollapseBtn(prev => !prev);
  }, [isMobile]);

  useEffect(() => {
    if (!isMobile || !isMaximized || !showCollapseBtn) return;
    const handler = (e) => {
      if (e.target.closest('.maximized-camera')) return;
      setShowCollapseBtn(false);
    };
    document.addEventListener('touchstart', handler);
    document.addEventListener('mousedown', handler);
    return () => {
      document.removeEventListener('touchstart', handler);
      document.removeEventListener('mousedown', handler);
    };
  }, [isMobile, isMaximized, showCollapseBtn]);

  const participantsKey = useMemo(() => {
    return participants.map(p => `${p.id}:${p._videoTrack?.sid || ''}:${p._audioTrack?.sid || ''}:${p.isActive ? 1 : 0}:${p.name || ''}:${p.isMuted ? 1 : 0}`).join('|');
  }, [participants]);

  const allFilled = useMemo(() => {
    const arr = [...participants];
    for (let i = arr.length; i < 6; i++) {
      arr.push({
        id: `empty_${i}`,
        name: null,
        videoStream: null,
        audioStream: null,
        isMuted: false,
        isActive: false,
      });
    }
    return arr.map(p =>
      (p.id === userId || p.id === 'local')
        ? { ...p, name: nickname || p.name || 'Вы', audioStream: null }
        : p
    );
  }, [participantsKey, userId, nickname]);

  const { left, right } = useMemo(() => ({
    left: allFilled.slice(0, 3),
    right: allFilled.slice(3, 6),
  }), [allFilled]);

  const maximizedParticipant = useMemo(() => {
    if (!maximizedCameraId) return null;
    return allFilled.find(p => p.id === maximizedCameraId) || null;
  }, [maximizedCameraId, allFilled]);

  if (!isActive) return null;

  return (
    <Portal>
      <NicknameModal isVisible={showNicknameModal} onConfirm={handleNicknameConfirm} isMobile={isMobile} />

      <div style={{ position: 'fixed', inset: 0, background: 'linear-gradient(135deg, #0a0a2a, #050518)', zIndex: 1000, overflow: 'hidden' }}>

        <div style={{
          position: 'absolute',
          inset: 0,
          pointerEvents: 'none',
          background: `
            radial-gradient(2px 2px at 15% 20%, rgba(255,255,255,0.7), transparent),
            radial-gradient(1px 1px at 25% 60%, rgba(255,255,255,0.5), transparent),
            radial-gradient(1.5px 1.5px at 80% 30%, rgba(0,255,255,0.6), transparent),
            radial-gradient(1px 1px at 45% 85%, rgba(255,255,255,0.4), transparent),
            radial-gradient(2px 2px at 90% 70%, rgba(255,0,255,0.5), transparent),
            radial-gradient(1px 1px at 60% 15%, rgba(255,255,255,0.6), transparent)
          `,
          zIndex: 0,
        }} />

        <div style={{ position: 'absolute', top: 0, bottom: 0, left: 0, right: 0, display: 'flex', flexDirection: 'row', alignItems: 'stretch', gap: isMobile ? '4px' : '8px', padding: '4px', boxSizing: 'border-box', zIndex: 10 }}>

          <div style={{ flex: 1, display: 'grid', gridTemplateRows: 'repeat(3, 1fr)', gap: isMobile ? '3px' : '6px', minWidth: 0, minHeight: 0 }}>
            {left.map((sp, i) => {
              const isThisMaximized = maximizedCameraId === sp.id;
              const isHidden = isMaximized && maximizedCameraId === sp.id;
              return (
                <MemoizedCameraSlot
                  key={sp.id || `left-${i}`}
                  sp={sp}
                  isThisMaximized={isThisMaximized}
                  isHidden={isHidden}
                  volume={volumes[sp.id] || 0.8}
                  handleCameraClick={handleCameraClick}
                  handleVolumeChange={handleVolumeChange}
                  handleReport={handleReport}
                  isMobile={isMobile}
                  userId={userId}
                />
              );
            })}
          </div>

          <div style={{
            flex: '0 0 auto',
            width: isMobile ? '36%' : '40%',
            minWidth: '100px',
            display: 'flex',
            flexDirection: 'column',
            height: '100%',
            gap: isMobile ? '3px' : '6px',
            boxSizing: 'border-box',
          }}>

            {isMaximized && maximizedParticipant && (
              <div
                className={`maximized-camera ${showCollapseBtn ? 'show-collapse-btn' : ''}`}
                onClick={handleMaximizedClick}
                style={{
                  flex: '0 0 calc(52% - 6px)',
                  minHeight: 0,
                  position: 'relative',
                  overflow: 'hidden',
                  borderRadius: '16px',
                  background: 'linear-gradient(135deg, #1a1a2e, #0a0a1a)',
                  border: '2.5px solid rgba(0,210,255,0.9)',
                  boxShadow: '0 0 35px rgba(0,210,255,0.6), 0 0 60px rgba(0,210,255,0.3), inset 0 0 20px rgba(0,210,255,0.15)',
                  transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
                }}
              >
                <VideoCamera
                  participant={maximizedParticipant}
                  isDimmed={false}
                  volume={volumes[maximizedParticipant.id] || 0.8}
                  onVolumeChange={handleVolumeChange}
                  onReport={handleReport}
                  isMuted={false}
                  isMobile={isMobile}
                  userId={userId}
              />

                <button
                  className="camera-collapse-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    setMaximizedCameraId(null);
                    setShowCollapseBtn(false);
                  }}
                  style={{
                    position: 'absolute', top: '8px', left: '8px',
                    background: 'rgba(0,0,0,0.75)', color: '#00d2ff',
                    padding: isMobile ? '6px 12px' : '6px 14px',
                    borderRadius: '10px', fontSize: isMobile ? '11px' : '12px',
                    fontWeight: '700', border: '1px solid rgba(0,210,255,0.6)',
                    cursor: 'pointer', zIndex: 999999, display: 'flex',
                    alignItems: 'center', gap: '6px',
                    WebkitTapHighlightColor: 'transparent',
                    boxShadow: '0 4px 15px rgba(0,210,255,0.4)',
                    transition: 'opacity 0.25s ease, transform 0.2s ease, background 0.2s ease',
                    fontFamily: 'inherit',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(0,210,255,0.25)';
                    e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,210,255,0.7)';
                    e.currentTarget.style.transform = 'scale(1.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(0,0,0,0.75)';
                    e.currentTarget.style.boxShadow = '0 4px 15px rgba(0,210,255,0.4)';
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                >
                  ✕ Свернуть
                </button>
              </div>
            )}

            {/* 🆕 ЧАТ (рендерится через Portal в этот слот) */}
            <div
              id="video6-chat-slot"
              style={{
                flex: isMaximized ? '1 1 auto' : '1 1 100%',
                minHeight: 0,
                display: 'flex',
                flexDirection: 'column',
              }}
            />

            <div style={{
              flexShrink: 0,
              marginTop: 'auto',
              padding: isMobile ? '2px' : '4px',
              borderTop: '1px solid rgba(0,255,255,0.2)',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'stretch',
              background: 'rgba(0,0,0,0.3)',
            }}>
              <div style={{
                display: 'flex',
                gap: isMobile ? '15px' : '16px',
                padding: isMobile ? '4px' : '6px',
                background: 'linear-gradient(135deg, rgba(15,15,40,0.95), rgba(8,8,25,0.98))',
                border: '1.5px solid rgba(0,255,255,0.35)',
                borderRadius: isMobile ? '12px' : '14px',
                boxShadow: '0 4px 20px rgba(0,247,255,0.2), inset 0 1px 0 rgba(255,255,255,0.05)',
                backdropFilter: 'blur(10px)',
                WebkitBackdropFilter: 'blur(10px)',
                width: '100%',
                justifyContent: 'center',
                alignItems: 'stretch',
              }}>
                <ControlButton
                  icon="🚪"
                  label="Выйти"
                  onClick={onExit}
                  variant="danger"
                  isMobile={isMobile}
                  compact={true}
                />
                <ControlButton
                  icon={isMicMuted ? '🔇' : '🎤'}
                  label={isMicMuted ? 'Вкл' : 'Выкл'}
                  onClick={onToggleMic}
                  variant={isMicMuted ? 'danger' : 'success'}
                  isActive={!isMicMuted}
                  isMobile={isMobile}
                  compact={true}
                />
                <ControlButton
                  icon={isSearchingNext ? '⏹️' : '⏭️'}
                  label={isSearchingNext ? 'Отмена' : 'Дальше'}
                  onClick={handleNextRoomClick}
                  variant={isSearchingNext ? 'danger' : 'primary'}
                  isMobile={isMobile}
                  compact={true}
                />
              </div>
            </div>

            {isSearchingNext && (
              <div style={{
                position: 'absolute',
                top: 0, left: 0, right: 0, bottom: 0,
                background: 'rgba(0,0,0,0.75)',
                backdropFilter: 'blur(6px)',
                WebkitBackdropFilter: 'blur(6px)',
                zIndex: 100,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '16px',
                pointerEvents: 'auto',
              }}>
                <div style={{
                  width: '64px',
                  height: '64px',
                  border: '4px solid rgba(0,255,255,0.15)',
                  borderTopColor: '#00ffff',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite',
                }} />

                <div style={{
                  fontSize: isMobile ? '1.1rem' : '1.4rem',
                  fontWeight: '700',
                  background: 'linear-gradient(135deg, #00ffff, #ff00ff)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  letterSpacing: '0.05em',
                  textAlign: 'center',
                }}>
                  ⏳ Поиск новой комнаты...
                </div>

                <div style={{
                  fontSize: isMobile ? '0.75rem' : '0.9rem',
                  color: '#a0e7ff',
                  textAlign: 'center',
                  maxWidth: '400px',
                  lineHeight: 1.5,
                  padding: '0 20px',
                }}>
                  Подбираем новых собеседников.<br/>
                  Подождите несколько секунд.
                </div>

                <button
                  onClick={onCancelSearch}
                  style={{
                    marginTop: '12px',
                    padding: '10px 24px',
                    background: 'linear-gradient(135deg, #ff4757, #ff3838)',
                    border: 'none',
                    borderRadius: '12px',
                    color: 'white',
                    fontWeight: '700',
                    fontSize: isMobile ? '13px' : '14px',
                    cursor: 'pointer',
                    boxShadow: '0 4px 20px rgba(255,71,87,0.4)',
                  }}
                >
                  ✕ Отменить поиск
                </button>
              </div>
            )}

          </div>

          <div style={{ flex: 1, display: 'grid', gridTemplateRows: 'repeat(3, 1fr)', gap: isMobile ? '3px' : '6px', minWidth: 0, minHeight: 0 }}>
            {right.map((sp, i) => {
              const isThisMaximized = maximizedCameraId === sp.id;
              const isHidden = isMaximized && maximizedCameraId === sp.id;
              return (
                <MemoizedCameraSlot
                  key={sp.id || `right-${i}`}
                  sp={sp}
                  isThisMaximized={isThisMaximized}
                  isHidden={isHidden}
                  volume={volumes[sp.id] || 0.8}
                  handleCameraClick={handleCameraClick}
                  handleVolumeChange={handleVolumeChange}
                  handleReport={handleReport}
                  isMobile={isMobile}
                  userId={userId}
                />
              );
            })}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .maximized-camera .camera-collapse-btn {
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.25s ease;
        }
        .maximized-camera:hover .camera-collapse-btn {
          opacity: 1;
          pointer-events: auto;
        }

        @media (hover: none) and (pointer: coarse) {
          .maximized-camera .camera-collapse-btn {
            opacity: 0;
            pointer-events: none;
          }
          .maximized-camera.show-collapse-btn .camera-collapse-btn {
            opacity: 1;
            pointer-events: auto;
          }
        }
      `}</style>

    {/* 🆕 ЧАТ РЕНДЕРИТСЯ ЧЕРЕЗ ПОРТАЛ В СЛОТ #video6-chat-slot */}
      {document.getElementById('video6-chat-slot') && createPortal(
        <Video6Chat
          messages={messages}
          messagesEndRef={messagesEndRef}
          onSendMessage={handleSendWithSpamCheck}
          nickname={nickname}
          isMobile={isMobile}
          isRateLimited={isRateLimited}
          cooldownSeconds={cooldownSeconds}
          isMaximized={isMaximized}
        />,
        document.getElementById('video6-chat-slot')
      )}

    </Portal>
  );
}, (prev, next) => {
  if (prev.isActive !== next.isActive) return false;
  if (prev.participants !== next.participants) return false;
  if (prev.messages !== next.messages) return false;
  if (prev.isSearchingNext !== next.isSearchingNext) return false;
  if (prev.isMobile !== next.isMobile) return false;
  if (prev.userId !== next.userId) return false;
  if (prev.isMicMuted !== next.isMicMuted) return false;
  if (prev.roomName !== next.roomName) return false;
  return true;
});

// ============= FLOATING ROCKET =============
const FloatingRocket = React.memo(() => {
  const containerRef = useRef(null);
  const animRef = useRef(null);
  const objectsRef = useRef([]);
  const dimRef = useRef({ width: window.innerWidth, height: window.innerHeight });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const objectTypes = [
      { type: 'rocket', symbol: '🚀', minSize: 18, maxSize: 28, opacityMin: 0.4, opacityMax: 0.8 },
      { type: 'planet', symbol: '🪐', minSize: 25, maxSize: 45, opacityMin: 0.3, opacityMax: 0.6 },
      { type: 'planet', symbol: '🌍', minSize: 22, maxSize: 40, opacityMin: 0.3, opacityMax: 0.6 },
      { type: 'planet', symbol: '⭐', minSize: 15, maxSize: 25, opacityMin: 0.5, opacityMax: 0.8 },
      { type: 'planet', symbol: '🌙', minSize: 18, maxSize: 30, opacityMin: 0.4, opacityMax: 0.7 }
    ];

    const COUNT = 15;
    const objects = [];
    const fragment = document.createDocumentFragment();

    for (let i = 0; i < COUNT; i++) {
      const td = objectTypes[Math.floor(Math.random() * objectTypes.length)];
      const el = document.createElement('div');
      el.textContent = td.symbol;
      el.style.position = 'fixed';
      el.style.pointerEvents = 'none';
      el.style.zIndex = '0';
      el.style.willChange = 'transform';
      el.style.fontSize = `${td.minSize + Math.random() * (td.maxSize - td.minSize)}px`;
      el.style.opacity = String(td.opacityMin + Math.random() * (td.opacityMax - td.opacityMin));
      fragment.appendChild(el);

      objects.push({
        el,
        type: td.type,
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 0.6,
        vy: (Math.random() - 0.5) * 0.4,
        rotation: Math.random() * 360,
        spinSpeed: td.type === 'planet' ? (Math.random() - 0.5) * 0.5 : 0,
      });
    }

    container.appendChild(fragment);
    objectsRef.current = objects;

    const handleResize = () => { dimRef.current = { width: window.innerWidth, height: window.innerHeight }; };
    window.addEventListener('resize', handleResize, { passive: true });

    let lastTime = performance.now();
    const TARGET_FPS = 30;
    const FRAME_INTERVAL = 1000 / TARGET_FPS;

    const animate = (time) => {
      animRef.current = requestAnimationFrame(animate);
      const delta = time - lastTime;
      if (delta < FRAME_INTERVAL) return;
      lastTime = time - (delta % FRAME_INTERVAL);

      const w = dimRef.current.width;
      const h = dimRef.current.height;

      for (let i = 0; i < objects.length; i++) {
        const obj = objects[i];
        let x = obj.x + obj.vx;
        let y = obj.y + obj.vy;
        let vx = obj.vx;
        let vy = obj.vy;

        if (x < 0) { x = 0; vx = Math.abs(vx); }
        if (x > w) { x = w; vx = -Math.abs(vx); }
        if (y < 0) { y = 0; vy = Math.abs(vy); }
        if (y > h) { y = h; vy = -Math.abs(vy); }

        obj.x = x; obj.y = y; obj.vx = vx; obj.vy = vy;

        if (obj.type === 'planet') obj.rotation += obj.spinSpeed || 0.3;

        obj.el.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%) rotate(${obj.rotation}deg)`;
      }
    };

    animRef.current = requestAnimationFrame(animate);

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
      window.removeEventListener('resize', handleResize);
      container.innerHTML = '';
      objectsRef.current = [];
    };
  }, []);

  return <div ref={containerRef} style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0, overflow: 'hidden' }} />;
});

// ============= HOME PAGE =============
const HomePage = React.memo(({ onBack, isMobile }) => {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let abortController = null;
    let interval = null;

    const fetchCount = async () => {
      if (abortController) abortController.abort();
      abortController = new AbortController();
      try {
        const r = await fetch(`${CONFIG.BACKEND_URL}/api/stats`, { signal: abortController.signal });
        if (r.ok) { const d = await r.json(); setCount(d.online_users || 0); }
      } catch(e) {}
    };
    fetchCount();
    interval = setInterval(fetchCount, 15000);
    return () => {
      if (interval) clearInterval(interval);
      if (abortController) abortController.abort();
    };
  }, []);

  const cosmicContainer = {
    background: 'linear-gradient(135deg, rgba(10,15,45,0.85), rgba(5,8,25,0.92))',
    backdropFilter: 'blur(14px)',
    WebkitBackdropFilter: 'blur(14px)',
    border: '1px solid rgba(0,247,255,0.35)',
    boxShadow: '0 8px 32px rgba(0,247,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05)',
    borderRadius: '16px',
    position: 'relative',
    overflow: 'hidden',
  };

  return (
    <div className="page-content cosmic-page" style={{ padding: isMobile ? '8px' : '0' }}>

      <div style={{ ...cosmicContainer, padding: isMobile ? '12px' : '20px' }}>

        <div style={{ position: 'absolute', top: '-10%', left: '-15%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />
        <div style={{ position: 'absolute', bottom: '-10%', right: '-15%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(255,0,255,0.12) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />

        <div style={{ position: 'relative', zIndex: 1 }}>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', paddingBottom: '12px', borderBottom: '1px solid rgba(0,247,255,0.25)' }}>
            <button
              onClick={onBack}
              style={{ background: 'linear-gradient(145deg, rgba(30,42,74,0.9), rgba(15,23,42,0.95))', border: '1px solid rgba(0,247,255,0.4)', borderRadius: '18px', padding: '6px 14px', color: '#a0e7ff', cursor: 'pointer', transition: 'all 0.2s ease' }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'linear-gradient(145deg, rgba(40,52,84,0.9), rgba(25,33,52,0.95))'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'linear-gradient(145deg, rgba(30,42,74,0.9), rgba(15,23,42,0.95))'}
            >
              ← Назад
            </button>
            <h2 style={{ margin: 0, fontSize: isMobile ? '20px' : '28px', fontWeight: '800', background: 'linear-gradient(135deg, #00ffff, #a0a0ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>ГЛАВНАЯ</h2>
            <div style={{ width: '80px' }}></div>
          </div>

          <div style={{ ...cosmicContainer, padding: isMobile ? '24px 14px' : '48px 34px', textAlign: 'center', marginBottom: '16px' }}>
            <div style={{ position: 'absolute', top: '-50%', right: '-10%', width: '60%', height: '200%', background: 'radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 65%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{ fontSize: '64px', marginBottom: '16px', filter: 'drop-shadow(0 0 20px rgba(0,247,255,0.4))' }}>🚀</div>
              <h1 style={{ fontSize: isMobile ? '1.6rem' : '2.6rem', marginBottom: '10px', fontWeight: '800', background: 'linear-gradient(135deg, #00ffff, #ff00ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>COSMIC VOICE</h1>
              <p style={{ color: '#a0e7ff', margin: 0 }}>Космическая связь через пространство и время</p>
            </div>
          </div>

          <div style={{ ...cosmicContainer, padding: isMobile ? '20px 14px' : '32px 26px', marginBottom: '16px', textAlign: 'left' }}>
            <div style={{ position: 'absolute', top: '-30%', left: '-20%', width: '60%', height: '160%', background: 'radial-gradient(circle, rgba(123,47,247,0.15) 0%, transparent 65%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>

              <h3 style={{
                textAlign: 'center',
                marginBottom: '20px',
                fontSize: isMobile ? '16px' : '20px',
                fontWeight: '800',
                background: 'linear-gradient(135deg, #00ffff, #ff00ff)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>🌌 Что такое Cosmic Voice?</h3>

              <p style={{
                color: '#c2c8d6',
                fontSize: isMobile ? '13px' : '14.5px',
                lineHeight: 1.7,
                marginBottom: '18px',
                textAlign: 'center',
              }}>
                <b style={{ color: '#fff' }}>Cosmic Voice</b> — это <b style={{ color: '#00ffff' }}>мультимедийная платформа</b> для анонимного общения в реальном времени. Мы объединяем <b style={{ color: '#00ff88' }}>голосовые</b>, <b style={{ color: '#ff00ff' }}>видео</b> и <b style={{ color: '#feca57' }}>текстовые</b> форматы связи в одном сервисе — без регистрации, без лишних данных, без границ.
              </p>

              <p style={{
                color: '#c2c8d6',
                fontSize: isMobile ? '13px' : '14.5px',
                lineHeight: 1.7,
                marginBottom: '20px',
                textAlign: 'center',
              }}>
                Общайтесь один на один или в групповых комнатах, находите собеседников по интересам, полу, возрасту и стране. Всё, что нужно — ник и браузер.
              </p>

              <div style={{
                display: 'grid',
                gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
                gap: '12px',
                marginTop: '20px',
              }}>
                {[
                  { icon: '🎤', title: 'Голосовой чат', desc: 'Анонимные звонки 1 на 1 с кристально чистым звуком WebRTC', color: '#00ffff' },
                  { icon: '🎥', title: 'Видео 1 на 1', desc: 'Личное видеообщение с HD-качеством и мгновенным поиском', color: '#00d2ff' },
                  { icon: '👥', title: 'Групповое видео (6)', desc: 'Комнаты до 6 участников с общим чатом и 6 камерами', color: '#00ff88' },
                  { icon: '💬', title: 'Текстовый чат', desc: 'Обмен сообщениями и эмодзи прямо во время общения', color: '#ff00ff' },
                  { icon: '🔍', title: 'Умный поиск', desc: 'Фильтры по теме, полу, возрасту и стране собеседника', color: '#feca57' },
                  { icon: '🛡️', title: 'Модерация 24/7', desc: 'Жалобы, предупреждения, баны и защита от спама', color: '#ff6b6b' },
                  { icon: '🔒', title: 'Полная анонимность', desc: 'Без регистрации. Ник и ID хранятся только в браузере', color: '#a0e7ff' },
                  { icon: '🚀', title: 'Космический стиль', desc: 'Тёмная тема, живые эффекты, плавные анимации', color: '#a78bfa' },
                ].map((f, i) => (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '12px',
                      padding: isMobile ? '10px 12px' : '14px 16px',
                      background: 'rgba(0,0,0,0.35)',
                      borderRadius: '12px',
                      border: `1px solid ${f.color}30`,
                      transition: 'all 0.2s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(0,0,0,0.55)';
                      e.currentTarget.style.borderColor = `${f.color}80`;
                      e.currentTarget.style.transform = 'translateY(-2px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'rgba(0,0,0,0.35)';
                      e.currentTarget.style.borderColor = `${f.color}30`;
                      e.currentTarget.style.transform = 'translateY(0)';
                    }}
                  >
                    <div style={{
                      fontSize: isMobile ? '24px' : '28px',
                      flexShrink: 0,
                      filter: `drop-shadow(0 0 8px ${f.color}80)`,
                    }}>{f.icon}</div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        color: f.color,
                        fontWeight: '700',
                        fontSize: isMobile ? '13px' : '14px',
                        marginBottom: '3px',
                      }}>{f.title}</div>
                      <div style={{
                        color: '#8a93a8',
                        fontSize: isMobile ? '11.5px' : '12.5px',
                        lineHeight: 1.45,
                      }}>{f.desc}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div style={{
                marginTop: '22px',
                padding: isMobile ? '14px' : '18px',
                background: 'linear-gradient(135deg, rgba(0,40,80,0.4), rgba(0,20,50,0.5))',
                borderRadius: '14px',
                border: '1px solid rgba(0,247,255,0.25)',
                textAlign: 'center',
              }}>
                <div style={{
                  color: '#00ffff',
                  fontWeight: '700',
                  fontSize: isMobile ? '13px' : '14.5px',
                  marginBottom: '8px',
                }}>✨ Почему выбирают Cosmic Voice</div>
                <div style={{
                  color: '#a0e7ff',
                  fontSize: isMobile ? '12px' : '13px',
                  lineHeight: 1.6,
                }}>
                  Мгновенный поиск за 3–10 секунд · Без рекламы в голосовом режиме · Работает на телефоне и ПК · Доступно без ограничений
                </div>
              </div>

            </div>
          </div>

          <div style={{ ...cosmicContainer, padding: isMobile ? '16px' : '20px' }}>
            <div style={{ position: 'absolute', bottom: '-50%', left: '-20%', width: '60%', height: '200%', background: 'radial-gradient(circle, rgba(255,0,255,0.1) 0%, transparent 65%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <h3 style={{ color: '#fff', textAlign: 'center', marginBottom: '20px', fontSize: isMobile ? '16px' : '18px', fontWeight: '700' }}>📊 Статистика</h3>
              <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : 'repeat(3, 1fr)', gap: '15px' }}>
                {[
                  { v: count.toLocaleString('ru-RU'), l: 'Онлайн', c: '#00ffff' },
                  { v: '24/7', l: 'Работаем', c: '#ff00ff' },
                  { v: '99%', l: 'Успешных Соединений', c: '#ffff00' }
                ].map((s, i) => (
                  <div
                    key={i}
                    style={{ background: 'rgba(0,0,0,0.4)', borderRadius: '14px', padding: '20px', textAlign: 'center', border: `1px solid ${s.c}40`, transition: 'all 0.2s ease' }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(0,0,0,0.6)'; e.currentTarget.style.borderColor = s.c; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(0,0,0,0.4)'; e.currentTarget.style.borderColor = `${s.c}40`; }}
                  >
                    <div style={{ fontSize: '2rem', fontWeight: '800', color: s.c, textShadow: `0 0 15px ${s.c}` }}>{s.v}</div>
                    <div style={{ color: '#ccc', marginTop: '4px' }}>{s.l}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div style={{
            textAlign: 'center',
            padding: isMobile ? '12px 8px 4px' : '16px 8px 6px',
            marginTop: '4px',
            borderTop: '1px solid rgba(0,247,255,0.15)',
            fontSize: isMobile ? '10px' : '11px',
            color: '#666',
            letterSpacing: '0.04em',
          }}>
            Cosmic-Voice. All rights reserved 2026-2027
          </div>

        </div>
      </div>
    </div>
  );
});

// ============= CONTACTS PAGE =============
const ContactsPage = React.memo(({ onBack, isMobile }) => {
  const cosmicContainer = {
    background: 'linear-gradient(135deg, rgba(10,15,45,0.85), rgba(5,8,25,0.92))',
    backdropFilter: 'blur(14px)',
    WebkitBackdropFilter: 'blur(14px)',
    border: '1px solid rgba(0,247,255,0.35)',
    boxShadow: '0 8px 32px rgba(0,247,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05)',
    borderRadius: '16px',
    position: 'relative',
    overflow: 'hidden',
  };

  const contacts = [
    { icon: '💬', title: 'Поддержка', email: 'support@cosmic-voice.ru', desc: 'Общие вопросы и помощь', color: '#00ffff' },
    { icon: '🛡️', title: 'Модерация', email: 'moderation@cosmic-voice.ru', desc: 'Жалобы и блокировки', color: '#ff6b6b' },
    { icon: '💳', title: 'Финансы', email: 'billing@cosmic-voice.ru', desc: 'Оплата и премиум', color: '#1dd1a1' },
    { icon: '⚖️', title: 'Юридические', email: 'legal@cosmic-voice.ru', desc: 'Правовые вопросы', color: '#feca57' },
  ];

  return (
    <div className="page-content cosmic-page" style={{ padding: isMobile ? '8px' : '0' }}>

      <div style={{ ...cosmicContainer, padding: isMobile ? '12px' : '20px' }}>

        <div style={{ position: 'absolute', top: '-10%', left: '-15%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(255,0,255,0.15) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />
        <div style={{ position: 'absolute', bottom: '-10%', right: '-15%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(0,255,255,0.12) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />

        <div style={{ position: 'relative', zIndex: 1 }}>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', paddingBottom: '12px', borderBottom: '1px solid rgba(255,0,255,0.25)' }}>
            <button
              onClick={onBack}
              style={{ background: 'linear-gradient(145deg, rgba(30,42,74,0.9), rgba(15,23,42,0.95))', border: '1px solid rgba(255,0,255,0.4)', borderRadius: '18px', padding: '6px 14px', color: '#ffa0ff', cursor: 'pointer', transition: 'all 0.2s ease' }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'linear-gradient(145deg, rgba(40,52,84,0.9), rgba(25,33,52,0.95))'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'linear-gradient(145deg, rgba(30,42,74,0.9), rgba(15,23,42,0.95))'}
            >
              ← Назад
            </button>
            <h2 style={{ margin: 0, fontSize: isMobile ? '20px' : '28px', fontWeight: '800', background: 'linear-gradient(135deg, #ff00ff, #ffa0ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>КОНТАКТЫ</h2>
            <div style={{ width: '80px' }}></div>
          </div>

          <div style={{ ...cosmicContainer, padding: isMobile ? '20px 14px' : '25px', textAlign: 'center', marginBottom: '16px' }}>
            <div style={{ position: 'absolute', top: '-30%', left: '-20%', width: '60%', height: '160%', background: 'radial-gradient(circle, rgba(255,0,255,0.15) 0%, transparent 65%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <h3 style={{ background: 'linear-gradient(135deg, #ff00ff, #00ffff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '20px', fontSize: isMobile ? '16px' : '18px', fontWeight: '700' }}>🌐 Мы в соцсетях</h3>
              <div style={{ display: 'flex', justifyContent: 'center', gap: isMobile ? '20px' : '30px' }}>
                <a href="#" style={{ color: '#0088cc', textDecoration: 'none', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', transition: 'transform 0.25s ease, filter 0.25s ease' }}
                  onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.filter = 'brightness(1.3)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.filter = 'brightness(1)'; }}>
                  <FaTelegram size={isMobile ? 32 : 36} />
                  <span style={{ fontSize: isMobile ? '11px' : '13px' }}>Telegram</span>
                </a>
                <a href="#" style={{ color: '#4C75A3', textDecoration: 'none', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', transition: 'transform 0.25s ease, filter 0.25s ease' }}
                  onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.filter = 'brightness(1.3)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.filter = 'brightness(1)'; }}>
                  <FaVk size={isMobile ? 32 : 36} />
                  <span style={{ fontSize: isMobile ? '11px' : '13px' }}>ВКонтакте</span>
                </a>
                <a href="#" style={{ color: '#fff', textDecoration: 'none', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', transition: 'transform 0.25s ease, filter 0.25s ease' }}
                  onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.filter = 'brightness(1.3)'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.filter = 'brightness(1)'; }}>
                  <FaTiktok size={isMobile ? 32 : 36} />
                  <span style={{ fontSize: isMobile ? '11px' : '13px' }}>TikTok</span>
                </a>
              </div>
            </div>
          </div>

          <h3 style={{ background: 'linear-gradient(135deg, #00ffff, #ff00ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', textAlign: 'center', marginBottom: '16px', fontSize: isMobile ? '16px' : '18px', fontWeight: '700' }}>📬 Поддержка</h3>

          {contacts.map((c, i) => (
            <div
              key={i}
              style={{ ...cosmicContainer, padding: isMobile ? '14px' : '16px', marginBottom: '12px', transition: 'all 0.2s ease' }}
              onMouseEnter={(e) => { e.currentTarget.style.boxShadow = `0 12px 40px ${c.color}25, inset 0 1px 0 rgba(255,255,255,0.08)`; e.currentTarget.style.borderColor = `${c.color}60`; }}
              onMouseLeave={(e) => { e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,247,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05)'; e.currentTarget.style.borderColor = 'rgba(0,247,255,0.35)'; }}
            >
              <div style={{ position: 'absolute', top: '-50%', right: '-20%', width: '50%', height: '200%', background: `radial-gradient(circle, ${c.color}15 0%, transparent 65%)`, pointerEvents: 'none' }} />
              <div style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                <div style={{ fontSize: isMobile ? '28px' : '34px', filter: `drop-shadow(0 0 10px ${c.color}80)`, flexShrink: 0 }}>{c.icon}</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <h4 style={{ color: '#fff', margin: '0 0 4px 0', fontSize: isMobile ? '14px' : '15px', fontWeight: '700' }}>{c.title}</h4>
                  <div style={{ color: '#8a93a8', fontSize: isMobile ? '11px' : '12px', marginBottom: '6px' }}>{c.desc}</div>
                  <CopyEmail
                    email={c.email}
                    color={c.color}
                    fontSize={isMobile ? '12px' : '13px'}
                  />
                </div>
              </div>
            </div>
          ))}

          <div style={{
            textAlign: 'center',
            padding: isMobile ? '14px 8px 4px' : '18px 8px 6px',
            marginTop: '16px',
            borderTop: '1px solid rgba(0,247,255,0.15)',
            fontSize: isMobile ? '10px' : '11px',
            color: '#666',
            letterSpacing: '0.04em',
          }}>
            Cosmic-Voice. All rights reserved 2026-2027
          </div>

        </div>
      </div>
    </div>
  );
});

// ============= RULES PAGE =============
const RulesPage = React.memo(({ onBack, isMobile }) => {
  const cosmicContainer = {
    background: 'linear-gradient(135deg, rgba(10,15,45,0.85), rgba(5,8,25,0.92))',
    backdropFilter: 'blur(14px)',
    WebkitBackdropFilter: 'blur(14px)',
    border: '1px solid rgba(0,247,255,0.35)',
    boxShadow: '0 8px 32px rgba(0,247,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05)',
    borderRadius: '16px',
    position: 'relative',
    overflow: 'hidden',
  };

  const forbiddenRules = [
    { icon: '🚫', text: 'Оскорбления и угрозы в адрес других пользователей' },
    { icon: '📢', text: 'Спам, реклама и самопиар' },
    { icon: '🔞', text: 'Непристойный контент и материалы для взрослых' },
    { icon: '⚠️', text: 'Пропаганда насилия, экстремизма, наркотиков' },
    { icon: '💸', text: 'Мошенничество и обман пользователей' },
    { icon: '🎭', text: 'Выдача себя за другого человека' },
    { icon: '📹', text: 'Запись и распространение разговоров без согласия' },
    { icon: '👶', text: 'Любые действия с несовершеннолетними' },
  ];

  const recommendedRules = [
    { icon: '🤝', text: 'Быть вежливым и уважительным' },
    { icon: '🎯', text: 'Соблюдать выбранную тему общения' },
    { icon: '🚨', text: 'Жаловаться на нарушения через кнопку "Жалоба"' },
    { icon: '🧹', text: 'Не мусорить в чате' },
    { icon: '🎧', text: 'Использовать наушники для лучшего качества' },
    { icon: '🔒', text: 'Не делиться личными данными' },
  ];

  return (
    <div className="page-content cosmic-page" style={{ padding: isMobile ? '8px' : '0' }}>

      <div style={{ ...cosmicContainer, padding: isMobile ? '12px' : '20px' }}>

        <div style={{ position: 'absolute', top: '-10%', left: '-15%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(255,71,87,0.15) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />
        <div style={{ position: 'absolute', bottom: '-10%', right: '-15%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(0,255,136,0.12) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />

        <div style={{ position: 'relative', zIndex: 1 }}>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', paddingBottom: '12px', borderBottom: '1px solid rgba(255,200,0,0.25)' }}>
            <button
              onClick={onBack}
              style={{ background: 'linear-gradient(145deg, rgba(30,42,74,0.9), rgba(15,23,42,0.95))', border: '1px solid rgba(255,200,0,0.4)', borderRadius: '18px', padding: '6px 14px', color: '#ffffa0', cursor: 'pointer', transition: 'all 0.2s ease' }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'linear-gradient(145deg, rgba(40,52,84,0.9), rgba(25,33,52,0.95))'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'linear-gradient(145deg, rgba(30,42,74,0.9), rgba(15,23,42,0.95))'}
            >
              ← Назад
            </button>
            <h2 style={{ margin: 0, fontSize: isMobile ? '20px' : '28px', fontWeight: '800', background: 'linear-gradient(135deg, #ffff00, #ffaa00)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>ПРАВИЛА</h2>
            <div style={{ width: '80px' }}></div>
          </div>

          <div style={{ ...cosmicContainer, padding: isMobile ? '16px' : '20px', marginBottom: '16px', borderColor: 'rgba(255,71,87,0.5)', boxShadow: '0 8px 32px rgba(255,71,87,0.2), inset 0 1px 0 rgba(255,255,255,0.05)' }}>
            <div style={{ position: 'absolute', top: '-30%', left: '-20%', width: '60%', height: '160%', background: 'radial-gradient(circle, rgba(255,71,87,0.15) 0%, transparent 65%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <h3 style={{ color: '#ff6b6b', marginBottom: '16px', fontSize: isMobile ? '16px' : '18px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '10px', borderBottom: '1px solid rgba(255,71,87,0.3)' }}>🚫 ЗАПРЕЩЕНО</h3>
              <ol style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {forbiddenRules.map((r, i) => (
                  <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '10px 12px', background: 'rgba(255,71,87,0.08)', borderRadius: '10px', border: '1px solid rgba(255,71,87,0.2)', color: '#e2e8f0', fontSize: isMobile ? '12.5px' : '13.5px', lineHeight: '1.5', transition: 'all 0.2s ease' }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,71,87,0.15)'; e.currentTarget.style.borderColor = 'rgba(255,71,87,0.4)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,71,87,0.08)'; e.currentTarget.style.borderColor = 'rgba(255,71,87,0.2)'; }}>
                    <span style={{ fontSize: isMobile ? '16px' : '18px', flexShrink: 0 }}>{r.icon}</span>
                    <span style={{ flex: 1 }}><strong style={{ color: '#ff9f9f', marginRight: '6px' }}>{i + 1}.</strong>{r.text}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>

          <div style={{ ...cosmicContainer, padding: isMobile ? '16px' : '20px', marginBottom: '16px', borderColor: 'rgba(0,255,136,0.5)', boxShadow: '0 8px 32px rgba(0,255,136,0.15), inset 0 1px 0 rgba(255,255,255,0.05)' }}>
            <div style={{ position: 'absolute', top: '-30%', right: '-20%', width: '60%', height: '160%', background: 'radial-gradient(circle, rgba(0,255,136,0.15) 0%, transparent 65%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <h3 style={{ color: '#6bff6b', marginBottom: '16px', fontSize: isMobile ? '16px' : '18px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px', paddingBottom: '10px', borderBottom: '1px solid rgba(0,255,136,0.3)' }}>✅ РЕКОМЕНДУЕТСЯ</h3>
              <ol style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {recommendedRules.map((r, i) => (
                  <li key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '10px 12px', background: 'rgba(0,255,136,0.08)', borderRadius: '10px', border: '1px solid rgba(0,255,136,0.2)', color: '#e2e8f0', fontSize: isMobile ? '12.5px' : '13.5px', lineHeight: '1.5', transition: 'all 0.2s ease' }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(0,255,136,0.15)'; e.currentTarget.style.borderColor = 'rgba(0,255,136,0.4)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(0,255,136,0.08)'; e.currentTarget.style.borderColor = 'rgba(0,255,136,0.2)'; }}>
                    <span style={{ fontSize: isMobile ? '16px' : '18px', flexShrink: 0 }}>{r.icon}</span>
                    <span style={{ flex: 1 }}><strong style={{ color: '#9fff9f', marginRight: '6px' }}>{i + 1}.</strong>{r.text}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>

          <div style={{ ...cosmicContainer, padding: isMobile ? '14px' : '16px', textAlign: 'center', borderColor: 'rgba(255,200,0,0.4)', boxShadow: '0 8px 24px rgba(255,200,0,0.12), inset 0 1px 0 rgba(255,255,255,0.05)' }}>
            <div style={{ position: 'absolute', top: '-50%', left: '-20%', width: '60%', height: '200%', background: 'radial-gradient(circle, rgba(255,200,0,0.1) 0%, transparent 65%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1, color: '#feca57', fontSize: isMobile ? '12px' : '13px', lineHeight: '1.6' }}>
              <div style={{ fontSize: '24px', marginBottom: '8px' }}>⚖️</div>
              <div>Незнание правил не освобождает от ответственности.</div>
              <div style={{ opacity: 0.8, marginTop: '6px' }}>Модераторы следят за порядком 24/7.</div>
            </div>
          </div>

          <div style={{
            textAlign: 'center',
            padding: isMobile ? '14px 8px 4px' : '18px 8px 6px',
            marginTop: '16px',
            borderTop: '1px solid rgba(255,200,0,0.15)',
            fontSize: isMobile ? '10px' : '11px',
            color: '#666',
            letterSpacing: '0.04em',
          }}>
            Cosmic-Voice. All rights reserved 2026-2027
          </div>

        </div>
      </div>
    </div>
  );
});

// ============= FAQ PAGE =============
const FAQPage = React.memo(({ onBack, isMobile }) => {
  const [openIndex, setOpenIndex] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const faqData = useMemo(() => [
    { category: '🚀 Начало работы', items: [
      { q: 'Как начать пользоваться Cosmic Voice?', a: 'Откройте сайт, и нажмите "НАЧАТЬ ПОИСК". Система найдёт собеседника за несколько секунд. Никакой регистрации не требуется.' },
      { q: 'Нужна ли регистрация?', a: 'Нет. Cosmic-Voice — анонимный сервис. Вы просто придумываете ник и общаетесь. Все данные (ник, ID) хранятся в localStorage вашего браузера и не передаются третьим лицам.' },
      { q: 'Какие браузеры поддерживаются?', a: 'Chrome, Edge, Firefox, Safari, Opera — последние версии. Для видео-режимов настоятельно рекомендуем Chrome или Edge — у них лучшая поддержка WebRTC и LiveKit.' },
      { q: 'Можно ли пользоваться с телефона?', a: 'Да. Сайт полностью адаптирован под мобильные устройства. Установите как PWA (Добавить на главный экран) для наилучшего опыта — как нативное приложение.' },
    ]},
    { category: '💬 Голосовой чат', items: [
      { q: 'Как искать собеседника?', a: 'На главной выберите тему (или оставьте "Все темы"), укажите свой пол и предпочтения по полу собеседника. Нажмите "НАЧАТЬ ПОИСК". Обычно поиск занимает 3-10 секунд.' },
      { q: 'Как остановить поиск?', a: 'Нажмите "ОСТАНОВИТЬ ПОИСК" — вы вернётесь в исходное состояние. Если уже в разговоре — нажмите "Стоп" или "Дальше".' },
      { q: 'Как работает кнопка "Дальше"?', a: 'Мгновенно отключает текущего собеседника и ищет нового. Используйте, если разговор не интересен. Собеседник получит уведомление об отключении.' },
      { q: 'Почему меня не слышат?', a: 'Проверьте: 1) разрешён ли доступ к микрофону в браузере, 2) не заглушён ли микрофон модератором, 3) не нажата ли кнопка "Выкл" на микрофоне, 4) работает ли сам микрофон в других приложениях.' },
      { q: 'Как выбрать тему общения?', a: 'Перед началом поиска откройте выпадающий список "Тема" — там 6 категорий: Все темы, Знакомства, Дружба, Развлечения, Советы, Игры. Собеседник подбирается по совпадению интересов.' },
    ]},
    { category: '🎥 Видео-режимы', items: [
      { q: 'Как работает Видео 1 на 1?', a: 'Открывается Видео-комната, система ищет второго участника. Вы видите и слышите друг друга в реальном времени. Каждый может пожаловаться, переключиться ("Дальше") или выйти.' },
      { q: 'Как работает Групповое видео (6)?', a: 'Комната до 6 участников. Каждая камера — отдельный слот (3 слева, 3 справа, чат по центру). Микрофон по умолчанию ВЫКЛЮЧЕН — включите кнопкой "Вкл".' },
      { q: 'Почему видео тормозит?', a: 'На слабом интернете LiveKit автоматически снижает качество (adaptiveStream). Закройте другие вкладки/приложения. Для 6 камер рекомендуем Wi-Fi и браузер Chrome.' },
      { q: 'Можно ли переключаться между комнатами в видео?', a: 'Да. Кнопка "Дальше" в видео-режиме отключает вас от текущей комнаты и ищет новую с другими участниками.' },
      { q: 'Как включить микрофон в видео?', a: 'В видео-режиме нажмите кнопку "🔇 Выкл" внизу экрана. Она переключится в "🎤 Вкл" — микрофон включён. По умолчанию он выключен для приватности.' },
    ]},
    { category: '🚨 Жалобы и модерация', items: [
      { q: 'Как пожаловаться на пользователя?', a: 'Наведите курсор на видео собеседника → появится кнопка 🚨 "Жалоба" в правом верхнем углу → выберите причину → опишите ситуацию → отправьте.' },
      { q: 'Что происходит после жалобы?', a: 'Жалоба попадает в очередь модерации. Модератор проверит и может: вынести предупреждение, заглушить, забанить или отклонить жалобу. Уведомление о решении придёт вам.' },
      { q: 'Сколько жалоб нужно для бана?', a: '3 жалобы от разных пользователей — автоматическая блокировка на 24 часа. Модераторы могут продлить срок или снять блокировку вручную.' },
      { q: 'Как оспорить блокировку?', a: 'Напишите на support@cosmic-voice.ru с вашим ником и описанием ситуации. Ответ в течение 24 часов. Приложите скриншоты, если есть.' },
      { q: 'Что такое расширенный бан?', a: 'Это блокировка по ID + IP + браузеру + fingerprint + подсети. Обходится только сменой устройства и IP. Применяется к злостным нарушителям.' },
    ]},
    { category: '🛠️ Техподдержка', items: [
      { q: 'Не работает камера — что делать?', a: 'Chrome: Настройки → Конфиденциальность → Камера → разрешите для сайта. Safari: Настройки → Сайты → Камера. После изменений перезагрузите страницу.' },
      { q: 'Постоянно теряется соединение', a: 'Проверьте интернет: ping < 100ms, скорость > 5 Мбит/с. Если используете VPN — попробуйте отключить (некоторые VPN блокируют WebRTC). Проверьте, не блокирует ли антивирус.' },
      { q: 'Как связаться с поддержкой?', a: '📧 support@cosmic-voice.ru — общие вопросы. 🛡️ moderation@cosmic-voice.ru — жалобы и блокировки. 💳 billing@cosmic-voice.ru — оплата и премиум. ⚖️ legal@cosmic-voice.ru — юридические вопросы.' },
      { q: 'Как удалить свой аккаунт?', a: 'Аккаунтов как таковых нет — всё хранится в браузере. Очистите данные сайта (Chrome: Настройки → Конфиденциальность → Данные сайтов → Удалить для cosmic-voice).' },
      { q: 'Можно ли использовать на нескольких устройствах?', a: 'Да. Один ник можно использовать где угодно. Но помните: на разных устройствах — разные "аккаунты" (разные ID).' },
    ]},
  ], []);

  const filteredData = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return faqData;
    return faqData
      .map(cat => ({ ...cat, items: cat.items.filter(item => item.q.toLowerCase().includes(q) || item.a.toLowerCase().includes(q)) }))
      .filter(cat => cat.items.length > 0);
  }, [faqData, searchQuery]);

  const totalQuestions = useMemo(() => faqData.reduce((sum, cat) => sum + cat.items.length, 0), [faqData]);

  const cosmicContainer = {
    background: 'linear-gradient(135deg, rgba(10,15,45,0.85), rgba(5,8,25,0.92))',
    backdropFilter: 'blur(14px)',
    WebkitBackdropFilter: 'blur(14px)',
    border: '1px solid rgba(0,247,255,0.35)',
    boxShadow: '0 8px 32px rgba(0,247,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05)',
    borderRadius: '16px',
    position: 'relative',
    overflow: 'hidden',
  };

  return (
    <div className="page-content cosmic-page" style={{ padding: isMobile ? '8px' : '0' }}>

      <div style={{ ...cosmicContainer, padding: isMobile ? '12px' : '20px' }}>

        <div style={{ position: 'absolute', top: '-10%', left: '-15%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />
        <div style={{ position: 'absolute', bottom: '-10%', right: '-15%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(255,0,255,0.12) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />

        <div style={{ position: 'relative', zIndex: 1 }}>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', paddingBottom: '12px', borderBottom: '1px solid rgba(0,247,255,0.25)' }}>
            <button
              onClick={onBack}
              style={{ background: 'linear-gradient(145deg, rgba(30,42,74,0.9), rgba(15,23,42,0.95))', border: '1px solid rgba(0,247,255,0.4)', borderRadius: '18px', padding: '6px 14px', color: '#a0e7ff', cursor: 'pointer', transition: 'all 0.2s ease' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'linear-gradient(145deg, rgba(40,52,84,0.9), rgba(25,33,52,0.95))'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'linear-gradient(145deg, rgba(30,42,74,0.9), rgba(15,23,42,0.95))'; }}
            >
              ← Назад
            </button>
            <h2 style={{ margin: 0, fontSize: isMobile ? '20px' : '28px', fontWeight: '800', background: 'linear-gradient(135deg, #00ffff, #aaffff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>FAQ</h2>
            <div style={{ width: '80px' }}></div>
          </div>

          <div style={{ position: 'relative', marginBottom: '20px' }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Поиск по вопросам..."
              style={{ width: '100%', padding: isMobile ? '12px 40px 12px 40px' : '14px 44px 14px 44px', background: 'rgba(0,0,0,0.4)', border: '1px solid rgba(0,247,255,0.3)', borderRadius: '12px', color: '#fff', fontSize: isMobile ? '13px' : '14px', outline: 'none', boxSizing: 'border-box', transition: 'border-color 0.2s ease' }}
              onFocus={(e) => e.target.style.borderColor = 'rgba(0,247,255,0.7)'}
              onBlur={(e) => e.target.style.borderColor = 'rgba(0,247,255,0.3)'}
            />
            <span style={{ position: 'absolute', left: isMobile ? '14px' : '16px', top: '50%', transform: 'translateY(-50%)', fontSize: isMobile ? '14px' : '16px', pointerEvents: 'none', opacity: 0.6 }}>🔍</span>
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', background: 'rgba(255,255,255,0.1)', border: 'none', borderRadius: '50%', width: '24px', height: '24px', color: '#fff', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'background 0.2s ease' }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.2)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
              >✕</button>
            )}
          </div>

          {searchQuery && (
            <div style={{ fontSize: '12px', color: '#8a93a8', marginBottom: '12px', textAlign: 'center' }}>
              Найдено: <span style={{ color: '#00ffff', fontWeight: '700' }}>{filteredData.reduce((sum, cat) => sum + cat.items.length, 0)}</span> из {totalQuestions}
            </div>
          )}

          {filteredData.length === 0 && searchQuery && (
            <div style={{ textAlign: 'center', padding: '30px 20px', color: '#666', background: 'rgba(0,0,0,0.3)', borderRadius: '12px', border: '1px dashed rgba(0,247,255,0.25)' }}>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>🔍</div>
              <div style={{ fontSize: '14px', marginBottom: '6px' }}>Ничего не найдено по запросу</div>
              <div style={{ fontSize: '13px', color: '#00ffff' }}>«{searchQuery}»</div>
            </div>
          )}

          {filteredData.map((cat, ci) => (
            <div key={ci} style={{ ...cosmicContainer, marginBottom: '16px', padding: isMobile ? '12px' : '16px' }}>
              <div style={{ position: 'absolute', top: '-30%', right: '-10%', width: '40%', height: '150%', background: 'radial-gradient(circle, rgba(0,255,255,0.08) 0%, transparent 65%)', pointerEvents: 'none' }} />
              <div style={{ position: 'relative', zIndex: 1 }}>
                <h3 style={{ fontSize: isMobile ? '14px' : '16px', color: '#a0e7ff', margin: '0 0 12px 0', paddingBottom: '8px', borderBottom: '1px solid rgba(0,247,255,0.2)', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '700' }}>
                  <span>{cat.category}</span>
                  <span style={{ fontSize: '10px', color: '#00ffff', background: 'rgba(0,247,255,0.12)', padding: '2px 8px', borderRadius: '10px', fontWeight: '600', border: '1px solid rgba(0,247,255,0.25)' }}>{cat.items.length}</span>
                </h3>

                {cat.items.map((f, i) => {
                  const key = `${ci}-${i}`;
                  const isOpen = openIndex === key;
                  return (
                    <div
                      key={i}
                      style={{
                        marginBottom: '6px',
                        background: isOpen ? 'linear-gradient(135deg, rgba(0,40,80,0.6), rgba(0,20,50,0.7))' : 'rgba(0,0,0,0.35)',
                        borderRadius: '10px',
                        border: isOpen ? '1px solid rgba(0,247,255,0.5)' : '1px solid rgba(0,247,255,0.15)',
                        overflow: 'hidden',
                        transition: 'all 0.2s ease',
                        boxShadow: isOpen ? '0 4px 16px rgba(0,247,255,0.15)' : 'none',
                      }}
                    >
                      <button
                        onClick={() => setOpenIndex(isOpen ? null : key)}
                        style={{ width: '100%', padding: isMobile ? '10px 12px' : '12px 14px', background: 'transparent', border: 'none', color: '#fff', fontSize: isMobile ? '12.5px' : '13.5px', fontWeight: '600', textAlign: 'left', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px', transition: 'background 0.15s ease' }}
                        onMouseEnter={(e) => { if (!isOpen) e.currentTarget.style.background = 'rgba(0,247,255,0.05)'; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                      >
                        <span style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', flex: 1 }}>
                          <FaQuestionCircle style={{ color: '#00ffff', marginTop: '2px', flexShrink: 0, fontSize: '12px' }} />
                          <span>{f.q}</span>
                        </span>
                        <span style={{ color: '#00ffff', transition: 'transform 0.25s ease', transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', flexShrink: 0, fontSize: '11px' }}>▼</span>
                      </button>
                      {isOpen && (
                        <div style={{ padding: isMobile ? '0 12px 12px 28px' : '0 14px 14px 32px', color: '#c2c8d6', fontSize: isMobile ? '12px' : '13px', lineHeight: '1.65', animation: 'fadeIn 0.25s ease', borderTop: '1px solid rgba(0,247,255,0.15)', paddingTop: '10px' }}>
                          {f.a}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          <div style={{ marginTop: '20px', padding: isMobile ? '14px' : '16px', background: 'linear-gradient(135deg, rgba(0,40,80,0.5), rgba(0,20,50,0.6))', borderRadius: '14px', border: '1px solid rgba(0,247,255,0.35)', boxShadow: '0 8px 24px rgba(0,247,255,0.12), inset 0 1px 0 rgba(255,255,255,0.05)', textAlign: 'center', position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: '-50%', left: '-20%', width: '60%', height: '200%', background: 'radial-gradient(circle, rgba(255,0,255,0.12) 0%, transparent 65%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{ color: '#a0e7ff', marginBottom: '10px', fontSize: isMobile ? '12px' : '13px', fontWeight: '600' }}>Не нашли ответ на свой вопрос?</div>
              <CopyEmail
                email="support@cosmic-voice.ru"
                color="#00ffff"
                fontSize={isMobile ? '13px' : '14px'}
                style={{
                  padding: '8px 18px',
                  background: 'rgba(0,247,255,0.12)',
                  borderRadius: '10px',
                  border: '1px solid rgba(0,247,255,0.35)',
                }}
              />
            </div>
          </div>

          <div style={{
            textAlign: 'center',
            padding: isMobile ? '14px 8px 4px' : '18px 8px 6px',
            marginTop: '16px',
            borderTop: '1px solid rgba(0,247,255,0.15)',
            fontSize: isMobile ? '10px' : '11px',
            color: '#666',
            letterSpacing: '0.04em',
          }}>
            Cosmic-Voice. All rights reserved 2026-2027
          </div>

        </div>
      </div>
    </div>
  );
});

// ============= AD PLAYER =============
const AdPlayer = React.memo(({ ad, type, onComplete, isMobile, isVideoMode }) => {
  const audioRef = useRef(null);
  const videoRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const [canSkip, setCanSkip] = useState(!isVideoMode);
  const [skipCountdown, setSkipCountdown] = useState(5);
  const [remainingSeconds, setRemainingSeconds] = useState(ad?.duration || 30);
  const isMountedRef = useRef(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    isMountedRef.current = true;
    return () => { isMountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (!isVideoMode) {
      setCanSkip(true);
      return;
    }
    setCanSkip(false);
    setSkipCountdown(5);
    const interval = setInterval(() => {
      setSkipCountdown(prev => {
        if (prev <= 1) {
          clearInterval(interval);
          setCanSkip(true);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [isVideoMode, ad?.id]);

  useEffect(() => {
    if (!ad?.file_data) return;
    const mediaEl = type === 'audio' ? audioRef.current : videoRef.current;
    if (!mediaEl) return;

    const handleTimeUpdate = () => {
      const dur = mediaEl.duration || ad.duration || 30;
      if (isMountedRef.current && dur > 0) {
        const p = (mediaEl.currentTime / dur) * 100;
        setProgress(p);
        setRemainingSeconds(Math.max(0, Math.ceil(dur - mediaEl.currentTime)));
      }
    };
    const handleEnded = () => { if (isMountedRef.current && onComplete) onComplete(); };
    const handleError = () => {
      if (isMountedRef.current) setError('Ошибка');
      setTimeout(() => { if (onComplete) onComplete(); }, 3000);
    };

    mediaEl.addEventListener('timeupdate', handleTimeUpdate);
    mediaEl.addEventListener('ended', handleEnded);
    mediaEl.addEventListener('error', handleError);
    mediaEl.src = ad.file_data;
    mediaEl.volume = 0.6;
    mediaEl.load();
    let cancelled = false;
    setTimeout(() => {
      if (cancelled || !isMountedRef.current) return;
      const p = mediaEl.play();
      if (p && typeof p.then === 'function') p.catch(e => {
        if (e.name === 'NotAllowedError') setError('Нажмите для звука');
      });
    }, 200);
    return () => {
      cancelled = true;
      mediaEl.removeEventListener('timeupdate', handleTimeUpdate);
      mediaEl.removeEventListener('ended', handleEnded);
      mediaEl.removeEventListener('error', handleError);
    };
  }, [ad?.id, ad?.file_data, type, onComplete]);

  const handleSkip = useCallback(() => {
    if (!canSkip) return;
    if (audioRef.current) try { audioRef.current.pause(); } catch(e){}
    if (videoRef.current) try { videoRef.current.pause(); } catch(e){}
    if (onComplete) onComplete();
  }, [onComplete, canSkip]);

  if (!ad) return null;

  return (
    <div style={{
      position: 'fixed', inset: 0,
      background: 'radial-gradient(ellipse at 50% 40%, rgba(20,10,50,0.92) 0%, rgba(0,0,0,0.97) 70%)',
      backdropFilter: 'blur(30px)',
      WebkitBackdropFilter: 'blur(30px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 999999,
      padding: isMobile ? '16px' : '30px',
    }}>
      <div style={{
        position: 'absolute', top: '10%', left: '10%',
        width: '40%', height: '40%',
        background: 'radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 65%)',
        pointerEvents: 'none',
        animation: 'adGlow 4s ease-in-out infinite alternate',
      }} />
      <div style={{
        position: 'absolute', bottom: '10%', right: '10%',
        width: '40%', height: '40%',
        background: 'radial-gradient(circle, rgba(255,0,255,0.12) 0%, transparent 65%)',
        pointerEvents: 'none',
        animation: 'adGlow 4s ease-in-out infinite alternate-reverse',
      }} />

      <div style={{
        position: 'relative',
        maxWidth: isMobile ? '100%' : '640px',
        width: '100%',
        background: 'linear-gradient(145deg, rgba(15,12,40,0.98), rgba(8,5,20,0.99))',
        borderRadius: '28px',
        border: '1.5px solid rgba(0,255,255,0.35)',
        boxShadow: '0 30px 80px rgba(0,0,0,0.8), 0 0 60px rgba(0,255,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05)',
        padding: isMobile ? '20px 18px' : '26px 26px 22px',
        animation: 'adCardIn 0.5s ease-out',
        overflow: 'hidden',
      }}>

        <div style={{
          position: 'absolute', top: '-50%', right: '-30%',
          width: '80%', height: '200%',
          background: 'radial-gradient(circle, rgba(0,255,255,0.08) 0%, transparent 65%)',
          pointerEvents: 'none',
          animation: 'adGlow 4s ease-in-out infinite alternate',
        }} />

        <div style={{
          position: 'relative', zIndex: 1,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '18px',
          gap: '12px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flex: 1, minWidth: 0 }}>
            <div style={{
              position: 'relative',
              width: isMobile ? '44px' : '54px',
              height: isMobile ? '44px' : '54px',
              borderRadius: '16px',
              background: 'linear-gradient(135deg, rgba(0,255,255,0.2), rgba(255,0,255,0.15))',
              border: '1.5px solid rgba(0,255,255,0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: isMobile ? '22px' : '28px',
              boxShadow: '0 0 20px rgba(0,255,255,0.3), inset 0 0 20px rgba(0,255,255,0.1)',
              flexShrink: 0,
              animation: 'adIconPulse 2s ease-in-out infinite',
            }}>
              {type === 'audio' ? '🎵' : '🎬'}
            </div>
            <div style={{ minWidth: 0, flex: 1 }}>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '2px 10px',
                background: 'linear-gradient(135deg, rgba(0,255,255,0.15), rgba(255,0,255,0.1))',
                border: '1px solid rgba(0,255,255,0.4)',
                borderRadius: '20px',
                marginBottom: '6px',
                fontSize: isMobile ? '9px' : '10px',
                fontWeight: '800',
                color: '#00ffff',
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
              }}>
                <span style={{
                  width: '6px', height: '6px',
                  borderRadius: '50%',
                  background: '#00ffff',
                  boxShadow: '0 0 8px #00ffff',
                  animation: 'adPulse 1.5s ease-in-out infinite',
                }} />
                Реклама
              </div>
              <div style={{
                color: '#fff',
                fontSize: isMobile ? '13px' : '15px',
                fontWeight: '700',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                lineHeight: 1.2,
              }}>
                {ad.name || 'Без названия'}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: isMobile ? '6px 10px' : '8px 14px',
              background: 'rgba(0,0,0,0.4)',
              border: '1.5px solid rgba(0,255,255,0.3)',
              borderRadius: '12px',
              fontSize: isMobile ? '13px' : '15px',
              fontWeight: '900',
              color: '#00ffff',
              textShadow: '0 0 10px rgba(0,255,255,0.6)',
              fontVariantNumeric: 'tabular-nums',
              minWidth: isMobile ? '50px' : '62px',
              justifyContent: 'center',
            }}>
              <span style={{ fontSize: isMobile ? '11px' : '13px' }}>⏱</span>
              <span>{remainingSeconds}</span>
            </div>

            <button
              onClick={handleSkip}
              disabled={!canSkip}
              style={{
                padding: isMobile ? '8px 12px' : '10px 18px',
                background: canSkip
                  ? 'linear-gradient(135deg, rgba(0,255,255,0.15), rgba(255,0,255,0.1))'
                  : 'rgba(60,60,80,0.3)',
                border: canSkip
                  ? '1.5px solid rgba(0,255,255,0.5)'
                  : '1.5px solid rgba(255,255,255,0.1)',
                borderRadius: '12px',
                color: canSkip ? '#00ffff' : '#666',
                fontSize: isMobile ? '11px' : '12px',
                fontWeight: '700',
                letterSpacing: '0.05em',
                cursor: canSkip ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'all 0.25s ease',
                boxShadow: canSkip ? '0 0 15px rgba(0,255,255,0.25)' : 'none',
                whiteSpace: 'nowrap',
              }}
              onMouseEnter={(e) => {
                if (canSkip) {
                  e.currentTarget.style.transform = 'translateY(-1px)';
                  e.currentTarget.style.boxShadow = '0 0 25px rgba(0,255,255,0.5)';
                  e.currentTarget.style.background = 'linear-gradient(135deg, rgba(0,255,255,0.25), rgba(255,0,255,0.15))';
                }
              }}
              onMouseLeave={(e) => {
                if (canSkip) {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 0 15px rgba(0,255,255,0.25)';
                  e.currentTarget.style.background = 'linear-gradient(135deg, rgba(0,255,255,0.15), rgba(255,0,255,0.1))';
                }
              }}
            >
              {canSkip ? (
                <>
                  <span>Пропустить</span>
                  <span style={{ fontSize: '14px' }}>✕</span>
                </>
              ) : (
                <span>⏳ {skipCountdown}с</span>
              )}
            </button>
          </div>
        </div>

        <div style={{ position: 'relative', zIndex: 1 }}>
          {type === 'audio' && (
            <div style={{
              position: 'relative',
              padding: isMobile ? '30px 20px' : '50px 30px',
              background: 'linear-gradient(135deg, rgba(0,255,255,0.08), rgba(255,0,255,0.06))',
              borderRadius: '20px',
              textAlign: 'center',
              border: '1px solid rgba(0,255,255,0.2)',
              overflow: 'hidden',
            }}>
              <div style={{
                position: 'absolute', top: '-50%', left: '-20%',
                width: '60%', height: '200%',
                background: 'radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 65%)',
                pointerEvents: 'none',
                animation: 'adGlow 3s ease-in-out infinite alternate',
              }} />
              <div style={{
                position: 'absolute', bottom: '-50%', right: '-20%',
                width: '60%', height: '200%',
                background: 'radial-gradient(circle, rgba(255,0,255,0.12) 0%, transparent 65%)',
                pointerEvents: 'none',
                animation: 'adGlow 3s ease-in-out infinite alternate-reverse',
              }} />

              <div style={{ position: 'relative', zIndex: 1, marginBottom: '16px' }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'flex-end',
                  gap: '4px',
                  height: isMobile ? '50px' : '70px',
                  marginBottom: '16px',
                }}>
                  {[1,2,3,4,5,6,7,8,9].map((i) => (
                    <div
                      key={i}
                      style={{
                        width: isMobile ? '5px' : '7px',
                        background: `linear-gradient(180deg, #00ffff, #ff00ff)`,
                        borderRadius: '4px',
                        boxShadow: '0 0 10px rgba(0,255,255,0.5)',
                        animation: `adEqualizer 0.8s ease-in-out infinite alternate`,
                        animationDelay: `${i * 0.1}s`,
                        height: `${30 + Math.sin(i) * 40}%`,
                      }}
                    />
                  ))}
                </div>

                <div style={{
                  fontSize: isMobile ? '64px' : '84px',
                  lineHeight: 1,
                  filter: 'drop-shadow(0 0 30px rgba(0,255,255,0.6))',
                  animation: 'adIconFloat 2s ease-in-out infinite',
                }}>
                  🎵
                </div>
              </div>

              <div style={{
                position: 'relative', zIndex: 1,
                color: '#00ffff',
                fontSize: isMobile ? '13px' : '14px',
                fontWeight: '600',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}>
                ✦ Воспроизведение аудио ✦
              </div>

              <audio ref={audioRef} preload="auto" style={{ display: 'none' }} />
            </div>
          )}

          {type === 'video' && (
            <div style={{
              position: 'relative',
              borderRadius: '20px',
              overflow: 'hidden',
              background: '#000',
              aspectRatio: '16/9',
              border: '1.5px solid rgba(0,255,255,0.3)',
              boxShadow: '0 0 40px rgba(0,255,255,0.15)',
            }}>
              <video
                ref={videoRef}
                preload="auto"
                playsInline
                controls={false}
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              />
            </div>
          )}
        </div>

        <div style={{ position: 'relative', zIndex: 1, marginTop: '18px' }}>
          <div style={{
            position: 'relative',
            height: '6px',
            background: 'rgba(255,255,255,0.06)',
            borderRadius: '3px',
            overflow: 'hidden',
            border: '1px solid rgba(255,255,255,0.05)',
          }}>
            <div style={{
              width: `${progress}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #00ffff, #7b2ff7, #ff00ff)',
              borderRadius: '3px',
              transition: 'width 0.3s linear',
              boxShadow: '0 0 15px rgba(0,255,255,0.6), 0 0 30px rgba(255,0,255,0.4)',
              position: 'relative',
              overflow: 'hidden',
            }}>
              <div style={{
                position: 'absolute', inset: 0,
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent)',
                animation: 'adShimmer 2s ease-in-out infinite',
              }} />
            </div>
          </div>
        </div>

        {error && (
          <div
            onClick={() => {
              const m = type === 'audio' ? audioRef.current : videoRef.current;
              if (m) m.play().then(() => setError(null)).catch(()=>{});
            }}
            style={{
              position: 'relative', zIndex: 1,
              marginTop: '14px',
              padding: '12px 16px',
              background: 'linear-gradient(135deg, rgba(255,165,0,0.15), rgba(255,100,0,0.1))',
              border: '1.5px solid rgba(255,159,67,0.5)',
              borderRadius: '12px',
              color: '#ff9f43',
              textAlign: 'center',
              cursor: 'pointer',
              fontSize: isMobile ? '12px' : '13px',
              fontWeight: '600',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
            }}
          >
            <span style={{ fontSize: '18px' }}>🔊</span>
            <span>{error}</span>
          </div>
        )}
      </div>

      <style>{`
        @keyframes adCardIn {
          from { opacity: 0; transform: translateY(30px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes adGlow {
          from { opacity: 0.6; transform: scale(1); }
          to { opacity: 1; transform: scale(1.15); }
        }
        @keyframes adIconFloat {
          0%, 100% { transform: translateY(0) rotate(-3deg); }
          50% { transform: translateY(-10px) rotate(3deg); }
        }
        @keyframes adIconPulse {
          0%, 100% { box-shadow: 0 0 20px rgba(0,255,255,0.3), inset 0 0 20px rgba(0,255,255,0.1); }
          50% { box-shadow: 0 0 35px rgba(0,255,255,0.6), inset 0 0 25px rgba(0,255,255,0.2); }
        }
        @keyframes adPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.7); }
        }
        @keyframes adEqualizer {
          0% { height: 20%; opacity: 0.7; }
          100% { height: 100%; opacity: 1; }
        }
        @keyframes adShimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </div>
  );
});

// ============= CHAT CONTENT (вынесено из App для оптимизации) =============
const ChatContent = React.memo(({
  isMobile,
  userId,
  userStatus,
  statusMessage,
  connectionStatus,
  moderationStatus,
  maintenanceMode,
  muffleEnabled,
  isConversationActive,
  isGroupVideo2Active,
  isGroupVideo6Active,
  volume,
  setVolume,
  isMicMuted,
  setIsMicMuted,
  localStream,
  selectedCategory,
  setSelectedCategory,
  showCategories,
  setShowCategories,
  userGender,
  setUserGender,
  searchGender,
  setSearchGender,
  selectedCountry,
  setSelectedCountry,
  showCountryDropdown,
  setShowCountryDropdown,
  userAge,
  setUserAge,
  showUserAgeDropdown,
  setShowUserAgeDropdown,
  searchAge,
  setSearchAge,
  showSearchAgeDropdown,
  setShowSearchAgeDropdown,
  showGenderDropdown,
  setShowGenderDropdown,
  showSearchGenderDropdown,
  setShowSearchGenderDropdown,
  getSelectedCategory,
  startSearch,
  stopSearch,
  nextPartner,
  stopConnection,
  quickStartVideoSingle,
  quickStartVideoGroup,
  onOpenReportModal,
}) => {
  return (
    <>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: isMobile ? '6px 8px' : '8px 12px',
        background: 'rgba(0,0,0,0.55)',
        backdropFilter: 'blur(10px)',
        borderRadius: '12px',
        marginBottom: isMobile ? '6px' : '10px',
        border: '1px solid rgba(0,247,255,0.15)',
        flexWrap: isMobile ? 'wrap' : 'nowrap',
        gap: isMobile ? '4px' : '0'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? '8px' : '12px' }}>
          <div
            className="logo-hover"
            style={{
              fontSize: isMobile ? '1.5rem' : '1.8rem',
              width: isMobile ? '38px' : '46px',
              height: isMobile ? '38px' : '46px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'linear-gradient(135deg, rgba(0,255,255,0.15), rgba(255,0,255,0.1))',
              borderRadius: '12px',
              border: '1px solid rgba(0,255,255,0.25)',
              boxShadow: '0 4px 20px rgba(0,255,255,0.2)',
              cursor: 'default',
            }}
          >🚀</div>
          <div>
            <h1 style={{ fontSize: isMobile ? '1.1rem' : '1.5rem', fontWeight: '800', background: 'linear-gradient(135deg, #00ffff, #ff00ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', margin: 0 }}>COSMIC VOICE</h1>
            <p style={{ fontSize: isMobile ? '0.65rem' : '0.75rem', color: '#9aa3b5', margin: 0 }}>Космическая связь</p>
          </div>
        </div>
        <OnlineUsersCounter isMobile={isMobile} />
      </div>

      <ConversationTimer isActive={isConversationActive} isMobile={isMobile} />

      <div style={{ marginBottom: isMobile ? '8px' : '12px' }}>
        {moderationStatus.is_banned && <div style={{ background: 'rgba(255,71,87,0.2)', border: '1px solid #ff4757', borderRadius: '8px', padding: '8px 12px', color: '#ff4757', marginBottom: '6px', fontSize: isMobile ? '12px' : '13px' }}>🚫 Ваш аккаунт заблокирован</div>}
        {moderationStatus.is_muted && <div style={{ background: 'rgba(255,165,0,0.2)', border: '1px solid #ff9f43', borderRadius: '8px', padding: '8px 12px', color: '#ff9f43', fontSize: isMobile ? '12px' : '13px' }}>🔇 Микрофон отключён модератором</div>}
      </div>

      {userStatus === 'connected' && !isGroupVideo2Active && !isGroupVideo6Active && (
        <div style={{ display: 'flex', gap: '8px', marginBottom: '10px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <button onClick={quickStartVideoSingle} className="form-select" style={{ padding: isMobile ? '8px 14px' : '10px 16px', background: 'linear-gradient(135deg, #00d2ff, #3a7bd5)', border: '1px solid transparent', borderRadius: '10px', color: 'white', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', flex: isMobile ? '1 1 45%' : '1', justifyContent: 'center', fontSize: isMobile ? '12px' : '13px' }}>
            <span style={{ fontSize: isMobile ? '14px' : '16px' }}>🎥</span> {isMobile ? '1 на 1' : 'Видео 1 на 1'}
          </button>
          <button onClick={quickStartVideoGroup} className="form-select" style={{ padding: isMobile ? '8px 14px' : '10px 16px', background: 'linear-gradient(135deg, #00c853, #00e676)', border: '1px solid transparent', borderRadius: '10px', color: 'white', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', flex: isMobile ? '1 1 45%' : '1', justifyContent: 'center', position: 'relative', fontSize: isMobile ? '12px' : '13px' }}>
            <span style={{ fontSize: isMobile ? '14px' : '16px' }}>👥</span> {isMobile ? '6 участников' : 'Групповое видео (6)'}
            <span style={{ position: 'absolute', top: '-6px', right: '-6px', background: 'linear-gradient(135deg, #ff4757, #ff3838)', color: 'white', fontSize: '9px', fontWeight: 'bold', padding: '1px 5px', borderRadius: '10px' }}>HOT</span>
          </button>
        </div>
      )}

      {isConversationActive && !isGroupVideo2Active && !isGroupVideo6Active && (
        <div style={{ marginBottom: '10px', display: 'flex', justifyContent: 'center' }}>
          <button onClick={onOpenReportModal} className="form-select" style={{ padding: '8px 16px', background: 'linear-gradient(135deg, #ff6b6b, #ee5a52)', border: '1px solid transparent', borderRadius: '8px', color: 'white', fontWeight: '600', cursor: 'pointer', fontSize: isMobile ? '12px' : '13px' }}>
            ⚠️ {isMobile ? 'Жалоба' : 'Сообщить о проблеме'}
          </button>
        </div>
      )}

      {userStatus === 'idle' && (
        <div style={{
          padding: isMobile ? '10px' : '14px',
          marginBottom: isMobile ? '8px' : '12px',
          background: 'rgba(0,0,0,0.55)',
          backdropFilter: 'blur(10px)',
          borderRadius: '12px',
          border: '1px solid rgba(0,247,255,0.1)',
          position: 'relative',
          zIndex: 100,
        }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: isMobile ? '8px' : '12px',
          }}>

            <div>
              <label style={{ fontSize: isMobile ? '0.78rem' : '0.85rem', color: '#aaa', display: 'block', marginBottom: '4px' }}>Тема</label>
              <div style={{ position: 'relative' }}>
                <button onClick={() => setShowCategories(!showCategories)} className="form-select" style={{ width: '100%', padding: isMobile ? '8px 12px' : '10px 14px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', fontSize: isMobile ? '12px' : '13px' }}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{getSelectedCategory().icon} {getSelectedCategory().name}</span>
                  <span style={{ flexShrink: 0, marginLeft: '4px' }}>▼</span>
                </button>
                {showCategories && (
                  <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'rgba(15,15,35,0.98)', border: '1px solid rgba(0,247,255,0.3)', borderRadius: '8px', marginTop: '4px', zIndex: 1000, boxShadow: '0 10px 30px rgba(0,0,0,0.8)', WebkitOverflowScrolling: 'touch', touchAction: 'pan-y', overscrollBehavior: 'contain', padding: '4px 0' }}>
                    {CATEGORIES.map(c => (
                      <button key={c.id} onClick={() => { setSelectedCategory(c.id); setShowCategories(false); }} style={{ width: '100%', padding: '10px 12px', background: selectedCategory === c.id ? 'rgba(0,247,255,0.1)' : 'transparent', border: 'none', color: selectedCategory === c.id ? '#00f7ff' : '#aaa', textAlign: 'left', cursor: 'pointer', display: 'flex', gap: '8px', fontSize: '12px' }}>
                        <span>{c.icon}</span><span>{c.name}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div>
              <label style={{ fontSize: isMobile ? '0.78rem' : '0.85rem', color: '#aaa', display: 'block', marginBottom: '4px' }}>Ищу</label>
              <div style={{ position: 'relative' }}>
                <button onClick={() => setShowSearchGenderDropdown(!showSearchGenderDropdown)} className="form-select" style={{ width: '100%', padding: isMobile ? '8px 12px' : '10px 14px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: isMobile ? '12px' : '13px' }}>
                  <span>{searchGender === 'male' ? '👨 Парней' : searchGender === 'female' ? '👩 Девушек' : '🌐 Всех'}</span>
                  <span>▼</span>
                </button>
                {showSearchGenderDropdown && (
                  <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'rgba(15,15,35,0.98)', border: '1px solid rgba(0,247,255,0.3)', borderRadius: '8px', marginTop: '4px', zIndex: 1000, boxShadow: '0 10px 30px rgba(0,0,0,0.8)', WebkitOverflowScrolling: 'touch', touchAction: 'pan-y', overscrollBehavior: 'contain', padding: '4px 0' }}>
                    {[{ v: 'any', l: '🌐 Всех' }, { v: 'male', l: '👨 Парней' }, { v: 'female', l: '👩 Девушек' }].map(o => (
                      <button key={o.v} onClick={() => { setSearchGender(o.v); setShowSearchGenderDropdown(false); }} style={{ width: '100%', padding: '10px 12px', background: searchGender === o.v ? 'rgba(0,247,255,0.1)' : 'transparent', border: 'none', color: searchGender === o.v ? '#00f7ff' : '#aaa', textAlign: 'left', cursor: 'pointer', fontSize: '12px' }}>{o.l}</button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div>
              <label style={{ fontSize: isMobile ? '0.78rem' : '0.85rem', color: '#aaa', display: 'block', marginBottom: '4px' }}>Твой пол</label>
              <div style={{ position: 'relative' }}>
                <button onClick={() => setShowGenderDropdown(!showGenderDropdown)} className="form-select" style={{ width: '100%', padding: isMobile ? '8px 12px' : '10px 14px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: isMobile ? '12px' : '13px' }}>
                  <span>{userGender === 'male' ? '👨 Парень' : userGender === 'female' ? '👩 Девушка' : '🌐 Не важно'}</span>
                  <span>▼</span>
                </button>
                {showGenderDropdown && (
                  <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'rgba(15,15,35,0.98)', border: '1px solid rgba(0,247,255,0.3)', borderRadius: '8px', marginTop: '4px', zIndex: 1000, boxShadow: '0 10px 30px rgba(0,0,0,0.8)', WebkitOverflowScrolling: 'touch', touchAction: 'pan-y', overscrollBehavior: 'contain', padding: '4px 0' }}>
                    {[{ v: 'any', l: '🌐 Не важно' }, { v: 'male', l: '👨 Парень' }, { v: 'female', l: '👩 Девушка' }].map(o => (
                      <button key={o.v} onClick={() => { setUserGender(o.v); setShowGenderDropdown(false); }} style={{ width: '100%', padding: '10px 12px', background: userGender === o.v ? 'rgba(0,247,255,0.1)' : 'transparent', border: 'none', color: userGender === o.v ? '#00f7ff' : '#aaa', textAlign: 'left', cursor: 'pointer', fontSize: '12px' }}>{o.l}</button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div>
              <label style={{ fontSize: isMobile ? '0.78rem' : '0.85rem', color: '#aaa', display: 'block', marginBottom: '4px' }}>🌍 Страна собеседника</label>
              <div style={{ position: 'relative' }}>
                <button onClick={() => setShowCountryDropdown(!showCountryDropdown)} className="form-select" style={{ width: '100%', padding: isMobile ? '8px 12px' : '10px 14px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: isMobile ? '12px' : '13px' }}>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{COUNTRIES.find(c => c.code === selectedCountry)?.name || '🌐 Любая'}</span>
                  <span style={{ flexShrink: 0, marginLeft: '4px' }}>▼</span>
                </button>
                {showCountryDropdown && (
                  <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'rgba(15,15,35,0.98)', border: '1px solid rgba(0,247,255,0.3)', borderRadius: '8px', marginTop: '4px', zIndex: 1000, boxShadow: '0 10px 30px rgba(0,0,0,0.8)', maxHeight: '220px', overflowY: 'auto', WebkitOverflowScrolling: 'touch', touchAction: 'pan-y', overscrollBehavior: 'contain' }}>
                    {COUNTRIES.map(c => (
                      <button key={c.code} onClick={() => { setSelectedCountry(c.code); setShowCountryDropdown(false); }} style={{ width: '100%', padding: '10px 12px', background: selectedCountry === c.code ? 'rgba(0,247,255,0.1)' : 'transparent', border: 'none', color: selectedCountry === c.code ? '#00f7ff' : '#aaa', textAlign: 'left', cursor: 'pointer', fontSize: '12px' }}>{c.name}</button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div>
              <label style={{ fontSize: isMobile ? '0.78rem' : '0.85rem', color: '#aaa', display: 'block', marginBottom: '4px' }}>👤Твой возраст</label>
              <div style={{ position: 'relative' }}>
                <button onClick={() => setShowUserAgeDropdown(!showUserAgeDropdown)} className="form-select" style={{ width: '100%', padding: isMobile ? '8px 12px' : '10px 14px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: isMobile ? '12px' : '13px' }}>
                  <span>{AGE_RANGES.find(a => a.id === userAge)?.name || '🌐 Любой'}</span>
                  <span>▼</span>
                </button>
                {showUserAgeDropdown && (
                  <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'rgba(15,15,35,0.98)', border: '1px solid rgba(0,247,255,0.3)', borderRadius: '8px', marginTop: '4px', zIndex: 1000, boxShadow: '0 10px 30px rgba(0,0,0,0.8)', WebkitOverflowScrolling: 'touch', touchAction: 'pan-y', overscrollBehavior: 'contain', padding: '4px 0' }}>
                    {AGE_RANGES.map(a => (
                      <button key={a.id} onClick={() => { setUserAge(a.id); setShowUserAgeDropdown(false); }} style={{ width: '100%', padding: '10px 12px', background: userAge === a.id ? 'rgba(0,247,255,0.1)' : 'transparent', border: 'none', color: userAge === a.id ? '#00f7ff' : '#aaa', textAlign: 'left', cursor: 'pointer', fontSize: '12px' }}>{a.name}</button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div>
              <label style={{ fontSize: isMobile ? '0.78rem' : '0.85rem', color: '#aaa', display: 'block', marginBottom: '4px' }}>🔍 Возраст собеседника</label>
              <div style={{ position: 'relative' }}>
                <button onClick={() => setShowSearchAgeDropdown(!showSearchAgeDropdown)} className="form-select" style={{ width: '100%', padding: isMobile ? '8px 12px' : '10px 14px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: 'white', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: isMobile ? '12px' : '13px' }}>
                  <span>{AGE_RANGES.find(a => a.id === searchAge)?.name || '🌐 Любой'}</span>
                  <span>▼</span>
                </button>
                {showSearchAgeDropdown && (
                  <div style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'rgba(15,15,35,0.98)', border: '1px solid rgba(0,247,255,0.3)', borderRadius: '8px', marginTop: '4px', zIndex: 1000, boxShadow: '0 10px 30px rgba(0,0,0,0.8)', WebkitOverflowScrolling: 'touch', touchAction: 'pan-y', overscrollBehavior: 'contain', padding: '4px 0' }}>
                    {AGE_RANGES.map(a => (
                      <button key={a.id} onClick={() => { setSearchAge(a.id); setShowSearchAgeDropdown(false); }} style={{ width: '100%', padding: '10px 12px', background: searchAge === a.id ? 'rgba(0,247,255,0.1)' : 'transparent', border: 'none', color: searchAge === a.id ? '#00f7ff' : '#aaa', textAlign: 'left', cursor: 'pointer', fontSize: '12px' }}>{a.name}</button>
                    ))}
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      )}

      <div style={{
        marginBottom: isMobile ? '6px' : '10px',
        padding: isMobile ? '6px 10px' : '8px 12px',
        background: 'rgba(0,0,0,0.55)',
        backdropFilter: 'blur(10px)',
        borderRadius: '10px',
        border: `1px solid ${userStatus === 'connected' ? 'rgba(0,255,136,0.25)' : userStatus === 'searching' ? 'rgba(59,130,246,0.25)' : 'rgba(255,255,255,0.08)'}`
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
          <span style={{ color: '#8a93a8', fontSize: isMobile ? '0.75rem' : '0.82rem' }}>Статус</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: userStatus === 'searching' ? '#3b82f6' : userStatus === 'connected' ? '#00ff88' : '#888' }} />
            <span style={{ color: 'white', fontSize: isMobile ? '0.8rem' : '0.85rem', fontWeight: '600' }}>{userStatus === 'searching' ? 'Поиск...' : userStatus === 'connected' ? 'В разговоре' : 'Готов'}</span>
          </div>
        </div>
        <div style={{ color: '#c2c8d6', fontSize: isMobile ? '0.8rem' : '0.88rem' }}>{statusMessage}</div>
      </div>

      <div>
        {userStatus === 'idle' && (
          <button
            onClick={startSearch}
            disabled={connectionStatus !== 'connected' || moderationStatus.is_banned || maintenanceMode || muffleEnabled}
            className="search-btn"
            style={{
              width: '100%',
              padding: isMobile ? '12px' : '16px',
              fontSize: isMobile ? '1rem' : '1.1rem',
              background: 'linear-gradient(135deg, #00d2ff, #7b2ff7, #ff00c8)',
              border: 'none',
              borderRadius: '14px',
              color: 'white',
              fontWeight: '800',
              letterSpacing: '0.03em',
              cursor: (connectionStatus !== 'connected' || moderationStatus.is_banned || maintenanceMode || muffleEnabled) ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              opacity: (connectionStatus !== 'connected' || moderationStatus.is_banned || maintenanceMode || muffleEnabled) ? 0.5 : 1,
              marginTop: '4px',
              boxShadow: '0 8px 24px rgba(0,210,255,0.35)',
            }}
          >
            <span>🎯</span> {isMobile ? 'Найти собеседника' : 'НАЧАТЬ ПОИСК'}
          </button>
        )}

        {userStatus === 'searching' && (
          <button onClick={stopSearch} style={{ width: '100%', padding: '14px', background: '#0a0f1a', border: '2px solid #1e3a5f', borderRadius: '14px', color: '#7aa2f7', fontWeight: '700', cursor: 'pointer', fontSize: isMobile ? '14px' : '15px' }}>
            ⏹️ ОСТАНОВИТЬ ПОИСК
          </button>
        )}
        {userStatus === 'connected' && (
          <div style={{
            padding: isMobile ? '8px' : '10px',
            background: 'linear-gradient(135deg, rgba(10,10,30,0.75), rgba(5,5,20,0.85))',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            borderRadius: '12px',
            border: '1px solid rgba(0,247,255,0.25)',
            boxShadow: '0 8px 32px rgba(0,247,255,0.15), inset 0 0 0 1px rgba(0,255,255,0.05)',
            position: 'relative',
            overflow: 'hidden',
          }}>
            <div style={{ position: 'absolute', top: '-30%', left: '-20%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 70%)', pointerEvents: 'none', zIndex: 0 }} />
            <div style={{ position: 'absolute', bottom: '-30%', right: '-20%', width: '60%', height: '60%', background: 'radial-gradient(circle, rgba(255,0,255,0.15) 0%, transparent 70%)', pointerEvents: 'none', zIndex: 0 }} />

            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', color: '#00ffff' }}>🔈</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={volume}
                  onChange={e => setVolume(parseFloat(e.target.value))}
                  style={{ flex: 1, accentColor: '#00ffff' }}
                />
                <span style={{ fontSize: '12px', color: '#00ffff' }}>🔊</span>
              </div>

              <div style={{
                display: 'flex',
                gap: isMobile ? '6px' : '8px',
                justifyContent: 'center',
                alignItems: 'center',
              }}>
                <ControlButton
                  icon={isMicMuted ? '🔇' : '🎤'}
                  label={isMicMuted ? 'Вкл' : 'Выкл'}
                  onClick={() => {
                    if (localStream) {
                      const n = !isMicMuted;
                      localStream.getAudioTracks().forEach(t => t.enabled = !n);
                      setIsMicMuted(n);
                    }
                  }}
                  variant={isMicMuted ? 'danger' : 'success'}
                  isActive={!isMicMuted}
                  disabled={moderationStatus.is_muted}
                  isMobile={isMobile}
                  compact={isMobile}
                />
                <ControlButton
                  icon="⏹️"
                  label="Стоп"
                  onClick={stopConnection}
                  variant="danger"
                  isMobile={isMobile}
                  compact={isMobile}
                />
                <ControlButton
                  icon="⏭️"
                  label="Дальше"
                  onClick={nextPartner}
                  variant="primary"
                  isMobile={isMobile}
                  compact={isMobile}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <div style={{
        textAlign: 'center',
        marginTop: isMobile ? '12px' : '16px',
        paddingTop: isMobile ? '10px' : '12px',
        borderTop: '1px solid rgba(0,247,255,0.15)',
        fontSize: isMobile ? '10px' : '11px',
        color: '#666',
        letterSpacing: '0.04em',
      }}>
        Cosmic-Voice. All rights reserved 2026-2027
      </div>
    </>
  );
}, (prev, next) => {
  if (prev.isMobile !== next.isMobile) return false;
  if (prev.userId !== next.userId) return false;
  if (prev.userStatus !== next.userStatus) return false;
  if (prev.statusMessage !== next.statusMessage) return false;
  if (prev.connectionStatus !== next.connectionStatus) return false;
  if (prev.moderationStatus !== next.moderationStatus) return false;
  if (prev.maintenanceMode !== next.maintenanceMode) return false;
  if (prev.muffleEnabled !== next.muffleEnabled) return false;
  if (prev.isConversationActive !== next.isConversationActive) return false;
  if (prev.isGroupVideo2Active !== next.isGroupVideo2Active) return false;
  if (prev.isGroupVideo6Active !== next.isGroupVideo6Active) return false;
  if (prev.volume !== next.volume) return false;
  if (prev.isMicMuted !== next.isMicMuted) return false;
  if (prev.localStream !== next.localStream) return false;
  if (prev.selectedCategory !== next.selectedCategory) return false;
  if (prev.showCategories !== next.showCategories) return false;
  if (prev.userGender !== next.userGender) return false;
  if (prev.searchGender !== next.searchGender) return false;
  if (prev.selectedCountry !== next.selectedCountry) return false;
  if (prev.showCountryDropdown !== next.showCountryDropdown) return false;
  if (prev.userAge !== next.userAge) return false;
  if (prev.showUserAgeDropdown !== next.showUserAgeDropdown) return false;
  if (prev.searchAge !== next.searchAge) return false;
  if (prev.showSearchAgeDropdown !== next.showSearchAgeDropdown) return false;
  if (prev.showGenderDropdown !== next.showGenderDropdown) return false;
  if (prev.showSearchGenderDropdown !== next.showSearchGenderDropdown) return false;
  return true;
});

// ============= ПРОВЕРКА МОЩНОСТИ УСТРОЙСТВА =============
const checkDevicePerformance = () => {
  const cpuCores = navigator.hardwareConcurrency || 2;
  const memoryGB = navigator.deviceMemory || 4;

  let gpuRenderer = 'unknown';
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (gl) {
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        gpuRenderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL).toLowerCase();
      }
    }
  } catch (e) {}

  const isOldGPU =
    gpuRenderer.includes('g210') ||
    gpuRenderer.includes('g100') ||
    gpuRenderer.includes('basic render') ||
    gpuRenderer.includes('swiftshader');

  const isLowCPU = cpuCores < 4;
  const isLowRAM = memoryGB < 4;

  const isWeak = isLowCPU || isLowRAM || isOldGPU;

  return {
    isWeak,
    cpuCores,
    memoryGB,
    gpuRenderer,
    reasons: [
      isLowCPU ? `Мало ядер CPU (${cpuCores} шт., нужно 4+)` : null,
      isLowRAM ? `Мало ОЗУ (${memoryGB} ГБ, нужно 4+)` : null,
      isOldGPU ? 'Устаревшая видеокарта' : null,
    ].filter(Boolean),
  };
};

// ============= WEAK DEVICE WARNING MODAL =============
const WeakDeviceWarningModal = React.memo(({ isVisible, onClose, deviceInfo, isMobile }) => {
  if (!isVisible || !deviceInfo) return null;

  return (
    <Portal>
      <div style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.85)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 100001,
        padding: isMobile ? '16px' : '30px',
      }}>
        <div style={{
          maxWidth: '480px',
          width: '100%',
          background: 'linear-gradient(145deg, rgba(20,10,30,0.98), rgba(10,5,20,0.99))',
          borderRadius: '24px',
          border: '2px solid rgba(255,200,0,0.6)',
          boxShadow: '0 30px 80px rgba(255,200,0,0.2)',
          padding: isMobile ? '28px 20px' : '36px 32px',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: isMobile ? '56px' : '72px', marginBottom: '16px' }}>⚠️</div>

          <h2 style={{
            fontSize: isMobile ? '1.3rem' : '1.5rem',
            fontWeight: '800',
            color: '#ffcc00',
            marginBottom: '12px',
          }}>
            Слабое устройство
          </h2>

          <p style={{
            color: '#c2c8d6',
            fontSize: isMobile ? '13px' : '14px',
            lineHeight: 1.6,
            marginBottom: '20px',
          }}>
            Ваше устройство может не справиться с групповым видео (6 участников). Это может вызвать <b style={{ color: '#ffcc00' }}>задержки, лаги и перегрев</b>.
          </p>

          <div style={{
            background: 'rgba(0,0,0,0.4)',
            borderRadius: '12px',
            padding: '14px 18px',
            marginBottom: '24px',
            textAlign: 'left',
          }}>
            <div style={{ fontSize: '12px', color: '#8a93a8', marginBottom: '8px', fontWeight: '600' }}>Причина:</div>
            {deviceInfo.reasons.map((r, i) => (
              <div key={i} style={{ fontSize: '13px', color: '#e2e8f0', marginBottom: '4px' }}>
                • {r}
              </div>
            ))}
          </div>

          <p style={{
            color: '#a0e7ff',
            fontSize: isMobile ? '12px' : '13px',
            lineHeight: 1.5,
            marginBottom: '24px',
          }}>
            💡 <b>Рекомендация:</b> Используйте режим <b>«Видео 1 на 1»</b> или <b>голосовой чат</b> для стабильной связи.
          </p>

          <button
            onClick={onClose}
            style={{
              width: '100%',
              padding: '14px',
              background: 'linear-gradient(135deg, #00d2ff, #7b2ff7)',
              border: 'none',
              borderRadius: '12px',
              color: 'white',
              fontSize: isMobile ? '14px' : '15px',
              fontWeight: '700',
              cursor: 'pointer',
              boxShadow: '0 4px 20px rgba(0,210,255,0.4)',
            }}
          >
            Понятно, продолжить
          </button>
        </div>
      </div>
    </Portal>
  );
});

// ============= ГЛАВНЫЙ КОМПОНЕНТ =============
function App() {
  const [isMobile, setIsMobile] = useState(isMobileDevice());

  const [userId] = useState(() => {
    const saved = localStorage.getItem('cosmicvoice_user_id');
    if (saved) return saved;
    const n = 'user_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('cosmicvoice_user_id', n);
    return n;
  });
  const [browserId] = useState(() => {
    const saved = localStorage.getItem('cosmicvoice_browser_id');
    if (saved) return saved;
    const n = 'browser_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('cosmicvoice_browser_id', n);
    return n;
  });

  const [showSystemMessage, setShowSystemMessage] = useState(false);
  const [systemMessage, setSystemMessage] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [userStatus, setUserStatus] = useState('idle');
  const [partnerId, setPartnerId] = useState(null);
  const [roomId, setRoomId] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Готов к подключению');
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [showCategories, setShowCategories] = useState(false);
  const [userGender, setUserGender] = useState('any');
  const [searchGender, setSearchGender] = useState('any');
  const [selectedCountry, setSelectedCountry] = useState('any');
  const [showCountryDropdown, setShowCountryDropdown] = useState(false);
  const [userAge, setUserAge] = useState('any');
  const [showUserAgeDropdown, setShowUserAgeDropdown] = useState(false);
  const [searchAge, setSearchAge] = useState('any');
  const [showSearchAgeDropdown, setShowSearchAgeDropdown] = useState(false);

  const [isConversationActive, setIsConversationActive] = useState(false);
  const [showReportModal, setShowReportModal] = useState(false);
  const [isSendingReport, setIsSendingReport] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [activeAds, setActiveAds] = useState({ audio: [], video: [] });
  const [currentAd, setCurrentAd] = useState(null);
  const [isPremium, setIsPremium] = useState(false);
  const [lastAudioAdTime, setLastAudioAdTime] = useState(0);
  const [lastVideoAdTime, setLastVideoAdTime] = useState(0);
  const [moderationStatus, setModerationStatus] = useState({ is_muted: false, is_banned: false, is_banned_extended: false, ban_type: null, ban_reason: null, warnings_count: 0, active_warnings: [], active_mutes: [], total_reports: 0 });
  const [banOverlay, setBanOverlay] = useState({ isVisible: false, type: null, reason: null, duration: null, banType: null });
  const [maintenanceMode, setMaintenanceMode] = useState(false);
  const [muffleEnabled, setMuffleEnabled] = useState(false);
  const [initialCheckComplete, setInitialCheckComplete] = useState(false);
  const [currentPage, setCurrentPage] = useState('chat');
  const [showGenderDropdown, setShowGenderDropdown] = useState(false);
  const [showSearchGenderDropdown, setShowSearchGenderDropdown] = useState(false);

  const [isGroupVideo2Active, setIsGroupVideo2Active] = useState(false);
  const [groupVideo2Participants, setGroupVideo2Participants] = useState([]);
  const [groupVideo2Messages, setGroupVideo2Messages] = useState([]);
  const [groupVideo2RoomName, setGroupVideo2RoomName] = useState(null);
  const [groupVideo2MicMuted, setGroupVideo2MicMuted] = useState(true);
  const [isVideoLoading, setIsVideoLoading] = useState(false);

  const [isGroupVideo6Active, setIsGroupVideo6Active] = useState(false);
  const [groupVideo6Participants, setGroupVideo6Participants] = useState([]);
  const [groupVideo6Messages, setGroupVideo6Messages] = useState([]);
  const [groupVideo6RoomName, setGroupVideo6RoomName] = useState(null);
  const [groupVideo6MicMuted, setGroupVideo6MicMuted] = useState(true);
  const [isSearchingNextRoom, setIsSearchingNextRoom] = useState(false);
  // 🆕 Стейт для предупреждения о слабом устройстве
  const [showWeakDeviceWarning, setShowWeakDeviceWarning] = useState(false);
  const [deviceInfo, setDeviceInfo] = useState(null);

  useEffect(() => {
    if (typeof ym === 'function' && process.env.NODE_ENV === 'production') {
      const pageTitles = {
        chat: 'Чат',
        home: 'Главная',
        contacts: 'Контакты',
        rules: 'Правила',
        faq: 'FAQ',
      };
      ym(YM_COUNTER_ID, 'hit', `/${currentPage}`, {
        title: pageTitles[currentPage] || currentPage,
        referer: document.referrer || undefined,
      });
    }
  }, [currentPage]);

  // 🆕 Проверка мощности устройства при старте (показываем только 1 раз за сессию)
  useEffect(() => {
    const info = checkDevicePerformance();
    setDeviceInfo(info);

    if (info.isWeak) {
      // Проверяем, показывали ли уже предупреждение в этой сессии
      const warningShown = sessionStorage.getItem('weak_device_warning_shown');
      
      if (!warningShown) {
        const timer = setTimeout(() => {
          setShowWeakDeviceWarning(true);
          // Запоминаем, что предупреждение показано
          sessionStorage.setItem('weak_device_warning_shown', 'true');
        }, 2000);
        return () => clearTimeout(timer);
      }
    }
  }, []);

  const wsManagerRef = useRef(null);
  const moderationWsRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const livekitRoomRef = useRef(null);
  const cancelVideoSearchRef = useRef(false);
  const videoSingleFirstEnterRef = useRef(true);
  const switchingVideo2Ref = useRef(false);
  const switchingVideo6Ref = useRef(false);
  const videoGroupFirstEnterRef = useRef(true);
  const nextRoomTimerRef = useRef(null);
  const localAudioRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const audioRef = useRef(typeof window !== 'undefined' ? new Audio(CONFIG.SESSION_SOUND) : null);
  const welcomeAudioRef = useRef(null);
  const welcomePlayedRef = useRef(false);
  const [isWelcomePlaying, setIsWelcomePlaying] = useState(false);
  const moderationReconnectTimeoutRef = useRef(null);
  const maintenanceCheckRef = useRef(null);
  const moderationCheckIntervalRef = useRef(null);
  const messageHandlerRef = useRef(null);
  const moderationCheckLockRef = useRef(false);
  const wsInitRef = useRef(false);
  const [isWsInitialized, setIsWsInitialized] = useState(false);
  const adCheckIntervalRef = useRef(null);
  const adPlayIntervalRef = useRef(null);
  const reportContextRef = useRef({ partnerId: null, partnerName: null, roomId: null });
  const spaceBgVideoRef = useRef(null);
  const prewarmedStreamRef = useRef(null);
  const isVideoModeActiveRef = useRef(false);

  useEffect(() => {
    isVideoModeActiveRef.current = isGroupVideo2Active || isGroupVideo6Active;
  }, [isGroupVideo2Active, isGroupVideo6Active]);

  useEffect(() => {
    const handleResize = () => setIsMobile(isMobileDevice());
    window.addEventListener('resize', handleResize, { passive: true });
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    return () => {
      if (nextRoomTimerRef.current) {
        clearTimeout(nextRoomTimerRef.current);
        nextRoomTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const video = spaceBgVideoRef.current;
    if (!video) return;

    if (isGroupVideo2Active || isGroupVideo6Active) {
      try {
        video.pause();
        video.currentTime = 0;
      } catch (e) {}
    } else {
      try {
        video.play().catch(() => {});
      } catch (e) {}
    }
  }, [isGroupVideo2Active, isGroupVideo6Active]);

  useEffect(() => {
    const handleBeforeUnload = () => {
      const roomName = window.__livekitRoomName;
      if (roomName) {
        const data = JSON.stringify({ user_id: userId, room_name: roomName });
        navigator.sendBeacon(
          `${CONFIG.BACKEND_URL}/api/livekit/leave`,
          new Blob([data], { type: 'application/json' })
        );
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('pagehide', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('pagehide', handleBeforeUnload);
    };
  }, [userId]);

  useEffect(() => {
    if (!partnerId && isConversationActive) {
      setIsConversationActive(false);
    }
  }, [partnerId, isConversationActive]);

  const getBrowserFingerprint = useCallback(async () => {
    try {
      const saved = localStorage.getItem('cosmicvoice_fingerprint');
      if (saved) return saved;
      const fp = { userAgent: navigator.userAgent, language: navigator.language, platform: navigator.platform, screen: `${screen.width}x${screen.height}`, timezone: new Date().getTimezoneOffset(), cookies: navigator.cookieEnabled, doNotTrack: navigator.doNotTrack || 'unknown' };
      const str = JSON.stringify(fp);
      let hash = 0;
      for (let i = 0; i < str.length; i++) { hash = ((hash << 5) - hash) + str.charCodeAt(i); hash |= 0; }
      const fh = Math.abs(hash).toString(16).substring(0, 8);
      localStorage.setItem('cosmicvoice_fingerprint', fh);
      return fh;
    } catch(e) { return 'unknown'; }
  }, []);

  const playSound = useCallback(() => { if (audioRef.current) { audioRef.current.currentTime = 0; audioRef.current.play().catch(()=>{}); } }, []);
  const playModerationSound = useCallback((type) => {
    try { const url = CONFIG.MODERATION_SOUNDS[type] || CONFIG.MODERATION_SOUNDS.warn; const a = new Audio(url); a.volume = 0.4; a.play().catch(()=>{}); } catch(e){}
  }, []);

  const recentNotificationsRef = useRef(new Map());

  const wasShownThisSession = (key) => {
    try {
      return sessionStorage.getItem(`notif_${key}`) === 'true';
    } catch (e) {
      return false;
    }
  };

  const markAsShown = (key) => {
    try {
      sessionStorage.setItem(`notif_${key}`, 'true');
    } catch (e) {}
  };

  const addNotification = useCallback((notification) => {
    if (isWelcomePlaying) return;

    const dedupKey = `${notification.type}|${notification.title}|${notification.message}`;
    const now = Date.now();
    const lastShown = recentNotificationsRef.current.get(dedupKey);

    if (lastShown && (now - lastShown) < 3000) {
      return;
    }

    recentNotificationsRef.current.set(dedupKey, now);

    if (recentNotificationsRef.current.size > 50) {
      const cutoff = now - 10000;
      for (const [key, ts] of recentNotificationsRef.current.entries()) {
        if (ts < cutoff) recentNotificationsRef.current.delete(key);
      }
    }

    const id = Date.now() + Math.random();
    setNotifications(prev => [...prev, { ...notification, id }]);
    setTimeout(() => setNotifications(prev => prev.filter(n => n.id !== id)), notification.timeout || 5000);
  }, [isWelcomePlaying]);

  const removeNotification = useCallback((id) => setNotifications(prev => prev.filter(n => n.id !== id)), []);

  const showBanOverlay = useCallback((type, reason, duration, banType) => {
    setBanOverlay({ isVisible: true, type, reason: reason || 'Нарушение правил', duration: duration || null, banType: banType || null });
  }, []);

  const closeBanOverlay = useCallback(() => {
    const wasBan = banOverlay.type === 'ban' || banOverlay.type === 'extended_ban';
    const wasKick = banOverlay.type === 'kick';
    setBanOverlay({ isVisible: false, type: null, reason: null, duration: null, banType: null });
    if (wasBan) {
      setTimeout(() => setBanOverlay({ isVisible: true, type: 'extended_ban', reason: moderationStatus.ban_reason || 'Нарушение правил', duration: null, banType: moderationStatus.ban_type }), 100);
      return;
    }
    if (wasKick) setTimeout(() => window.location.reload(), 500);
  }, [banOverlay.type, moderationStatus.ban_reason, moderationStatus.ban_type]);

  const getSelectedCategory = useCallback(() => CATEGORIES.find(c => c.id === selectedCategory) || CATEGORIES[0], [selectedCategory]);

  const handleSendReport = useCallback(async (reason, description) => {
    const ctx = reportContextRef.current;
    const targetId = ctx.partnerId || partnerId;
    const targetRoom = ctx.roomId || roomId;

    if (!targetId) {
      addNotification({ type: 'warning', icon: '⚠️', title: 'Ошибка', message: 'Нет активного собеседника для жалобы' });
      return;
    }

    setIsSendingReport(true);
    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/api/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          partner_id: targetId,
          room_id: targetRoom || 'unknown',
          reason,
          description,
          timestamp: new Date().toISOString()
        })
      });
      if (response.ok) {
        addNotification({ type: 'success', icon: '✅', title: 'Жалоба отправлена', message: 'Модераторы рассмотрят жалобу' });
        setShowReportModal(false);
      } else {
        const err = await response.json().catch(() => ({}));
        addNotification({ type: 'error', icon: '❌', title: 'Ошибка', message: err.detail || err.message || 'Не удалось отправить' });
      }
    } catch (error) {
      addNotification({ type: 'error', icon: '❌', title: 'Ошибка сети', message: 'Проверьте подключение' });
    } finally { setIsSendingReport(false); }
  }, [userId, partnerId, roomId, addNotification]);

  const forceMuteMicrophone = useCallback((mute = true) => {
    if (localStream) { localStream.getAudioTracks().forEach(t => { t.enabled = !mute; }); setIsMicMuted(mute); }
    if (window.__livekitRoom) {
      try { window.__livekitRoom.localParticipant.setMicrophoneEnabled(!mute); } catch(e){}
    }
  }, [localStream]);

  const sendMessage = useCallback((message) => {
    if (wsManagerRef.current && wsManagerRef.current.isConnected()) {
      try { wsManagerRef.current.sendMessage(message); return true; } catch(e) { return false; }
    }
    return false;
  }, []);

  const micCheckedRef = useRef(false);
  const micAvailableRef = useRef(null);

  const startSearch = useCallback(async () => {
    if (isSearchingNextRoom) {
      addNotification({
        type: 'warning',
        icon: '⏳',
        title: 'Подождите',
        message: 'Идёт поиск новой видео-комнаты. Дождитесь завершения или отмените.',
        timeout: 4000
      });
      return;
    }

    if (isGroupVideo2Active || isGroupVideo6Active) {
      addNotification({ type: 'warning', icon: '⚠️', title: 'Вы в видео-режиме', message: 'Сначала выйдите из видео' });
      return;
    }
    if (connectionStatus !== 'connected') { addNotification({ type: 'warning', icon: '⚠️', title: 'Нет подключения', message: 'Подождите' }); return; }
    if (moderationStatus.is_banned) { addNotification({ type: 'error', icon: '🚫', title: 'Заблокирован', message: 'Доступ запрещён' }); return; }

    if (!micCheckedRef.current) {
      try {
        setStatusMessage('Проверка микрофона...');
        const testStream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
        testStream.getTracks().forEach(t => t.stop());
        console.log('✅ [MIC] Микрофон доступен');
        micCheckedRef.current = true;
        micAvailableRef.current = true;
      } catch (micError) {
        console.error('❌ [MIC] Микрофон недоступен:', micError);
        micCheckedRef.current = true;
        micAvailableRef.current = false;
        addNotification({
          type: 'error',
          icon: '🎤',
          title: 'Нет доступа к микрофону',
          message: 'Разрешите доступ к микрофону в настройках браузера. Без него поиск собеседника невозможен.',
          timeout: 8000
        });
        setStatusMessage('Нет доступа к микрофону');
        return;
      }
    } else if (micAvailableRef.current === false) {
      addNotification({
        type: 'error',
        icon: '🎤',
        title: 'Микрофон недоступен',
        message: 'Разрешите доступ к микрофону и обновите страницу',
        timeout: 5000
      });
      return;
    }

    const searchParams = {
      category: selectedCategory || 'all',
      user_gender: userGender || 'any',
      search_gender: searchGender || 'any',
      country: selectedCountry || 'any',
      user_age: userAge || 'any',
      search_age: searchAge || 'any',
      topics: [],
    };
    sendMessage({ type: 'start_search', search_params: searchParams, user_id: userId });
    setUserStatus('searching'); setStatusMessage('Поиск собеседника...');
  }, [
    isSearchingNextRoom,
    connectionStatus,
    moderationStatus.is_banned,
    selectedCategory,
    userGender,
    searchGender,
    selectedCountry,
    userAge,
    searchAge,
    userId,
    sendMessage,
    addNotification,
    isGroupVideo2Active,
    isGroupVideo6Active
  ]);

  const cleanupVoiceChat = useCallback(() => {
    if (prewarmedStreamRef.current) {
      prewarmedStreamRef.current.getTracks().forEach(t => t.stop());
      prewarmedStreamRef.current = null;
    }
    if (localStream) { localStream.getTracks().forEach(t => t.stop()); setLocalStream(null); }
    if (peerConnectionRef.current) { peerConnectionRef.current.close(); peerConnectionRef.current = null; }
    setRemoteStream(null); setIsVoiceEnabled(false); setIsMicMuted(false); setIsConversationActive(false);
  }, [localStream]);

  const initializeVoiceChat = useCallback(async (roomIdParam, partnerIdParam) => {
    const currentRoomId = roomIdParam || roomId;
    const currentPartnerId = partnerIdParam || partnerId;
    if (!currentRoomId || !currentPartnerId) return;
    try {
      let stream = prewarmedStreamRef.current;
      prewarmedStreamRef.current = null;

      if (stream && stream.getAudioTracks()[0]?.readyState === 'live') {
        stream.getAudioTracks().forEach(t => { t.enabled = true; });
        log('⚡ [AUDIO] Reusing prewarmed stream');
      } else {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
        log('🎤 [AUDIO] Fresh stream');
      }

      setLocalStream(stream); setIsVoiceEnabled(true); setIsConversationActive(true); setStatusMessage('Голосовая связь установлена!');
      const pc = new RTCPeerConnection(ICE_SERVERS);
      peerConnectionRef.current = pc;
      stream.getAudioTracks().forEach(t => pc.addTrack(t, stream));

      const audioTimeout = setTimeout(() => {
        if (!pc.currentRemoteDescription) {
          console.warn('⏰ [AUDIO] Таймаут ожидания соединения (15 сек)');
          addNotification({
            type: 'warning',
            icon: '⏰',
            title: 'Не удалось установить связь',
            message: 'Проблемы с сетью у собеседника. Ищем нового...',
            timeout: 5000
          });
          try { pc.close(); } catch(e) {}
          peerConnectionRef.current = null;
          if (localStream) { localStream.getTracks().forEach(t => t.stop()); setLocalStream(null); }

          sendMessage({
            type: 'client_cant_connect',
            room_id: currentRoomId,
            partner_id: currentPartnerId,
            reason: 'webrtc_timeout'
          });

          setUserStatus('idle');
          setPartnerId(null);
          setRoomId(null);
          setIsConversationActive(false);
        }
      }, 15000);

      pc.ontrack = (e) => {
        clearTimeout(audioTimeout);
        const [rs] = e.streams;
        if (rs) setRemoteStream(rs);
      };

      pc.onconnectionstatechange = () => {
        if (pc.connectionState === 'connected') {
          clearTimeout(audioTimeout);
        }
      };
      pc.onicecandidate = (e) => { if (e.candidate) sendMessage({ type: 'webrtc_signal', signal: { type: 'ice-candidate', candidate: e.candidate }, room_id: currentRoomId, partner_id: currentPartnerId }); };
      const offer = await pc.createOffer({ offerToReceiveAudio: true });
      await pc.setLocalDescription(offer);
      sendMessage({ type: 'webrtc_signal', signal: { type: 'offer', offer }, room_id: currentRoomId, partner_id: currentPartnerId });
    } catch (error) {
      console.error('❌ [AUDIO] Microphone error:', error);
      addNotification({ type: 'error', icon: '🎤', title: 'Ошибка микрофона', message: 'Не удалось получить доступ. Поиск нового собеседника...', timeout: 5000 });

      sendMessage({
        type: 'client_cant_connect',
        room_id: currentRoomId,
        partner_id: currentPartnerId,
        reason: 'microphone_error'
      });

      setUserStatus('idle');
      setPartnerId(null);
      setRoomId(null);
      setIsConversationActive(false);
      setStatusMessage('Ошибка микрофона');
    }
  }, [roomId, partnerId, sendMessage, addNotification]);

  const handlePartnerDisconnected = useCallback((msg) => {
    setUserStatus('idle'); setPartnerId(null); setRoomId(null); setIsConversationActive(false);
    setStatusMessage(msg || 'Соединение разорвано');
    playSound(); cleanupVoiceChat();
  }, [playSound, cleanupVoiceChat]);

  const stopConnection = useCallback(() => {
    if (userStatus === 'connected') { sendMessage({ type: 'disconnect', room_id: roomId, partner_id: partnerId }); handlePartnerDisconnected('Вы разорвали соединение'); }
  }, [userStatus, roomId, partnerId, sendMessage, handlePartnerDisconnected]);

  const stopSearch = useCallback(() => sendMessage({ type: 'stop_search' }), [sendMessage]);
  const nextPartner = useCallback(() => { sendMessage({ type: 'next_partner', room_id: roomId, partner_id: partnerId }); cleanupVoiceChat(); }, [roomId, partnerId, sendMessage, cleanupVoiceChat]);

  const checkMaintenanceMode = useCallback(async () => {
    try {
      const r = await fetch(`${CONFIG.BACKEND_URL}/api/maintenance-status`, { cache: 'no-store' });
      if (r.ok) {
        const d = await r.json();
        const m = d.maintenance_mode || d.muffle_enabled;
        if (d.maintenance_message) setSystemMessage(d.maintenance_message);
        if (m) { setMaintenanceMode(true); localStorage.setItem('maintenance_mode', 'true'); return true; }
        else { setMaintenanceMode(false); setMuffleEnabled(false); localStorage.removeItem('maintenance_mode'); localStorage.removeItem('muffle_enabled'); return false; }
      }
      return false;
    } catch(e) { return maintenanceMode; }
  }, [maintenanceMode]);

  const checkServerStatus = useCallback(async () => { try { const r = await fetch(`${CONFIG.BACKEND_URL}/api/server-status`); return r.ok; } catch(e) { return false; } }, []);
  const checkMuffleStatus = useCallback(async () => {
    try { const r = await fetch(`${CONFIG.BACKEND_URL}/api/maintenance-status`, { cache: 'no-store' }); if (r.ok) { const s = await r.json(); if (s.muffle_enabled) { setMuffleEnabled(true); localStorage.setItem('muffle_enabled', 'true'); } else { setMuffleEnabled(false); localStorage.removeItem('muffle_enabled'); } } } catch(e){}
  }, []);

  const checkModerationStatus = useCallback(async () => {
    if (moderationCheckLockRef.current) return;
    moderationCheckLockRef.current = true;
    try {
      const fp = await getBrowserFingerprint();
      const url = `${CONFIG.BACKEND_URL}/api/user/${userId}/moderation-status?browser_id=${browserId}&fingerprint=${fp}`;
      const r = await fetch(url);
      if (r.ok) {
        const s = await r.json();
        setModerationStatus(prev => {
          if (s.is_muted && !prev.is_muted) {
            const muteId = s.active_mutes?.[0]?._id || s.active_mutes?.[0]?.id || `mute_${s.is_muted}`;
            if (!wasShownThisSession(`mute_${muteId}`)) {
              markAsShown(`mute_${muteId}`);
              playModerationSound('mute');
              addNotification({ type: 'mute', icon: '🔇', title: 'Вы заглушены', message: 'Модератор отключил микрофон', timeout: 8000 });
            }
            forceMuteMicrophone(true);
          }
          if (!s.is_muted && prev.is_muted) {
            if (!wasShownThisSession(`unmute_session`)) {
              markAsShown(`unmute_session`);
              playModerationSound('unmute');
              addNotification({ type: 'unmute', icon: '🔊', title: 'Заглушка снята', timeout: 5000 });
            }
            forceMuteMicrophone(false);
          }
          if (s.is_banned && !prev.is_banned) {
            if (!wasShownThisSession(`ban_session`)) {
              markAsShown(`ban_session`);
              playModerationSound('ban');
              showBanOverlay('ban', s.ban_reason, null, null);
            }
          }
          if (s.is_banned_extended && !prev.is_banned_extended) {
            if (!wasShownThisSession(`extended_ban_session`)) {
              markAsShown(`extended_ban_session`);
              playModerationSound('ban');
              showBanOverlay('extended_ban', s.ban_reason, null, s.ban_type);
            }
          }
          const banDuration = prev.ban_started_at ? Date.now() - prev.ban_started_at : 0;
          const wasBanned = (prev.is_banned || prev.is_banned_extended);
          const isNowBanned = (s.is_banned || s.is_banned_extended);

          if (wasBanned && !isNowBanned && banDuration > 5000) {
            playModerationSound('unban');
            addNotification({ type: 'unban', icon: '✅', title: 'Блокировка снята', timeout: 5000 });
            setBanOverlay({ isVisible: false, type: null, reason: null, duration: null, banType: null });
            if (wsManagerRef.current) { wsManagerRef.current.resetBlock(); wsInitRef.current = false; setIsWsInitialized(false); }
          }
          if (s.warnings_count > prev.warnings_count) {
            const warnId = s.active_warnings?.[0]?._id || s.active_warnings?.[0]?.id || `warn_${s.warnings_count}`;
            if (!wasShownThisSession(`warn_${warnId}`)) {
              markAsShown(`warn_${warnId}`);
              playModerationSound('warn');
              addNotification({
                type: 'warn',
                icon: '⚠️',
                title: 'Предупреждение',
                message: translateReason(s.active_warnings?.[0]?.reason),
                submessage: `Всего: ${s.warnings_count}`,
                timeout: 10000
              });
            }
          }
          return {
            is_muted: s.is_muted || false,
            is_banned: s.is_banned || false,
            is_banned_extended: s.is_banned_extended || false,
            ban_type: s.ban_type || null,
            ban_reason: s.ban_reason || null,
            warnings_count: s.warnings_count || 0,
            active_warnings: s.active_warnings || [],
            active_mutes: s.active_mutes || [],
            total_reports: s.total_reports || 0,
            ban_started_at: (s.is_banned || s.is_banned_extended) && !(prev.is_banned || prev.is_banned_extended)
              ? Date.now()
              : (prev.ban_started_at || null)
          };
        });
      }
    } catch(e) {}
    finally { moderationCheckLockRef.current = false; }
  }, [userId, browserId, getBrowserFingerprint, addNotification, forceMuteMicrophone, playModerationSound, showBanOverlay]);

  const handleWebRTCSignal = useCallback(async (signal) => {
    const pc = peerConnectionRef.current;
    if (!pc) return;
    try {
      if (signal.type === 'offer') { await pc.setRemoteDescription(signal.offer); const answer = await pc.createAnswer(); await pc.setLocalDescription(answer); sendMessage({ type: 'webrtc_signal', signal: { type: 'answer', answer } }); }
      else if (signal.type === 'answer') { await pc.setRemoteDescription(signal.answer); }
      else if (signal.type === 'ice-candidate') { if (pc.remoteDescription) await pc.addIceCandidate(new RTCIceCandidate(signal.candidate)); }
    } catch(e) {}
  }, [sendMessage]);

  // ==================== LIVEKIT 1-НА-1 ====================
  const quickStartVideoSingle = useCallback(async () => {
    setIsVideoLoading(true);

    if (isConversationActive || userStatus === 'connected') { stopConnection(); cleanupVoiceChat(); }

    let mediaStream;
    try {
      log('🎥 [MEDIA] Запрашиваем доступ к камере и микрофону...');
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
      log('✅ [MEDIA] Доступ получен');
    } catch (mediaError) {
      console.error('❌ [MEDIA] Доступ отклонён:', mediaError);
      addNotification({ type: 'error', icon: '📷', title: 'Нет доступа к камере', message: 'Разрешите доступ к камере и микрофону. Видео-режим недоступен без камеры.', timeout: 8000 });
      setIsVideoLoading(false);
      return;
    }

    try {
      const nickname = localStorage.getItem('cosmic_nickname') || 'User';

      const res = await fetch(`${CONFIG.BACKEND_URL}/api/livekit/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, nickname, room_type: 'single' })
      });
      if (!res.ok) throw new Error('Не удалось получить токен');
      const { token, url, room: roomName } = await res.json();
      log('🎥 [LIVEKIT 1-1] Connecting to room:', roomName);

      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        videoCaptureDefaults: {
          resolution: VideoPresets.h720.resolution
        },
        publishDefaults: {
          simulcast: false,
          videoEncoding: {
            maxBitrate: 1_200_000,
            maxFramerate: 30,
          },
          audioPreset: { maxBitrate: 32_000 },
          dtx: true,
          red: false,
        },
      });

      window.__livekitRoom = room;
      window.__livekitRoomName = roomName;

      const handleTrackSubscribed = (track, publication, participant) => {
        log('🎥 [1-1] Track subscribed:', track.kind, 'from', participant.identity);

        const element = track.attach();
        const stream = element.srcObject instanceof MediaStream
          ? element.srcObject
          : new MediaStream([track.mediaStreamTrack]);

        setGroupVideo2Participants(prev => {
          const exists = prev.some(p => p.id === participant.identity);
          if (!exists) {
            return [...prev, {
              id: participant.identity,
              name: participant.name || participant.identity.slice(0, 6),
              videoStream: track.kind === Track.Kind.Video ? stream : null,
              audioStream: track.kind === Track.Kind.Audio ? stream : null,
              isActive: true,
              isMuted: false,
              _videoTrack: track.kind === Track.Kind.Video ? track : null,
              _audioTrack: track.kind === Track.Kind.Audio ? track : null,
            }];
          }
          return prev.map(p => {
            if (p.id !== participant.identity) return p;
            if (track.kind === Track.Kind.Video) {
              return { ...p, videoStream: stream, isActive: true, _videoTrack: track };
            }
            if (track.kind === Track.Kind.Audio) {
              return { ...p, audioStream: stream, _audioTrack: track };
            }
            return p;
          });
        });
      };

      room
        .on(RoomEvent.ParticipantConnected, (participant) => {
          log('👤 [1-1] joined:', participant.identity);
          setGroupVideo2Participants(prev => {
            if (prev.some(p => p.id === participant.identity)) return prev;
            return [...prev, { id: participant.identity, name: participant.name || participant.identity.slice(0,6), videoStream: null, audioStream: null, isActive: true, isMuted: false }];
          });
          participant.trackPublications.forEach(pub => {
            if (pub.track && pub.isSubscribed) {
              handleTrackSubscribed(pub.track, pub, participant);
            }
          });
        })
        .on(RoomEvent.ParticipantDisconnected, (participant) => {
          setGroupVideo2Participants(prev => prev.filter(p => p.id !== participant.identity));
        })
        .on(RoomEvent.TrackSubscribed, handleTrackSubscribed)
        .on(RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
          try { track.detach().forEach(el => el.remove()); } catch (e) {}
          setGroupVideo2Participants(prev => prev.map(p => {
            if (p.id !== participant.identity) return p;
            if (track.kind === Track.Kind.Video) return { ...p, videoStream: null, isActive: false, _videoTrack: null };
            if (track.kind === Track.Kind.Audio) return { ...p, audioStream: null, _audioTrack: null };
            return p;
          }));
        })
        .on(RoomEvent.Disconnected, () => {
          if (switchingVideo2Ref.current) {
            log('🔄 [1-1] Disconnected during switch, staying in mode');
            livekitRoomRef.current = null;
            window.__livekitRoom = null;
            return;
          }
          setIsGroupVideo2Active(false);
          livekitRoomRef.current = null;
          window.__livekitRoom = null;
        });

      await room.connect(url, token);
      livekitRoomRef.current = room;

      if (cancelVideoSearchRef.current) {
        log('⏹️ [1-1] Search cancelled, disconnecting from', roomName);
        try { room.disconnect(); } catch(e){}
        livekitRoomRef.current = null;
        window.__livekitRoom = null;
        window.__livekitRoomName = null;
        return;
      }

      const videoTrack = mediaStream.getVideoTracks()[0];
      const audioTrack = mediaStream.getAudioTracks()[0];

      if (videoTrack) {
        await room.localParticipant.publishTrack(videoTrack, { source: Track.Source.Camera });
        log('✅ [PUBLISH] Video track published');
      }
      if (audioTrack) {
        await room.localParticipant.publishTrack(audioTrack, { source: Track.Source.Microphone });
        audioTrack.enabled = false;
        log('✅ [PUBLISH] Audio track published');
      }

      setGroupVideo2MicMuted(true);

      room.remoteParticipants.forEach(participant => {
        setGroupVideo2Participants(prev => {
          if (prev.some(p => p.id === participant.identity)) return prev;
          return [...prev, { id: participant.identity, name: participant.name || participant.identity.slice(0,6), videoStream: null, audioStream: null, isActive: true, isMuted: false }];
        });
        participant.trackPublications.forEach(pub => {
          if (pub.track && pub.isSubscribed) {
            handleTrackSubscribed(pub.track, pub, participant);
          }
        });
      });

      const localVideoPub = room.localParticipant.getTrackPublication(Track.Source.Camera);
      let localVideoStream = null;
      let localVideoTrackRef = null;

      if (localVideoPub && localVideoPub.videoTrack) {
        localVideoTrackRef = localVideoPub.videoTrack;
        try {
          const localVideoEl = localVideoTrackRef.attach();
          localVideoStream = localVideoEl.srcObject instanceof MediaStream
            ? localVideoEl.srcObject
            : new MediaStream([localVideoTrackRef.mediaStreamTrack]);
          log('✅ [1-1] Local video attached: stream=', !!localVideoStream, 'track=', !!localVideoTrackRef);
        } catch (e) {
          console.error('❌ [1-1] Local attach error:', e);
          localVideoStream = new MediaStream([localVideoTrackRef.mediaStreamTrack]);
        }
      } else {
        warn('❌ [1-1] Local video publication NOT FOUND');
        if (videoTrack) {
          localVideoStream = new MediaStream([videoTrack]);
        }
      }

      setGroupVideo2Participants(prev => {
        const others = prev.filter(p => p.id !== userId);
        return [
          {
            id: userId,
            name: `${nickname} (Вы)`,
            videoStream: localVideoStream,
            audioStream: null,
            isActive: true,
            isMuted: false,
            _videoTrack: localVideoTrackRef,
          },
          ...others
        ];
      });

      setGroupVideo2Messages([]);
      setGroupVideo2RoomName(roomName);
      setIsGroupVideo2Active(true);
      setIsSearchingNextRoom(false);

      if (videoSingleFirstEnterRef.current) {
        addNotification({
          type: 'success',
          icon: '🎥',
          title: 'Видео 1 на 1',
          message: `Комната: ${(roomName || window.__livekitRoomName || '').slice(0,20)}...`,
          timeout: 5000
        });
        videoSingleFirstEnterRef.current = false;
      }
    } catch (error) {
      console.error('❌ [LIVEKIT 1-1]', error);
      addNotification({ type: 'error', icon: '❌', title: 'Ошибка подключения', message: error.message, timeout: 5000 });
    } finally {
      setIsVideoLoading(false);
    }
  }, [isConversationActive, userStatus, stopConnection, cleanupVoiceChat, userId, addNotification]);

  const exitVideo2Mode = useCallback(() => {
    videoSingleFirstEnterRef.current = true;
    switchingVideo2Ref.current = false;
    const roomName = window.__livekitRoomName;
    if (livekitRoomRef.current) { try { livekitRoomRef.current.disconnect(); } catch(e){} livekitRoomRef.current = null; }
    window.__livekitRoom = null;
    if (roomName) {
      fetch(`${CONFIG.BACKEND_URL}/api/livekit/leave`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_id: userId, room_name: roomName }) }).catch(()=>{});
      window.__livekitRoomName = null;
    }
    setIsGroupVideo2Active(false);
    setGroupVideo2Participants([]);
    setGroupVideo2Messages([]);
    setGroupVideo2RoomName(null);
    setIsConversationActive(false);
    addNotification({ type: 'info', icon: '🔊', title: 'Выход из видео', message: 'Вы вернулись в голосовой режим' });
  }, [userId, addNotification]);

  const nextVideo2Mode = useCallback(async (isCancel = false) => {
    if (isCancel) {
      cancelVideoSearchRef.current = true;
      addNotification({ type: 'info', icon: '⏹️', title: 'Поиск отменён', timeout: 3000 });
      return;
    }
    cancelVideoSearchRef.current = false;
    switchingVideo2Ref.current = true;

    const roomName = window.__livekitRoomName;
    if (livekitRoomRef.current) {
      try { livekitRoomRef.current.disconnect(); } catch(e){}
      livekitRoomRef.current = null;
    }
    window.__livekitRoom = null;

    if (roomName) {
      try {
        await fetch(`${CONFIG.BACKEND_URL}/api/livekit/leave`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, room_name: roomName })
        });
        log('✅ [LEAVE] Backend removed us from', roomName);
      } catch(e) {
        log('⚠️ [LEAVE] Error:', e);
      }
      window.__livekitRoomName = null;
    }

    setGroupVideo2Participants(prev => prev.filter(p => p.id === userId || p.id === 'local'));
    setGroupVideo2Messages([]);

    setTimeout(() => {
      quickStartVideoSingle();
      setTimeout(() => {
        switchingVideo2Ref.current = false;
      }, 500);
    }, 100);
    setTimeout(() => setIsSearchingNextRoom(false), 1500);
  }, [userId, quickStartVideoSingle, addNotification]);

  const handleVideo2SendMessage = useCallback((text, nickname) => {
    const newMessage = { id: Date.now(), text, senderId: userId, senderName: userId.slice(0,6), senderNickname: nickname || 'User', isOwn: true, timestamp: new Date().toISOString() };
    setGroupVideo2Messages(prev => {
      const updated = [...prev, newMessage];
      return updated.length > 100 ? updated.slice(-100) : updated;
    });
    sendMessage({ type: 'video_message', message: text, room_id: roomId || window.__livekitRoomName, user_id: userId, user_name: userId.slice(0,6), nickname: nickname || 'User' });
  }, [userId, roomId, sendMessage]);

  const handleVideo2Report = useCallback((participantId, participantName) => {
    reportContextRef.current = { partnerId: participantId, partnerName: participantName, roomId: window.__livekitRoomName || roomId };
    setShowReportModal(true);
  }, [roomId]);

  const toggleVideo2Mic = useCallback(() => {
    const newMuted = !groupVideo2MicMuted;
    if (window.__livekitRoom) {
      try { window.__livekitRoom.localParticipant.setMicrophoneEnabled(!newMuted); } catch(e){}
    }
    setGroupVideo2MicMuted(newMuted);
  }, [groupVideo2MicMuted]);

  // ==================== LIVEKIT 6 КАМЕР ====================
    const quickStartVideoGroup = useCallback(async () => {
    // 🆕 Предупреждаем о слабом устройстве перед запуском 6-режима
    if (deviceInfo?.isWeak) {
      // Показываем confirm только если ещё не показывали в этой сессии
      const confirmShown = sessionStorage.getItem('weak_device_confirm_shown');
      if (!confirmShown) {
        const confirmed = window.confirm(
          '⚠️ Ваше устройство может не справиться с 6 камерами.\n\n' +
          'Возможны лаги и задержки.\n\n' +
          'Рекомендуем использовать "Видео 1 на 1" или голосовой чат.\n\n' +
          'Продолжить?'
        );
        if (!confirmed) return;
        sessionStorage.setItem('weak_device_confirm_shown', 'true');
      }
    }

    // 🆕 Показываем спиннер
    setIsVideoLoading(true);

    if (isConversationActive || userStatus === 'connected') { stopConnection(); cleanupVoiceChat(); }

    let mediaStream;
    try {
      log('🎥 [MEDIA] Запрашиваем доступ к камере и микрофону...');
      // 🆕 РАЗРЕШЕНИЕ ДЛЯ 6-РЕЖИМА: 720p (как вы просили)
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
      log('✅ [MEDIA] Доступ получен');
    } catch (mediaError) {
      console.error('❌ [MEDIA] Доступ отклонён:', mediaError);
      addNotification({ type: 'error', icon: '📷', title: 'Нет доступа к камере', message: 'Разрешите доступ к камере и микрофону. Видео-режим недоступен без камеры.', timeout: 8000 });
      setIsVideoLoading(false);
      return;
    }

    try {
      const nickname = localStorage.getItem('cosmic_nickname') || 'User';

      const res = await fetch(`${CONFIG.BACKEND_URL}/api/livekit/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, nickname, room_type: 'group6' })
      });
      if (!res.ok) throw new Error('Не удалось получить токен');
      const { token, url, room: roomName } = await res.json();
      log('🎥 [LIVEKIT 6] Connecting to room:', roomName);

      // 🆕 НАСТРОЙКИ LIVEKIT ДЛЯ 6-РЕЖИМА (720p)
      const room = new Room({
        adaptiveStream: false,
        dynacast: true,
        videoCaptureDefaults: {
          resolution: VideoPresets.h720.resolution, // 🆕 720p
        },
        publishDefaults: {
          simulcast: true, // 🆕 Оставляем simulcast для лучшей адаптации
          videoSimulcastLayers: [
            VideoPresets.h360,
            VideoPresets.h180,
          ],
          videoEncoding: {
            maxBitrate: 400_000, // 🆕 400 kbps
            maxFramerate: 24,    // 🆕 24 fps
          },
          audioPreset: { maxBitrate: 24_000 },
          dtx: true,
          red: false,
        },
      });

      window.__livekitRoom = room;
      window.__livekitRoomName = roomName;

      const handleTrackSubscribed = (track, publication, participant) => {
        setIsSearchingNextRoom(false);
        const element = track.attach();
        const stream = element.srcObject instanceof MediaStream
          ? element.srcObject
          : new MediaStream([track.mediaStreamTrack]);

        setGroupVideo6Participants(prev => {
          const exists = prev.some(p => p.id === participant.identity);
          const nameToUse = participant.name || participant.identity.slice(0, 6);

          if (!exists) {
            return [...prev, {
              id: participant.identity,
              name: nameToUse,
              videoStream: track.kind === Track.Kind.Video ? stream : null,
              audioStream: track.kind === Track.Kind.Audio ? stream : null,
              isActive: true,
              isMuted: false,
              _videoTrack: track.kind === Track.Kind.Video ? track : null,
              _audioTrack: track.kind === Track.Kind.Audio ? track : null,
            }];
          }
          return prev.map(p => {
            if (p.id !== participant.identity) return p;
            const updated = { ...p, name: nameToUse };
            if (track.kind === Track.Kind.Video) {
              return { ...updated, videoStream: stream, isActive: true, _videoTrack: track };
            }
            if (track.kind === Track.Kind.Audio) {
              return { ...updated, audioStream: stream, _audioTrack: track };
            }
            return updated;
          });
        });
      };

      room
        .on(RoomEvent.ParticipantConnected, (participant) => {
          log('👤 [6] joined:', participant.identity);
          setIsSearchingNextRoom(false);

          const newEntry = {
            id: participant.identity,
            name: participant.name || participant.identity.slice(0, 6),
            videoStream: null,
            audioStream: null,
            isActive: true,
            isMuted: false,
          };

          participant.trackPublications.forEach(pub => {
            if (!pub.track || !pub.isSubscribed) return;
            if (pub.track.kind === Track.Kind.Video) {
              newEntry._videoTrack = pub.track;
              try {
                const el = pub.track.attach();
                newEntry.videoStream = el.srcObject instanceof MediaStream
                  ? el.srcObject
                  : new MediaStream([pub.track.mediaStreamTrack]);
              } catch (e) {
                newEntry.videoStream = new MediaStream([pub.track.mediaStreamTrack]);
              }
            }
            if (pub.track.kind === Track.Kind.Audio) {
              newEntry._audioTrack = pub.track;
              try {
                const el = pub.track.attach();
                newEntry.audioStream = el.srcObject instanceof MediaStream
                  ? el.srcObject
                  : new MediaStream([pub.track.mediaStreamTrack]);
              } catch (e) {
                newEntry.audioStream = new MediaStream([pub.track.mediaStreamTrack]);
              }
            }
          });

          setGroupVideo6Participants(prev => {
            if (prev.some(p => p.id === participant.identity)) return prev;
            return [...prev, newEntry];
          });
        })
        .on(RoomEvent.ParticipantDisconnected, (participant) => {
          setGroupVideo6Participants(prev => prev.filter(p => p.id !== participant.identity));
        })
        .on(RoomEvent.TrackSubscribed, handleTrackSubscribed)
        .on(RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
          try { track.detach().forEach(el => el.remove()); } catch (e) {}
          setGroupVideo6Participants(prev => prev.map(p => {
            if (p.id !== participant.identity) return p;
            if (track.kind === Track.Kind.Video) return { ...p, videoStream: null, isActive: false, _videoTrack: null };
            if (track.kind === Track.Kind.Audio) return { ...p, audioStream: null, _audioTrack: null };
            return p;
          }));
        })
        .on(RoomEvent.Disconnected, () => {
          if (switchingVideo6Ref.current) {
            log('🔄 [6] Disconnected during switch, staying in mode');
            livekitRoomRef.current = null;
            window.__livekitRoom = null;
            return;
          }
          setIsGroupVideo6Active(false);
          livekitRoomRef.current = null;
          window.__livekitRoom = null;
        });

      await room.connect(url, token);
      livekitRoomRef.current = room;

      if (cancelVideoSearchRef.current) {
        log('⏹️ [6] Search cancelled, disconnecting from', roomName);
        try { room.disconnect(); } catch(e){}
        livekitRoomRef.current = null;
        window.__livekitRoom = null;
        window.__livekitRoomName = null;
        return;
      }

      const videoTrack = mediaStream.getVideoTracks()[0];
      const audioTrack = mediaStream.getAudioTracks()[0];

      if (videoTrack) {
        await room.localParticipant.publishTrack(videoTrack, { source: Track.Source.Camera });
        log('✅ [PUBLISH] Video track published');
      }
      if (audioTrack) {
        await room.localParticipant.publishTrack(audioTrack, { source: Track.Source.Microphone });
        audioTrack.enabled = false;
        log('✅ [PUBLISH] Audio track published');
      }

      setGroupVideo6MicMuted(true);

      const initialParticipants = [];

      room.remoteParticipants.forEach(participant => {
        const nameToUse =
          (participant.name && participant.name !== 'User')
            ? participant.name
            : participant.identity.slice(0, 6);

        const entry = {
          id: participant.identity,
          name: nameToUse,
          videoStream: null,
          audioStream: null,
          isActive: true,
          isMuted: false,
        };

        participant.trackPublications.forEach(pub => {
          if (pub.track && pub.isSubscribed) {
            if (pub.track.kind === Track.Kind.Video) {
              entry._videoTrack = pub.track;
              try {
                const el = pub.track.attach();
                entry.videoStream = el.srcObject instanceof MediaStream
                  ? el.srcObject
                  : new MediaStream([pub.track.mediaStreamTrack]);
              } catch (e) {
                entry.videoStream = new MediaStream([pub.track.mediaStreamTrack]);
              }
            }
            if (pub.track.kind === Track.Kind.Audio) {
              entry._audioTrack = pub.track;
              try {
                const el = pub.track.attach();
                entry.audioStream = el.srcObject instanceof MediaStream
                  ? el.srcObject
                  : new MediaStream([pub.track.mediaStreamTrack]);
              } catch (e) {
                entry.audioStream = new MediaStream([pub.track.mediaStreamTrack]);
              }
            }
          }
        });

        initialParticipants.push(entry);
      });

      if (initialParticipants.length > 0) {
        setGroupVideo6Participants(prev => {
          const existingIds = new Set(prev.map(p => p.id));
          const newOnes = initialParticipants.filter(p => !existingIds.has(p.id));
          return [...prev, ...newOnes];
        });
      }

      const localVideoPub = room.localParticipant.getTrackPublication(Track.Source.Camera);
      let localVideoStream = null;
      let localVideoTrackRef = null;

      if (localVideoPub && localVideoPub.videoTrack) {
        localVideoTrackRef = localVideoPub.videoTrack;
        try {
          const localVideoEl = localVideoTrackRef.attach();
          localVideoStream = localVideoEl.srcObject instanceof MediaStream
            ? localVideoEl.srcObject
            : new MediaStream([localVideoTrackRef.mediaStreamTrack]);
          log('✅ [6] Local video attached: stream=', !!localVideoStream, 'track=', !!localVideoTrackRef);
        } catch (e) {
          console.error('❌ [6] Local attach error:', e);
          localVideoStream = new MediaStream([localVideoTrackRef.mediaStreamTrack]);
        }
      } else {
        warn('❌ [6] Local video publication NOT FOUND');
        if (videoTrack) {
          localVideoStream = new MediaStream([videoTrack]);
        }
      }

      setGroupVideo6Participants(prev => {
        const others = prev.filter(p => p.id !== userId);
        return [
          {
            id: userId,
            name: `${nickname} (Вы)`,
            videoStream: localVideoStream,
            audioStream: null,
            isActive: true,
            isMuted: false,
            _videoTrack: localVideoTrackRef,
          },
          ...others
        ];
      });

      setGroupVideo6Messages([]);
      setGroupVideo6RoomName(roomName);
      setIsGroupVideo6Active(true);

      if (videoGroupFirstEnterRef.current) {
        addNotification({
          type: 'success',
          icon: '👥',
          title: 'Групповое видео',
          message: `Комната: ${roomName.slice(0,20)}...`,
          timeout: 5000
        });
        videoGroupFirstEnterRef.current = false;
      }
    } catch (error) {
      console.error('❌ [LIVEKIT 6]', error);
      addNotification({ type: 'error', icon: '❌', title: 'Ошибка', message: error.message, timeout: 5000 });
      setIsSearchingNextRoom(false);
    } finally {
      setIsVideoLoading(false);
    }
  }, [isConversationActive, userStatus, stopConnection, cleanupVoiceChat, userId, addNotification, deviceInfo]);

  const exitGroupVideo6Mode = useCallback(() => {
    videoGroupFirstEnterRef.current = true;
    switchingVideo6Ref.current = false;
    if (nextRoomTimerRef.current) {
      clearTimeout(nextRoomTimerRef.current);
      nextRoomTimerRef.current = null;
    }
    setIsSearchingNextRoom(false);

    const roomName = window.__livekitRoomName;
    if (livekitRoomRef.current) {
      try { livekitRoomRef.current.disconnect(); } catch(e){}
      livekitRoomRef.current = null;
    }
    window.__livekitRoom = null;
    if (roomName) {
      fetch(`${CONFIG.BACKEND_URL}/api/livekit/leave`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, room_name: roomName })
      }).catch(()=>{});
      window.__livekitRoomName = null;
    }

    setIsGroupVideo6Active(false);
    setGroupVideo6Participants([]);
    setGroupVideo6Messages([]);
    setGroupVideo6RoomName(null);
    setIsConversationActive(false);
    addNotification({ type: 'info', icon: '🔊', title: 'Выход', message: 'Вы вернулись в голосовой режим' });
  }, [userId, addNotification]);

  const handleGroupVideo6Next = useCallback(async () => {
    setIsSearchingNextRoom(true);
    cancelVideoSearchRef.current = false;
    switchingVideo6Ref.current = true;

    const roomName = window.__livekitRoomName;
    if (livekitRoomRef.current) {
      try { livekitRoomRef.current.disconnect(); } catch(e){}
      livekitRoomRef.current = null;
    }
    window.__livekitRoom = null;
    if (roomName) {
      try {
        await fetch(`${CONFIG.BACKEND_URL}/api/livekit/leave`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, room_name: roomName })
        });
      } catch(e) {}
      window.__livekitRoomName = null;
    }

    setGroupVideo6Participants(prev => prev.filter(p => p.id === userId || p.id === 'local'));

    if (nextRoomTimerRef.current) clearTimeout(nextRoomTimerRef.current);
    nextRoomTimerRef.current = setTimeout(() => {
      nextRoomTimerRef.current = null;
      quickStartVideoGroup();
      setTimeout(() => {
        switchingVideo6Ref.current = false;
      }, 500);
    }, 100);
  }, [userId, quickStartVideoGroup]);

  const handleCancelSearchRoom = useCallback(() => {
    setIsSearchingNextRoom(false);
    cancelVideoSearchRef.current = true;
    switchingVideo6Ref.current = true;

    if (nextRoomTimerRef.current) {
      clearTimeout(nextRoomTimerRef.current);
      nextRoomTimerRef.current = null;
    }

    if (livekitRoomRef.current) {
      try { livekitRoomRef.current.disconnect(); } catch(e){}
      livekitRoomRef.current = null;
    }
    window.__livekitRoom = null;

    setGroupVideo6Participants(prev => prev.filter(p => p.id === userId || p.id === 'local'));

    setTimeout(() => {
      switchingVideo6Ref.current = false;
    }, 500);

    sendMessage({ type: 'stop_search' });
  }, [sendMessage, userId]);

  const handleGroupVideo6SendMessage = useCallback((text, nickname) => {
    const newMessage = { id: Date.now(), text, senderId: userId, senderName: userId.slice(0,6), senderNickname: nickname || 'User', isOwn: true, timestamp: new Date().toISOString() };
    setGroupVideo6Messages(prev => {
      const updated = [...prev, newMessage];
      return updated.length > 100 ? updated.slice(-100) : updated;
    });
    sendMessage({ type: 'group_video_message', message: text, room_id: roomId || window.__livekitRoomName, user_id: userId, user_name: userId.slice(0,6), nickname: nickname || 'User' });
  }, [userId, roomId, sendMessage]);

  const handleGroupVideo6Report = useCallback((participantId, participantName) => {
    reportContextRef.current = { partnerId: participantId, partnerName: participantName, roomId: window.__livekitRoomName || roomId };
    setShowReportModal(true);
  }, [roomId]);

  const toggleVideo6Mic = useCallback(() => {
    const newMuted = !groupVideo6MicMuted;
    if (window.__livekitRoom) {
      try { window.__livekitRoom.localParticipant.setMicrophoneEnabled(!newMuted); } catch(e){}
    }
    setGroupVideo6MicMuted(newMuted);
  }, [groupVideo6MicMuted]);

  // ==================== WEBSOCKET HANDLERS ====================
  const handleVideoWebSocketMessage = useCallback((message) => {
    if (message.type === 'video_message' && (isGroupVideo2Active || isGroupVideo6Active)) {
      const newMessage = { id: Date.now(), text: message.message, senderId: message.user_id, senderName: message.user_name || 'User', senderNickname: message.nickname || 'User', isOwn: message.user_id === userId, timestamp: message.timestamp || new Date().toISOString() };
      if (isGroupVideo2Active) setGroupVideo2Messages(prev => {
        const updated = [...prev, newMessage];
        return updated.length > 100 ? updated.slice(-100) : updated;
      });
      if (isGroupVideo6Active) setGroupVideo6Messages(prev => {
        const updated = [...prev, newMessage];
        return updated.length > 100 ? updated.slice(-100) : updated;
      });
    }
    if (message.type === 'group_video_message' && isGroupVideo6Active) {
      const newMessage = { id: Date.now(), text: message.message, senderId: message.user_id, senderName: message.user_name || 'User', senderNickname: message.nickname || 'User', isOwn: message.user_id === userId, timestamp: message.timestamp || new Date().toISOString() };
      setGroupVideo6Messages(prev => {
        const updated = [...prev, newMessage];
        return updated.length > 100 ? updated.slice(-100) : updated;
      });
    }
  }, [isGroupVideo2Active, isGroupVideo6Active, userId]);

  const connectWebSocket = useCallback(async () => {
    if (wsManagerRef.current && wsManagerRef.current.isConnected()) return true;
    if (wsManagerRef.current && wsManagerRef.current.isConnectingState()) return false;
    if (wsManagerRef.current) { try { wsManagerRef.current.disconnect(); } catch(e){} wsManagerRef.current = null; }
    try {
      const fp = await getBrowserFingerprint();
      wsManagerRef.current = new WebSocketManager(userId, browserId, fp, (status) => {
        setConnectionStatus(status);
        if (status === 'connected') setStatusMessage('Подключен к чату');
        else if (status === 'disconnected') setStatusMessage('Переподключение...');
        else if (status === 'blocked') setStatusMessage('Доступ заблокирован');
      });
      wsManagerRef.current.addMessageHandler((m) => { if (messageHandlerRef.current) messageHandlerRef.current(m); });
      const ok = await wsManagerRef.current.connect();
      if (ok) setIsWsInitialized(true);
      return ok;
    } catch(e) { return false; }
  }, [getBrowserFingerprint, userId, browserId]);

  const connectModerationWebSocket = useCallback(() => {
    if (isModerationWebSocketConnecting) return;
    if (moderationWsRef.current && moderationWsRef.current.readyState === WebSocket.OPEN) return;
    isModerationWebSocketConnecting = true;
    try {
      moderationWsRef.current = new WebSocket(`${CONFIG.WS_URL}/ws/moderation`);
      moderationWsRef.current.onopen = () => {
        isModerationWebSocketConnecting = false;
        checkModerationStatus();
        if (moderationReconnectTimeoutRef.current) { clearTimeout(moderationReconnectTimeoutRef.current); moderationReconnectTimeoutRef.current = null; }
      };
      moderationWsRef.current.onmessage = (ev) => {
        try { const m = JSON.parse(ev.data); if (m.type === 'ping') { moderationWsRef.current.send(JSON.stringify({ type: 'pong' })); return; } handleModerationMessage(m); } catch(e){}
      };
      moderationWsRef.current.onclose = () => {
        isModerationWebSocketConnecting = false;
        if (!moderationReconnectTimeoutRef.current) moderationReconnectTimeoutRef.current = setTimeout(() => connectModerationWebSocket(), 5000);
      };
      moderationWsRef.current.onerror = () => { isModerationWebSocketConnecting = false; };
    } catch(e) { isModerationWebSocketConnecting = false; }
  }, [checkModerationStatus]);

  const handleModerationMessage = useCallback((message) => {
    switch (message.type) {
      case 'user_warned':
        if (message.user_id === userId) {
          const wsWarnId = message.warning_id || message.id || `ws_warn_${message.reason || ''}_${message.timestamp || Date.now()}`;
          const sessionKey = `ws_warn_${wsWarnId}`;
          if (!wasShownThisSession(sessionKey)) {
            markAsShown(sessionKey);
            playModerationSound('warn');
            addNotification({
              type: 'warn',
              icon: '⚠️',
              title: 'Предупреждение',
              message: translateReason(message.reason),
              timeout: 10000
            });
          }
        }
        break;

      case 'user_muted':
        if (message.user_id === userId) {
          const wsMuteId = message.mute_id || message.id || `ws_mute_${message.reason || ''}_${message.timestamp || Date.now()}`;
          const sessionKey = `ws_mute_${wsMuteId}`;
          if (!wasShownThisSession(sessionKey)) {
            markAsShown(sessionKey);
            playModerationSound('mute');
            addNotification({
              type: 'mute',
              icon: '🔇',
              title: 'Вы заглушены',
              message: translateReason(message.reason),
              timeout: 10000
            });
            setModerationStatus(prev => ({ ...prev, is_muted: true }));
            forceMuteMicrophone(true);
          }
        }
        break;

      case 'user_unmuted':
        if (message.user_id === userId) {
          if (!wasShownThisSession('ws_unmute_session')) {
            markAsShown('ws_unmute_session');
            playModerationSound('unmute');
            addNotification({
              type: 'unmute',
              icon: '🔊',
              title: 'Заглушка снята',
              timeout: 5000
            });
            setModerationStatus(prev => ({ ...prev, is_muted: false }));
            forceMuteMicrophone(false);
          }
        }
        break;

      case 'user_kicked':
        if (message.user_id === userId) {
          if (!wasShownThisSession('ws_kick_session')) {
            markAsShown('ws_kick_session');
            playModerationSound('kick');
            showBanOverlay('kick', translateReason(message.reason), null, null);
            if (userStatus === 'connected') stopConnection();
            if (userStatus === 'searching') stopSearch();
          }
        }
        break;

      case 'user_banned':
        if (message.user_id === userId) {
          if (!wasShownThisSession('ws_ban_session')) {
            markAsShown('ws_ban_session');
            playModerationSound('ban');
            showBanOverlay(
              'ban',
              translateReason(message.reason),
              message.duration ? `${message.duration} ч` : '24 ч',
              null
            );
            setModerationStatus(prev => ({ ...prev, is_banned: true, ban_reason: message.reason }));
          }
        }
        break;

      case 'user_unbanned':
        if (message.user_id === userId) {
          if (!wasShownThisSession('ws_unban_session')) {
            markAsShown('ws_unban_session');
            playModerationSound('unban');
            addNotification({
              type: 'unban',
              icon: '✅',
              title: 'Блокировка снята',
              timeout: 5000
            });
            setModerationStatus(prev => ({ ...prev, is_banned: false, is_banned_extended: false }));
            setBanOverlay({ isVisible: false, type: null, reason: null, duration: null, banType: null });
            if (wsManagerRef.current) {
              wsManagerRef.current.resetBlock();
              wsInitRef.current = false;
              setIsWsInitialized(false);
            }
          }
        }
        break;

      case 'extended_ban':
        if (message.user_id === userId) {
          if (!wasShownThisSession('ws_extended_ban_session')) {
            markAsShown('ws_extended_ban_session');
            playModerationSound('ban');
            showBanOverlay('extended_ban', translateReason(message.reason), null, message.ban_type);
          }
        }
        break;

      default:
        break;
    }
  }, [userId, userStatus, addNotification, forceMuteMicrophone, stopConnection, stopSearch, playModerationSound, showBanOverlay]);

  const handleWebSocketMessage = useCallback(async (message) => {
    handleVideoWebSocketMessage(message);
    switch (message.type) {
      case 'pong': break;
      case 'blocked': playModerationSound('ban'); showBanOverlay('ban', message.reason, null, message.ban_type); setModerationStatus(prev => ({ ...prev, is_banned: true, is_banned_extended: true, ban_type: message.ban_type, ban_reason: message.reason })); if (wsManagerRef.current) { wsManagerRef.current.blockedByBan = true; wsManagerRef.current.isShuttingDown = true; } break;
      case 'connected': setConnectionStatus('connected'); setStatusMessage('Подключен'); break;
      case 'search_started': setUserStatus('searching'); setStatusMessage('Поиск...'); break;
      case 'search_error': setUserStatus('idle'); setStatusMessage('Ошибка поиска'); break;
      case 'no_partners_available': setUserStatus('idle'); setStatusMessage('Нет собеседников'); addNotification({ type: 'warning', icon: '👥', title: 'Никого нет', message: 'Попробуйте позже' }); break;
      case 'search_stopped':
        setUserStatus('idle');
        setStatusMessage('Поиск остановлен');
        setIsConversationActive(false);
        setPartnerId(null);
        setRoomId(null);
        break;
      case 'match_found': setUserStatus('connected'); setPartnerId(message.partner_id); setRoomId(message.room_id); setStatusMessage('Контакт установлен!'); playSound(); setTimeout(() => initializeVoiceChat(message.room_id, message.partner_id), 100); break;
      case 'partner_disconnected':
        setUserStatus('idle');
        setPartnerId(null);
        setRoomId(null);
        setIsConversationActive(false);
        setStatusMessage('Собеседник отключился');
        cleanupVoiceChat();
        break;
      case 'frontend_update_available':
        break;
      case 'webrtc_signal': handleWebRTCSignal(message.signal); break;
      case 'force_disconnect': handlePartnerDisconnected('Соединение разорвано'); addNotification({ type: 'admin', icon: '👨‍💼', title: 'Администратор', message: 'Вы отключены' }); break;
      case 'search_timeout': setUserStatus('idle'); setStatusMessage('Не удалось найти'); break;
      case 'system_message': { const key = `sys_${message.id || 'welcome'}`; if (!sessionStorage.getItem(key)) { setSystemMessage(message.message); setShowSystemMessage(true); sessionStorage.setItem(key, 'true'); } break; }
      case 'global_video_mode_changed': addNotification({ type: 'system', icon: message.enabled ? '🎥' : '🚫', title: 'Видео-режим', message: message.message }); break;
      case 'moderation_action': addNotification({ type: 'moderator', icon: '🛡️', title: 'Модерация', message: translateReason(message.message) }); break;
      case 'admin_message': addNotification({ type: 'admin', icon: '👨‍💼', title: message.title || 'Администратор', message: translateReason(message.text) }); break;
      case 'muffle_status': setMuffleEnabled(message.enabled); if (message.enabled) localStorage.setItem('muffle_enabled', 'true'); else localStorage.removeItem('muffle_enabled'); break;
      case 'maintenance_mode': setMaintenanceMode(message.enabled); if (message.enabled) localStorage.setItem('maintenance_mode', 'true'); else localStorage.removeItem('maintenance_mode'); break;
      case 'force_mute': addNotification({ type: 'mute', icon: '🔇', title: 'Заглушены', message: translateReason(message.reason), timeout: 10000 }); setModerationStatus(prev => ({ ...prev, is_muted: true })); forceMuteMicrophone(true); break;
      case 'report_duplicate': addNotification({ type: 'warning', icon: '⚠️', title: 'Уже отправлено', message: 'Вы уже жаловались на этого пользователя' }); break;
      case 'warning':
        playModerationSound('warn');
        addNotification({
          type: 'warning',
          icon: message.icon || '⚠️',
          title: message.title || 'Предупреждение',
          message: message.message || '',
          submessage: message.submessage || '',
          timeout: message.timeout || 8000,
        });
        break;

      case 'report_accepted':
        addNotification({
          type: 'success',
          icon: message.icon || '✅',
          title: message.title || 'Жалоба принята',
          message: message.message || '',
          submessage: message.submessage || '',
          timeout: message.timeout || 6000,
        });
        break;

      case 'report_resolved':
        playModerationSound('success');
        addNotification({
          type: 'success',
          icon: message.icon || '✅',
          title: message.title || 'Ваша жалоба обработана',
          message: message.message || '',
          submessage: message.submessage || '',
          timeout: message.timeout || 12000,
        });
        break;

      case 'moderation_action':
        playModerationSound('ban');
        addNotification({
          type: 'moderator',
          icon: message.icon || '🛡️',
          title: message.title || 'Действие модератора',
          message: message.message || '',
          submessage: message.submessage || '',
          timeout: message.timeout || 12000,
        });
        break;
      default: handleModerationMessage(message);
    }
  }, [userId, addNotification, handlePartnerDisconnected, handleWebRTCSignal, playSound, forceMuteMicrophone, handleVideoWebSocketMessage, initializeVoiceChat, playModerationSound, showBanOverlay, handleModerationMessage]);

  useEffect(() => { messageHandlerRef.current = handleWebSocketMessage; }, [handleWebSocketMessage]);

  useEffect(() => {
    const initApp = async () => {
      if (initialCheckComplete) return;
      await checkServerStatus();
      try {
        const isM = await checkMaintenanceMode();
        if (isM) { setMaintenanceMode(true); return; }
        setInitialCheckComplete(true);
      } catch(e) { setInitialCheckComplete(true); }
    };
    initApp();
  }, [initialCheckComplete, checkServerStatus, checkMaintenanceMode]);

  useEffect(() => {
    if (!initialCheckComplete || maintenanceMode || muffleEnabled) return;
    if (micCheckedRef.current) return;

    (async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
        s.getAudioTracks().forEach(t => { t.enabled = false; });
        prewarmedStreamRef.current = s;
        micCheckedRef.current = true;
        micAvailableRef.current = true;
        log('✅ [MIC] prewarmed at startup');
      } catch (e) {
        micCheckedRef.current = true;
        micAvailableRef.current = false;
        warn('❌ [MIC] prewarm failed:', e);
      }
    })();
  }, [initialCheckComplete, maintenanceMode, muffleEnabled]);

  useEffect(() => {
    if (maintenanceMode || muffleEnabled) {
      if (wsManagerRef.current) { try { wsManagerRef.current.disconnect(); } catch(e){} wsManagerRef.current = null; }
      wsInitRef.current = false; setIsWsInitialized(false);
      if (maintenanceCheckRef.current) { clearInterval(maintenanceCheckRef.current); maintenanceCheckRef.current = null; }
      return;
    }
    if (wsInitRef.current || !initialCheckComplete || isWsInitialized) return;
    wsInitRef.current = true;
    const init = async () => {
      const ok = await connectWebSocket();
      if (ok) setIsWsInitialized(true); else wsInitRef.current = false;
      setTimeout(() => connectModerationWebSocket(), 500);
    };
    init();
    maintenanceCheckRef.current = setInterval(async () => {
      if (isVideoModeActiveRef.current) return;
      const isM = await checkMaintenanceMode();
      checkMuffleStatus();
      if (!isM && (maintenanceMode || muffleEnabled)) { wsInitRef.current = false; setIsWsInitialized(false); }
    }, 15000);
    return () => { if (maintenanceCheckRef.current) clearInterval(maintenanceCheckRef.current); };
  }, [maintenanceMode, muffleEnabled, initialCheckComplete, isWsInitialized, connectWebSocket, connectModerationWebSocket, checkMaintenanceMode, checkMuffleStatus]);

  useEffect(() => {
    if (userStatus === 'connected' || isGroupVideo2Active || isGroupVideo6Active) {
      if (moderationCheckIntervalRef.current) {
        clearInterval(moderationCheckIntervalRef.current);
        moderationCheckIntervalRef.current = null;
      }
      return;
    }
    checkModerationStatus();
    moderationCheckIntervalRef.current = setInterval(() => checkModerationStatus(), CONFIG.MODERATION_CHECK_INTERVAL);
    return () => { if (moderationCheckIntervalRef.current) clearInterval(moderationCheckIntervalRef.current); };
  }, [checkModerationStatus, userStatus, isGroupVideo2Active, isGroupVideo6Active]);

  useEffect(() => { if (moderationStatus.is_muted) forceMuteMicrophone(true); }, [moderationStatus.is_muted, forceMuteMicrophone]);

  useEffect(() => {
    if (remoteAudioRef.current && remoteStream) {
      remoteAudioRef.current.srcObject = remoteStream;
      remoteAudioRef.current.play().catch(()=>{});
    }
    if (localAudioRef.current && localStream) {
      localAudioRef.current.srcObject = localStream;
    }
  }, [remoteStream, localStream]);

  useEffect(() => {
    if (remoteAudioRef.current) {
      remoteAudioRef.current.volume = volume;
    }
  }, [volume]);

  useEffect(() => {
    if (maintenanceMode || muffleEnabled || !initialCheckComplete) return;
    const loadAds = async () => { try { const r = await fetch(`${CONFIG.BACKEND_URL}/api/ads/active`, { cache: 'no-store' }); if (r.ok) { const d = await r.json(); setActiveAds({ audio: d.audio || [], video: d.video || [] }); } } catch(e){} };
    const checkPrem = async () => { try { const r = await fetch(`${CONFIG.BACKEND_URL}/api/user/${userId}/is-premium`, { cache: 'no-store' }); if (r.ok) { const d = await r.json(); setIsPremium(d.is_premium || false); } } catch(e){} };
    loadAds(); checkPrem();
    adCheckIntervalRef.current = setInterval(loadAds, 120000);
    return () => { if (adCheckIntervalRef.current) clearInterval(adCheckIntervalRef.current); };
  }, [userId, maintenanceMode, muffleEnabled, initialCheckComplete]);

  useEffect(() => {
    if (isPremium || maintenanceMode || muffleEnabled || !initialCheckComplete || currentAd) return;
    if (isGroupVideo2Active || isGroupVideo6Active) return;
    const AUDIO_AD_INTERVAL = 5 * 60 * 1000;
    const VIDEO_AD_INTERVAL = 10 * 60 * 1000;
    const MIN_INITIAL_DELAY = 60 * 1000;
    const now = Date.now();
    const showAd = async (ad, type) => {
      try { const r = await fetch(`${CONFIG.BACKEND_URL}/api/ads/${type}/${ad.id}/data`, { cache: 'no-store' }); if (r.ok) { const full = await r.json(); setCurrentAd({ ad: { ...ad, file_data: full.file_data }, type }); } } catch(e){}
    };
    if (activeAds.audio.length > 0 && now - lastAudioAdTime > Math.max(AUDIO_AD_INTERVAL, MIN_INITIAL_DELAY)) {
      showAd(activeAds.audio[Math.floor(Math.random() * activeAds.audio.length)], 'audio');
      setLastAudioAdTime(now); return;
    }
    if (activeAds.video.length > 0 && now - lastVideoAdTime > Math.max(VIDEO_AD_INTERVAL, MIN_INITIAL_DELAY)) {
      showAd(activeAds.video[Math.floor(Math.random() * activeAds.video.length)], 'video');
      setLastVideoAdTime(now); return;
    }
    adPlayIntervalRef.current = setInterval(() => {
      const n = Date.now();
      if (isPremium || maintenanceMode || muffleEnabled || isGroupVideo2Active || isGroupVideo6Active) return;
      if (activeAds.audio.length > 0 && n - lastAudioAdTime > Math.max(AUDIO_AD_INTERVAL, MIN_INITIAL_DELAY)) {
        showAd(activeAds.audio[Math.floor(Math.random() * activeAds.audio.length)], 'audio'); setLastAudioAdTime(n);
      } else if (activeAds.video.length > 0 && n - lastVideoAdTime > Math.max(VIDEO_AD_INTERVAL, MIN_INITIAL_DELAY)) {
        showAd(activeAds.video[Math.floor(Math.random() * activeAds.video.length)], 'video'); setLastVideoAdTime(n);
      }
    }, 60000);
    return () => { if (adPlayIntervalRef.current) clearInterval(adPlayIntervalRef.current); };
  }, [activeAds, isPremium, maintenanceMode, muffleEnabled, initialCheckComplete, isGroupVideo2Active, isGroupVideo6Active, lastAudioAdTime, lastVideoAdTime, currentAd]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!welcomeAudioRef.current) {
      welcomeAudioRef.current = new Audio(CONFIG.WELCOME_SOUND);
      welcomeAudioRef.current.preload = 'auto';
      welcomeAudioRef.current.volume = 0.5;
    }
  }, []);

  useEffect(() => {
    const a = welcomeAudioRef.current;
    if (!a) return;
    const handleEnded = () => setIsWelcomePlaying(false);
    a.addEventListener('ended', handleEnded);
    return () => a.removeEventListener('ended', handleEnded);
  }, []);

  useEffect(() => {
    if (!initialCheckComplete) return;
    if (maintenanceMode || muffleEnabled) return;
    if (welcomePlayedRef.current) return;

    const welcomeShown = sessionStorage.getItem('welcome_audio_played');
    if (welcomeShown) { welcomePlayedRef.current = true; return; }

    const a = welcomeAudioRef.current;
    if (!a) return;

    const tryPlay = async () => {
      try {
        setIsWelcomePlaying(true);
        await a.play();
        welcomePlayedRef.current = true;
        sessionStorage.setItem('welcome_audio_played', 'true');
        console.log('🔊 [WELCOME] played');
        setTimeout(() => setIsWelcomePlaying(false), 15000);
      } catch (e) {
        console.log('🔊 [WELCOME] blocked:', e.name);
        setIsWelcomePlaying(false);
        if (e.name === 'NotAllowedError') {
          const events = ['click', 'touchstart', 'keydown', 'pointerdown', 'mousedown'];
          const unlock = () => {
            setIsWelcomePlaying(true);
            a.play().then(() => {
              welcomePlayedRef.current = true;
              sessionStorage.setItem('welcome_audio_played', 'true');
              console.log('🔊 [WELCOME] played after unlock');
              setTimeout(() => setIsWelcomePlaying(false), 15000);
            }).catch(() => {
              setIsWelcomePlaying(false);
            });
            events.forEach(evt => document.removeEventListener(evt, unlock));
          };
          events.forEach(evt =>
            document.addEventListener(evt, unlock, { once: true, passive: true })
          );
        }
      }
    };

    tryPlay();
  }, [initialCheckComplete, maintenanceMode, muffleEnabled]);

  useEffect(() => {
    if (maintenanceMode || muffleEnabled) return;

    let initialVersion = sessionStorage.getItem('cosmic_loaded_version');

    const checkUpdates = async () => {
      try {
        const r = await fetch(`${CONFIG.BACKEND_URL}/api/frontend-version`, { cache: 'no-store' });
        if (!r.ok) return;
        const d = await r.json();

        if (!initialVersion) {
          sessionStorage.setItem('cosmic_loaded_version', d.version);
          initialVersion = d.version;
          console.log('🔄 [UPDATES] Initial version set:', d.version);
          return;
        }

        if (d.version && d.version !== initialVersion) {
          console.log('🔄 [UPDATES] New version:', d.version, '(was:', initialVersion, ')');
          addNotification({
            type: 'info',
            icon: '🔄',
            title: 'Доступно обновление',
            message: 'Обновите страницу, чтобы получить новую версию',
            submessage: `Версия: ${d.version.slice(0, 20)}`,
            timeout: 60000
          });

          sessionStorage.setItem('cosmic_loaded_version', d.version);
          initialVersion = d.version;
        }
      } catch (e) {}
    };

    checkUpdates();
    const interval = setInterval(checkUpdates, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [maintenanceMode, muffleEnabled, addNotification]);

  const handleOpenReportModal = useCallback(() => {
    reportContextRef.current = { partnerId, partnerName: 'Собеседник', roomId };
    setShowReportModal(true);
  }, [partnerId, roomId]);

  if (maintenanceMode || muffleEnabled) {
    return <MaintenancePage isMobile={isMobile} systemMessage={systemMessage} muffleEnabled={muffleEnabled} onRetry={Object.assign(async () => {
      try {
        const r = await fetch(`${CONFIG.BACKEND_URL}/api/maintenance-status`, { cache: 'no-store' });
        if (r.ok) {
          const d = await r.json();
          if (!d.maintenance_mode && !d.muffle_enabled) return { available: true };
          return { available: false };
        }
        return { available: false };
      } catch(e) { return { available: false }; }
    }, {
      release: () => {
        setMaintenanceMode(false); setMuffleEnabled(false); setInitialCheckComplete(false);
        wsInitRef.current = false; setIsWsInitialized(false);
        localStorage.removeItem('maintenance_mode'); localStorage.removeItem('muffle_enabled');
        setTimeout(async () => { await connectWebSocket(); setTimeout(() => connectModerationWebSocket(), 500); }, 200);
      }
    })} />;
  }

  return (
    <div className="app cosmic-app" style={{
      minHeight: '100vh',
      background: currentPage === 'chat' ? '#000' : 'radial-gradient(ellipse at 30% 40%, #0a0a2a 0%, #030318 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'flex-start',
      padding: isMobile ? '0 8px 8px' : '20px',
      paddingTop: isMobile ? 'env(safe-area-inset-top)' : '20px',
      position: 'relative',
      overflowX: 'hidden',
      width: '100%',
      boxSizing: 'border-box'
    }}>
      {false && <ConsoleGuard
        enabled={false}
        showWarning={false}
        blockHotkeys={false}
        blockContextMenu={false}
        detectDevTools={false}
        blockChromeConsole={false}
        antiDebugger={false}
        cleanGlobals={false}
      />}

      {process.env.NODE_ENV === 'production' && (
        <MetrikaCounter
          id={YM_COUNTER_ID}
          options={{
            webvisor: true,
            clickmap: true,
            trackLinks: true,
            accurateTrackBounce: true,
            trackHash: false,
            ssr: true,
            ecommerce: false,
            childIframe: false,
          }}
        />
      )}

      {currentPage !== 'chat' && (
        <>
          <FloatingRocket />
          <ScrollToTopButton isMobile={isMobile} />
        </>
      )}

      {currentPage === 'chat' && !isGroupVideo2Active && !isGroupVideo6Active && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 1, pointerEvents: 'none', overflow: 'hidden',
          background: 'radial-gradient(ellipse at 30% 20%, #1a0a3a 0%, #0a0520 50%, #05030f 100%)',
        }}>
          <video
            ref={spaceBgVideoRef}
            autoPlay
            loop
            muted
            playsInline
            preload="auto"
            poster="/space-bg-poster.jpg"
            disablePictureInPicture
            controlsList="nodownload nofullscreen noremoteplayback"
            style={{
              position: 'absolute', top: 0, left: 0,
              width: '100%', height: '100%',
              objectFit: 'cover',
              transform: 'scale(1.02)',
              transformOrigin: 'center center',
            }}
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
          >
            <source src="/space-bg.mp4" type="video/mp4" />
          </video>
        </div>
      )}

      {isVideoLoading && (
        <Portal>
          <div style={{
            position: 'fixed', inset: 0,
            background: 'radial-gradient(ellipse at 50% 40%, rgba(20,10,50,0.95) 0%, rgba(0,0,0,0.98) 70%)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            zIndex: 999998,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '24px',
            padding: '20px',
            overflow: 'hidden',
          }}>
            <div style={{
              position: 'absolute', inset: 0, pointerEvents: 'none',
              background: `
                radial-gradient(2px 2px at 15% 20%, rgba(255,255,255,0.7), transparent),
                radial-gradient(1px 1px at 25% 60%, rgba(255,255,255,0.5), transparent),
                radial-gradient(1.5px 1.5px at 80% 30%, rgba(0,255,255,0.6), transparent),
                radial-gradient(1px 1px at 45% 85%, rgba(255,255,255,0.4), transparent),
                radial-gradient(2px 2px at 90% 70%, rgba(255,0,255,0.5), transparent),
                radial-gradient(1px 1px at 60% 15%, rgba(255,255,255,0.6), transparent)
              `,
              animation: 'starDrift 20s linear infinite',
            }} />
            <div style={{
              position: 'relative',
              width: isMobile ? '100px' : '140px',
              height: isMobile ? '140px' : '180px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              animation: 'rocketFloat 2s ease-in-out infinite',
            }}>
              <svg
                width={isMobile ? '80' : '110'}
                height={isMobile ? '80' : '110'}
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                style={{
                  filter: 'drop-shadow(0 0 20px rgba(0,255,255,0.6)) drop-shadow(0 0 40px rgba(255,100,0,0.4))',
                  position: 'relative',
                  zIndex: 2,
                }}
              >
                <path d="M12 2C12 2 8 6 8 12V18L6 20H18L16 18V12C16 6 12 2 12 2Z" fill="url(#loadingRocketBody)" stroke="rgba(0,255,255,0.8)" strokeWidth="0.8" strokeLinejoin="round" />
                <circle cx="12" cy="10" r="2" fill="#00d2ff" opacity="0.9" />
                <circle cx="12" cy="10" r="1" fill="#fff" opacity="0.9" />
                <path d="M8 16L5 20V22L8 19V16Z" fill="url(#loadingRocketFin)" stroke="rgba(0,255,255,0.5)" strokeWidth="0.4" />
                <path d="M16 16L19 20V22L16 19V16Z" fill="url(#loadingRocketFin)" stroke="rgba(0,255,255,0.5)" strokeWidth="0.4" />
                <path d="M10 20H14L13 22H11L10 20Z" fill="#333" stroke="rgba(0,255,255,0.4)" strokeWidth="0.4" />
                <defs>
                  <linearGradient id="loadingRocketBody" x1="12" y1="2" x2="12" y2="20" gradientUnits="userSpaceOnUse">
                    <stop offset="0" stopColor="#e0e8ff" />
                    <stop offset="0.5" stopColor="#a0b8ff" />
                    <stop offset="1" stopColor="#7088cc" />
                  </linearGradient>
                  <linearGradient id="loadingRocketFin" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0" stopColor="#ff6b6b" />
                    <stop offset="1" stopColor="#c0392b" />
                  </linearGradient>
                </defs>
              </svg>

              <div style={{
                position: 'absolute',
                top: isMobile ? '68%' : '66%',
                left: '50%',
                transform: 'translateX(-50%)',
                width: isMobile ? '30px' : '40px',
                height: isMobile ? '50px' : '70px',
                zIndex: 1,
                pointerEvents: 'none',
              }}>
                <div style={{
                  position: 'absolute', left: '50%', top: 0, transform: 'translateX(-50%)',
                  width: '100%', height: '100%',
                  background: 'radial-gradient(ellipse at 50% 0%, rgba(255,100,0,0.95) 0%, rgba(255,60,0,0.6) 40%, transparent 75%)',
                  borderRadius: '50% 50% 40% 40%', filter: 'blur(2px)',
                  animation: 'fireFlicker1 0.15s ease-in-out infinite alternate',
                  transformOrigin: 'top center',
                }} />
                <div style={{
                  position: 'absolute', left: '50%', top: '10%', transform: 'translateX(-50%)',
                  width: '70%', height: '80%',
                  background: 'radial-gradient(ellipse at 50% 0%, rgba(255,180,0,0.95) 0%, rgba(255,120,0,0.7) 50%, transparent 80%)',
                  borderRadius: '50% 50% 40% 40%', filter: 'blur(1px)',
                  animation: 'fireFlicker2 0.12s ease-in-out infinite alternate',
                  transformOrigin: 'top center',
                }} />
                <div style={{
                  position: 'absolute', left: '50%', top: '20%', transform: 'translateX(-50%)',
                  width: '40%', height: '60%',
                  background: 'radial-gradient(ellipse at 50% 0%, rgba(255,255,150,1) 0%, rgba(255,220,0,0.8) 50%, transparent 80%)',
                  borderRadius: '50% 50% 40% 40%', filter: 'blur(0.5px)',
                  animation: 'fireFlicker3 0.1s ease-in-out infinite alternate',
                  transformOrigin: 'top center',
                }} />
                <div style={{
                  position: 'absolute', left: '50%', top: '60%', width: '6px', height: '6px',
                  borderRadius: '50%', background: '#ffcc00', boxShadow: '0 0 8px #ffcc00',
                  animation: 'spark1 0.8s linear infinite',
                }} />
                <div style={{
                  position: 'absolute', left: '35%', top: '60%', width: '4px', height: '4px',
                  borderRadius: '50%', background: '#ff8800', boxShadow: '0 0 6px #ff8800',
                  animation: 'spark2 1s linear infinite', animationDelay: '0.3s',
                }} />
                <div style={{
                  position: 'absolute', left: '65%', top: '60%', width: '5px', height: '5px',
                  borderRadius: '50%', background: '#ffaa00', boxShadow: '0 0 7px #ffaa00',
                  animation: 'spark1 0.9s linear infinite', animationDelay: '0.5s',
                }} />
              </div>

              <div style={{
                position: 'absolute',
                top: isMobile ? '60%' : '58%',
                left: '50%',
                transform: 'translateX(-50%)',
                width: isMobile ? '70px' : '90px',
                height: isMobile ? '40px' : '50px',
                background: 'radial-gradient(ellipse, rgba(255,120,0,0.6) 0%, rgba(255,60,0,0.3) 40%, transparent 75%)',
                filter: 'blur(12px)', pointerEvents: 'none',
                animation: 'fireFlicker1 0.25s ease-in-out infinite alternate',
                zIndex: 0,
              }} />
            </div>

            <div style={{
              fontSize: isMobile ? '1.1rem' : '1.4rem',
              fontWeight: '800',
              background: 'linear-gradient(135deg, #00ffff, #ff00ff)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              letterSpacing: '0.05em',
              textAlign: 'center',
              position: 'relative',
              zIndex: 2,
            }}>
              🚀 Подключение к видео...
            </div>

            <div style={{
              fontSize: isMobile ? '12px' : '13px',
              color: '#a0e7ff',
              textAlign: 'center',
              maxWidth: '400px',
              lineHeight: 1.6,
              padding: '0 20px',
              position: 'relative',
              zIndex: 2,
            }}>
              Разрешите доступ к камере и микрофону,<br />
              если браузер запросит
            </div>

            <div style={{ display: 'flex', gap: '8px', marginTop: '8px', position: 'relative', zIndex: 2 }}>
              {[0, 1, 2].map(i => (
                <div
                  key={i}
                  style={{
                    width: '10px', height: '10px', borderRadius: '50%',
                    background: '#00ffff', boxShadow: '0 0 12px #00ffff',
                    animation: 'videoPulse 1.2s ease-in-out infinite',
                    animationDelay: `${i * 0.2}s`,
                  }}
                />
              ))}
            </div>

            <style>{`
              @keyframes rocketFloat {
                0%, 100% { transform: translateY(0) rotate(-2deg); }
                50% { transform: translateY(-12px) rotate(2deg); }
              }
              @keyframes fireFlicker1 {
                0% { transform: translateX(-50%) scaleY(1) scaleX(1); opacity: 0.95; }
                100% { transform: translateX(-50%) scaleY(1.15) scaleX(0.9); opacity: 0.85; }
              }
              @keyframes fireFlicker2 {
                0% { transform: translateX(-50%) scaleY(1) scaleX(1); opacity: 1; }
                100% { transform: translateX(-50%) scaleY(1.2) scaleX(0.88); opacity: 0.9; }
              }
              @keyframes fireFlicker3 {
                0% { transform: translateX(-50%) scaleY(1) scaleX(1); opacity: 1; }
                100% { transform: translateX(-50%) scaleY(1.25) scaleX(0.85); opacity: 0.95; }
              }
              @keyframes spark1 {
                0% { transform: translate(-50%, 0) scale(1); opacity: 1; }
                100% { transform: translate(-50%, 30px) scale(0.3); opacity: 0; }
              }
              @keyframes spark2 {
                0% { transform: translate(-50%, 0) scale(1); opacity: 1; }
                100% { transform: translate(-50%, 25px) scale(0.2); opacity: 0; }
              }
              @keyframes videoPulse {
                0%, 100% { transform: scale(0.6); opacity: 0.4; }
                50% { transform: scale(1.2); opacity: 1; }
              }
              @keyframes starDrift {
                0% { transform: translate(0, 0); }
                100% { transform: translate(-30px, -20px); }
              }
            `}</style>
          </div>
        </Portal>
      )}

      <Portal>
        <BanOverlay isVisible={banOverlay.isVisible} type={banOverlay.type} reason={banOverlay.reason} duration={banOverlay.duration} banType={banOverlay.banType} onClose={closeBanOverlay} isMobile={isMobile} autoCloseSeconds={banOverlay.type === 'kick' ? CONFIG.KICK_OVERLAY_AUTOCLOSE_SECONDS : CONFIG.BAN_OVERLAY_AUTOCLOSE_SECONDS} />
      </Portal>

      <div style={{
        maxWidth: isMobile ? '100%' : '600px',
        width: '100%',
        background: 'linear-gradient(135deg, rgba(15,15,40,0.85), rgba(8,8,25,0.9))',
        backdropFilter: 'blur(16px)',
        borderRadius: isMobile ? '14px' : '20px',
        border: '1px solid rgba(0,247,255,0.15)',
        marginBottom: isMobile ? '6px' : '16px',
        marginTop: 0,
        zIndex: 10,
        display: 'flex',
        justifyContent: 'center',
        flexShrink: 0
      }}>
        <div style={{
          display: 'flex',
          gap: isMobile ? '2px' : '5px',
          width: '100%',
          justifyContent: 'space-around',
          padding: isMobile ? '2px' : '4px'
        }}>
          {[
            { id: 'chat', icon: '💬', label: 'ЧАТ' },
            { id: 'home', icon: '🏠', label: 'ГЛАВНАЯ' },
            { id: 'contacts', icon: '📞', label: 'КОНТАКТЫ' },
            { id: 'rules', icon: '📜', label: 'ПРАВИЛА' },
            { id: 'faq', icon: '❓', label: 'FAQ' }
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id)}
              className={`nav-btn ${currentPage === item.id ? 'active' : ''}`}
              style={{
                padding: isMobile ? '6px 4px' : '10px 8px',
                fontSize: isMobile ? '9px' : '11px',
                minHeight: isMobile ? '42px' : '52px',
                background: 'transparent',
                border: '1px solid transparent',
                borderRadius: isMobile ? '10px' : '14px',
                color: currentPage === item.id ? '#00f7ff' : '#888',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '4px',
                flex: 1,
                fontWeight: currentPage === item.id ? '700' : '500',
              }}
            >
              <span style={{
                fontSize: isMobile ? '18px' : '22px',
                transition: 'transform 0.25s ease',
                filter: currentPage === item.id ? 'drop-shadow(0 0 8px rgba(0,247,255,0.6))' : 'none',
              }}>{item.icon}</span>
              <span style={{ fontSize: isMobile ? '8px' : '10px' }}>{isMobile ? '' : item.label}</span>
            </button>
          ))}
        </div>
      </div>

      <Portal>
        <div style={{ position: 'fixed', top: isMobile ? '10px' : '20px', right: isMobile ? '10px' : '20px', maxWidth: isMobile ? '90vw' : '350px', zIndex: 99999, display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {notifications.map(n => <Notification key={n.id} notification={n} onClose={() => removeNotification(n.id)} isMobile={isMobile} />)}
        </div>
      </Portal>

      <Portal>
        {showSystemMessage && (
          <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.9)', backdropFilter: 'blur(8px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000 }}>
            <div style={{ width: isMobile ? '90%' : '450px', padding: '35px 30px', background: 'linear-gradient(135deg, rgba(15,25,45,0.98), rgba(10,20,40,0.98))', borderRadius: '24px', border: '1px solid rgba(0,247,255,0.3)', textAlign: 'center' }}>
              <div style={{ fontSize: '72px', marginBottom: '20px' }}>🚀</div>
              <h3 style={{ fontSize: '1.6rem', fontWeight: '800', marginBottom: '12px', background: 'linear-gradient(135deg, #00ffff, #ff00ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Системное сообщение</h3>
              <p style={{ color: '#e2e8f0', marginBottom: '30px' }}>{systemMessage}</p>
              <button onClick={() => setShowSystemMessage(false)} style={{ background: 'linear-gradient(135deg, #00d2ff, #3a7bd5)', border: 'none', borderRadius: '40px', padding: '14px 40px', color: 'white', fontSize: '1.1rem', fontWeight: '700', cursor: 'pointer' }}>✨ Полетели ✨</button>
            </div>
          </div>
        )}
      </Portal>

      <Portal>
        <ReportModal isVisible={showReportModal} onClose={() => setShowReportModal(false)} onSendReport={handleSendReport} isSending={isSendingReport} isMobile={isMobile} />
      </Portal>

      {/* 🆕 Предупреждение о слабом устройстве */}
      <WeakDeviceWarningModal
        isVisible={showWeakDeviceWarning}
        onClose={() => setShowWeakDeviceWarning(false)}
        deviceInfo={deviceInfo}
        isMobile={isMobile}
      />

      {isGroupVideo2Active && (
        <Video2Mode
          isActive={isGroupVideo2Active}
          participants={groupVideo2Participants}
          messages={groupVideo2Messages}
          onSendMessage={handleVideo2SendMessage}
          onReport={handleVideo2Report}
          onExit={exitVideo2Mode}
          onNext={nextVideo2Mode}
          isMobile={isMobile}
          userId={userId}
          onToggleMic={toggleVideo2Mic}
          isMicMuted={groupVideo2MicMuted}
        />
      )}

      {isGroupVideo6Active && (
        <Video6Mode
          isActive={isGroupVideo6Active}
          participants={groupVideo6Participants}
          messages={groupVideo6Messages}
          onSendMessage={handleGroupVideo6SendMessage}
          onReport={handleGroupVideo6Report}
          onExit={exitGroupVideo6Mode}
          onNextRoom={handleGroupVideo6Next}
          onCancelSearch={handleCancelSearchRoom}
          isSearchingNext={isSearchingNextRoom}
          isMobile={isMobile}
          userId={userId}
          onToggleMic={toggleVideo6Mic}
          isMicMuted={groupVideo6MicMuted}
          roomName={groupVideo6RoomName}
        />
      )}

      <div
        className="chat-card"
        style={{
          maxWidth: isMobile ? '100%' : '600px',
          width: '100%',
          background: 'linear-gradient(135deg, rgba(10,15,45,0.7), rgba(5,8,25,0.8))',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          borderRadius: isMobile ? '16px' : '24px',
          border: '1px solid rgba(0,247,255,0.3)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.5), 0 0 60px rgba(0,247,255,0.12), inset 0 1px 0 rgba(255,255,255,0.05)',
          padding: isMobile ? '10px' : '16px',
          zIndex: 5,
          position: 'relative',
          overflow: 'visible',
        }}
      >
        <div style={{
          position: 'absolute',
          top: '-10%',
          left: '-15%',
          width: '60%',
          height: '60%',
          background: 'radial-gradient(circle, rgba(0,255,255,0.15) 0%, transparent 65%)',
          pointerEvents: 'none',
          zIndex: 0,
        }} />
        <div style={{
          position: 'absolute',
          bottom: '-10%',
          right: '-15%',
          width: '60%',
          height: '60%',
          background: 'radial-gradient(circle, rgba(255,0,255,0.12) 0%, transparent 65%)',
          pointerEvents: 'none',
          zIndex: 0,
        }} />

        {currentPage === 'chat' ? (
          <ChatContent
            isMobile={isMobile}
            userId={userId}
            userStatus={userStatus}
            statusMessage={statusMessage}
            connectionStatus={connectionStatus}
            moderationStatus={moderationStatus}
            maintenanceMode={maintenanceMode}
            muffleEnabled={muffleEnabled}
            isConversationActive={isConversationActive}
            isGroupVideo2Active={isGroupVideo2Active}
            isGroupVideo6Active={isGroupVideo6Active}
            volume={volume}
            setVolume={setVolume}
            isMicMuted={isMicMuted}
            setIsMicMuted={setIsMicMuted}
            localStream={localStream}
            selectedCategory={selectedCategory}
            setSelectedCategory={setSelectedCategory}
            showCategories={showCategories}
            setShowCategories={setShowCategories}
            userGender={userGender}
            setUserGender={setUserGender}
            searchGender={searchGender}
            setSearchGender={setSearchGender}
            selectedCountry={selectedCountry}
            setSelectedCountry={setSelectedCountry}
            showCountryDropdown={showCountryDropdown}
            setShowCountryDropdown={setShowCountryDropdown}
            userAge={userAge}
            setUserAge={setUserAge}
            showUserAgeDropdown={showUserAgeDropdown}
            setShowUserAgeDropdown={setShowUserAgeDropdown}
            searchAge={searchAge}
            setSearchAge={setSearchAge}
            showSearchAgeDropdown={showSearchAgeDropdown}
            setShowSearchAgeDropdown={setShowSearchAgeDropdown}
            showGenderDropdown={showGenderDropdown}
            setShowGenderDropdown={setShowGenderDropdown}
            showSearchGenderDropdown={showSearchGenderDropdown}
            setShowSearchGenderDropdown={setShowSearchGenderDropdown}
            getSelectedCategory={getSelectedCategory}
            startSearch={startSearch}
            stopSearch={stopSearch}
            nextPartner={nextPartner}
            stopConnection={stopConnection}
            quickStartVideoSingle={quickStartVideoSingle}
            quickStartVideoGroup={quickStartVideoGroup}
            onOpenReportModal={handleOpenReportModal}
          />
        ) : currentPage === 'home' ? <HomePage onBack={() => setCurrentPage('chat')} isMobile={isMobile} />
        : currentPage === 'contacts' ? <ContactsPage onBack={() => setCurrentPage('chat')} isMobile={isMobile} />
        : currentPage === 'rules' ? <RulesPage onBack={() => setCurrentPage('chat')} isMobile={isMobile} />
        : currentPage === 'faq' ? <FAQPage onBack={() => setCurrentPage('chat')} isMobile={isMobile} />
        : null}
      </div>

      <Portal>
        {currentAd && <AdPlayer ad={currentAd.ad} type={currentAd.type} onComplete={async () => {
          try { await fetch(`${CONFIG.BACKEND_URL}/api/ads/${currentAd.type}/${currentAd.ad.id}/played`, { method: 'POST', headers: { 'Content-Type': 'application/json' } }); } catch(e){}
          setCurrentAd(null);
        }} isMobile={isMobile} />}
      </Portal>

      <audio ref={localAudioRef} muted playsInline style={{ display: 'none' }} />
      <audio ref={remoteAudioRef} autoPlay playsInline style={{ display: 'none' }} />

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes neonPulse { 0%, 100% { opacity: 0.8; } 50% { opacity: 1; } }
        @keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

        @keyframes cardSlideIn {
          from { opacity: 0; transform: translateY(20px) scale(0.98); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }

        @keyframes onlinePulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.4); }
          50% { box-shadow: 0 0 0 8px rgba(0, 255, 136, 0); }
        }

        @keyframes buttonGlow {
          0%, 100% { box-shadow: 0 8px 30px rgba(0,210,255,0.35), 0 0 20px rgba(123,47,247,0.3); }
          50% { box-shadow: 0 8px 40px rgba(0,210,255,0.55), 0 0 30px rgba(123,47,247,0.5); }
        }

        .nav-btn {
          transition: transform 0.25s ease, background 0.25s ease, border-color 0.25s ease, opacity 0.25s ease;
          position: relative;
          transform: translateZ(0);
          backface-visibility: hidden;
          will-change: transform;
        }
        .nav-btn:hover {
          background: rgba(0,247,255,0.08) !important;
          transform: translateY(-1px) translateZ(0);
        }
        .nav-btn.active {
          background: linear-gradient(135deg, rgba(0,247,255,0.22), rgba(255,0,255,0.12)) !important;
          border-color: rgba(0,247,255,0.4) !important;
          color: #00f7ff !important;
          box-shadow: 0 4px 15px rgba(0,247,255,0.2);
        }

        .form-select {
          transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
          transform: translateZ(0);
          backface-visibility: hidden;
        }
        .form-select:hover {
          border-color: rgba(0,247,255,0.5) !important;
          background: rgba(0,247,255,0.05) !important;
          transform: translateY(-1px) translateZ(0);
          box-shadow: 0 4px 12px rgba(0,247,255,0.15);
        }
        .form-select:focus {
          border-color: rgba(0,247,255,0.7) !important;
          box-shadow: 0 0 0 3px rgba(0,247,255,0.15);
          outline: none;
        }

        .search-btn {
          transition: transform 0.3s ease, filter 0.3s ease, opacity 0.3s ease;
          animation: buttonGlow 3s ease-in-out infinite;
          transform: translateZ(0);
          backface-visibility: hidden;
          will-change: transform;
        }
        .search-btn:hover:not(:disabled) {
          transform: translateY(-2px) scale(1.02) translateZ(0);
          filter: brightness(1.15);
        }
        .search-btn:active:not(:disabled) {
          transform: translateY(0) scale(0.99) translateZ(0);
        }
        .search-btn:disabled {
          animation: none;
        }

        .logo-hover {
          transition: transform 0.3s ease;
          transform: translateZ(0);
          backface-visibility: hidden;
        }
        .logo-hover:hover {
          transform: scale(1.05) rotate(-3deg) translateZ(0);
        }

        .online-counter-block {
          animation: onlinePulse 2s ease-in-out infinite;
          transition: transform 0.2s ease;
          transform: translateZ(0);
          backface-visibility: hidden;
        }
        .online-counter-block:hover {
          transform: scale(1.05) translateZ(0);
        }

        .chat-card {
          animation: cardSlideIn 0.5s ease-out;
        }

        .social-link {
          transition: transform 0.25s ease, filter 0.25s ease;
          transform: translateZ(0);
          backface-visibility: hidden;
        }
        .social-link:hover {
          transform: translateY(-3px) translateZ(0);
          filter: brightness(1.3);
        }

        .video-camera .report-btn {
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.2s ease;
        }
        .video-camera:hover .report-btn {
          opacity: 1;
          pointer-events: auto;
        }

        @media (hover: none) and (pointer: coarse) {
          .video-camera .report-btn {
            opacity: 0.95;
            pointer-events: auto;
          }
        }

        @media (max-width: 768px) {
          .video-camera .report-btn {
            top: 4px !important;
            right: 4px !important;
            padding: 5px 8px !important;
            font-size: 12px !important;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        }

        @keyframes copyPop {
          0% { opacity: 0; transform: translateX(-50%) translateY(4px) scale(0.8); }
          50% { transform: translateX(-50%) translateY(-2px) scale(1.05); }
          100% { opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }
        }

        @keyframes scrollTopPulse {
          0%, 100% { transform: scale(1); opacity: 0.6; }
          50% { transform: scale(1.15); opacity: 0; }
        }

        @keyframes fireFlicker1 {
          0% { transform: translateX(-50%) scaleY(1) scaleX(1); opacity: 0.95; }
          100% { transform: translateX(-50%) scaleY(1.15) scaleX(0.9); opacity: 0.85; }
        }
        @keyframes fireFlicker2 {
          0% { transform: translateX(-50%) scaleY(1) scaleX(1); opacity: 1; }
          100% { transform: translateX(-50%) scaleY(1.2) scaleX(0.88); opacity: 0.9; }
        }
        @keyframes fireFlicker3 {
          0% { transform: translateX(-50%) scaleY(1) scaleX(1); opacity: 1; }
          100% { transform: translateX(-50%) scaleY(1.25) scaleX(0.85); opacity: 0.95; }
        }

        @keyframes spark1 {
          0% { transform: translate(-50%, 0) scale(1); opacity: 1; }
          100% { transform: translate(-50%, 20px) scale(0.3); opacity: 0; }
        }
        @keyframes spark2 {
          0% { transform: translate(-50%, 0) scale(1); opacity: 1; }
          100% { transform: translate(-50%, 18px) scale(0.2); opacity: 0; }
        }

        @keyframes maintenanceIconFloat {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-8px); }
        }

        @keyframes maintenanceRocket {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-12px) rotate(-5deg); }
        }

        @keyframes maintenanceGradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        @keyframes maintenanceGlow {
          0% { opacity: 0.6; transform: scale(1); }
          100% { opacity: 1; transform: scale(1.1); }
        }

        @keyframes maintenanceStars {
          0% { transform: translate(0, 0); }
          100% { transform: translate(-20px, -10px); }
        }

        @media (max-width: 768px) {
          .online-counter-block {
            animation: none;
          }
          .chat-card {
            animation: none;
          }
        }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0,247,255,0.3); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0,247,255,0.5); }
      `}</style>
    </div>
  );
}

export default App;