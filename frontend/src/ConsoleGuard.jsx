/*
============================================================
ConsoleGuard — защита от выполнения команд через консоль
⚠️ ВАЖНО: работает ТОЛЬКО в production-сборке!
На localhost (npm start) защита отключена, чтобы не ломать React DevTools
============================================================
*/

import { useEffect } from 'react';

// Определяем, production ли это
const IS_PRODUCTION = 
  process.env.NODE_ENV === 'production' &&
  typeof window !== 'undefined' &&
  !window.location.hostname.includes('localhost') &&
  !window.location.hostname.includes('127.0.0.1') &&
  !window.location.hostname.includes('192.168.');

const ConsoleGuard = ({ 
  enabled = true,
  showWarning = true,
  blockHotkeys = true,
  blockContextMenu = false,
  detectDevTools = true,
  blockChromeConsole = true,
  onDevToolsDetected = null
}) => {
  useEffect(() => {
    // ⚠️ НЕ РАБОТАЕТ в dev-режиме — иначе сломает React
    if (!enabled || !IS_PRODUCTION) {
      if (process.env.NODE_ENV !== 'production') {
        console.log(
          '%c🛡️ ConsoleGuard: отключён в dev-режиме (чтобы не ломать React DevTools)',
          'color: #888; font-size: 12px;'
        );
      }
      return;
    }

    console.log(
      '%c🛡️ ConsoleGuard: АКТИВЕН',
      'color: #00ff88; font-size: 14px; font-weight: bold;'
    );

    // ============================================================
    // 1. БЛОКИРОВКА CHROME CONSOLE
    // ============================================================
    let originalConsole = null;
    let originalLog = null;
    let originalWarn = null;
    let originalError = null;
    let originalClear = null;

    if (blockChromeConsole) {
      try {
        originalConsole = window.console;
        originalLog = window.console.log;
        originalWarn = window.console.warn;
        originalError = window.console.error;
        originalClear = window.console.clear;

        let _console = originalConsole;

        Object.defineProperty(window, 'console', {
          get: function () {
            // Эта проверка ловит ТОЛЬКО команды из консоли браузера
            // (React и обычный код её не триггерят)
            if (
              typeof window._commandLineAPI !== 'undefined' &&
              window._commandLineAPI !== null
            ) {
              throw new Error('🚫 Script execution from console is not permitted');
            }
            return _console;
          },
          set: function (val) {
            _console = val;
          },
          configurable: true,
        });
      } catch (e) {
        console.warn('ConsoleGuard: не удалось установить защиту console', e);
      }
    }

    // ============================================================
    // 2. ПРЕДУПРЕЖДЕНИЕ
    // ============================================================
    if (showWarning && originalLog) {
      try {
        const bigRed = 'color: red; font-size: 42px; font-weight: 900; text-shadow: 2px 2px 0 #000;';
        const mediumYellow = 'color: #ffcc00; font-size: 18px; font-weight: bold;';
        const normalWhite = 'color: #fff; font-size: 14px;';

        originalLog.call(window.console, '%c⛔ STOP!', bigRed);
        originalLog.call(window.console, '%cЭтот раздел предназначен для разработчиков.', mediumYellow);
        originalLog.call(window.console, '%cЕсли кто-то попросил вас вставить сюда код — это МОШЕННИЧЕСТВО.', normalWhite);
        originalLog.call(window.console, '%cВставка кода может привести к краже ваших данных.', normalWhite);
      } catch (e) {}
    }

    // ============================================================
    // 3. БЛОКИРОВКА ГОРЯЧИХ КЛАВИШ
    // ============================================================
    const hotkeyHandler = (e) => {
      if (e.key === 'F12' || e.keyCode === 123) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
      if (
        (e.ctrlKey || e.metaKey) &&
        e.shiftKey &&
        ['I', 'i', 'J', 'j', 'C', 'c'].includes(e.key)
      ) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
      if ((e.ctrlKey || e.metaKey) && ['U', 'u', 'S', 's'].includes(e.key)) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
    };

    if (blockHotkeys) {
      document.addEventListener('keydown', hotkeyHandler, true);
    }

    // ============================================================
    // 4. БЛОКИРОВКА КОНТЕКСТНОГО МЕНЮ
    // ============================================================
    const contextMenuHandler = (e) => {
      e.preventDefault();
      return false;
    };
    if (blockContextMenu) {
      document.addEventListener('contextmenu', contextMenuHandler, true);
    }

    // ============================================================
    // 5. ДЕТЕКТОР DEVTOOLS
    // ============================================================
    let devToolsInterval = null;
    let devToolsDetected = false;

    if (detectDevTools) {
      const threshold = 160;

      const checkDevTools = () => {
        const widthDiff = window.outerWidth - window.innerWidth;
        const heightDiff = window.outerHeight - window.innerHeight;
        const isOpen = widthDiff > threshold || heightDiff > threshold;

        if (isOpen && !devToolsDetected) {
          devToolsDetected = true;
          if (onDevToolsDetected) {
            try { onDevToolsDetected(); } catch (e) {}
          }
          if (showWarning && originalLog) {
            try {
              originalClear.call(window.console);
              originalLog.call(window.console, '%c🚨 ОБНАРУЖЕНЫ ИНСТРУМЕНТЫ РАЗРАБОТЧИКА', 'color: red; font-size: 24px; font-weight: bold;');
              originalLog.call(window.console, '%cЗакройте DevTools, чтобы продолжить.', 'color: #ffcc00; font-size: 16px;');
            } catch (e) {}
          }
        } else if (!isOpen && devToolsDetected) {
          devToolsDetected = false;
        }
      };

      devToolsInterval = setInterval(checkDevTools, 1000);
    }

    // ============================================================
    // CLEANUP
    // ============================================================
    return () => {
      if (blockHotkeys) {
        document.removeEventListener('keydown', hotkeyHandler, true);
      }
      if (blockContextMenu) {
        document.removeEventListener('contextmenu', contextMenuHandler, true);
      }
      if (devToolsInterval) {
        clearInterval(devToolsInterval);
      }
    };
  }, [
    enabled,
    showWarning,
    blockHotkeys,
    blockContextMenu,
    detectDevTools,
    blockChromeConsole,
    onDevToolsDetected,
  ]);

  return null;
};

export default ConsoleGuard;