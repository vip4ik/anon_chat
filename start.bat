@echo off
start "Backend" cmd /k "cd C:\Users\NACHISTO\anon_chat\backend && uvicorn server:app --reload --host 192.168.0.102 --port 8000"
start "Frontend" cmd /k "cd C:\Users\NACHISTO\anon_chat\frontend && npm start"
 