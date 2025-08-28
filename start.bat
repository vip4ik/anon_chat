@echo off
start "Backend" cmd /k "cd C:\Users\NACHISTO\anon_chat\backend && uvicorn server:app --host 192.168.0.102 --port 8000 --reload"
start "Frontend" cmd /k "cd C:\Users\NACHISTO\anon_chat\frontend && npm start"
python C:\Users\NACHISTO\anon_chat\monitoring.py 