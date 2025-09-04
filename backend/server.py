from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import json
import logging
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uuid
from datetime import datetime, timedelta
from enum import Enum
import secrets
import base64
import numpy as np
import audioop
import bcrypt
import time
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
MAX_LOGIN_ATTEMPTS = 5
LOGIN_BLOCK_TIME = 300  # 5 minutes

# Security functions
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password with timing attack protection"""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except (ValueError, TypeError):
        # Always perform check for timing attack protection
        bcrypt.checkpw(b"dummy_password", hashed_password.encode('utf-8'))
        return False

# Global variables for brute force protection
login_attempts = {}
lock = asyncio.Lock()

async def check_brute_force_protection(username: str, request: Request) -> bool:
    """Check brute force protection with async lock"""
    client_ip = get_remote_address(request)
    key = f"{client_ip}:{username}"
    
    now = time.time()
    
    async with lock:
        if key in login_attempts:
            # Clean old attempts
            login_attempts[key] = [attempt for attempt in login_attempts[key] if now - attempt < LOGIN_BLOCK_TIME]
        
        if key not in login_attempts:
            login_attempts[key] = []
        
        # Check if exceeded max attempts
        if len(login_attempts[key]) >= MAX_LOGIN_ATTEMPTS:
            log_suspicious_activity(request, username, "brute_force_attempt")
            return False
        
        login_attempts[key].append(now)
        return True

async def reset_login_attempts(username: str, request: Request):
    """Reset login attempts counter"""
    client_ip = get_remote_address(request)
    key = f"{client_ip}:{username}"
    async with lock:
        if key in login_attempts:
            del login_attempts[key]

def log_suspicious_activity(request: Request, username: str, event_type: str):
    """Log suspicious activity"""
    client_ip = get_remote_address(request)
    user_agent = request.headers.get("user-agent", "")
    
    logger.warning(
        f"Suspicious activity - IP: {client_ip}, User: {username}, "
        f"Event: {event_type}, User-Agent: {user_agent}"
    )

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Application starting up")
    
    # MongoDB connection
    mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    app.state.client = AsyncIOMotorClient(mongo_url)
    app.state.db = app.state.client[os.environ.get('DB_NAME', 'voice_chat')]
    logger.info("Connected to MongoDB")
    
    yield
    
    # Shutdown logic
    logger.info("Application shutting down")
    app.state.client.close()
    logger.info("MongoDB connection closed")

# Initialize FastAPI with limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self' https://cdn.tailwindcss.com; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "microphone=(), camera=()"
    return response

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please try again later."},
        headers={"Retry-After": "60"}
    )

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Admin router
admin_router = APIRouter(prefix="/admin")

# Настройки авторизации (в реальном приложении хранить в .env)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
# Используем заранее сгенерированный хэш для пароля "admin123"
ADMIN_PASSWORD_HASHED = os.environ.get("ADMIN_PASSWORD_HASHED", "$2a$12$bvukpUyyRb6tg0.vE/bISOOEY3nw9N7S19O4F5pD6B4TlfYOr1khy")

security = HTTPBasic()

async def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate admin with brute force protection"""
    # Create a mock request object for brute force protection
    class MockRequest:
        def __init__(self):
            self.headers = {}
            self.client = type('Obj', (object,), {'host': 'localhost'})()
    
    mock_request = MockRequest()
    
    if not await check_brute_force_protection(credentials.username, mock_request):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again in 5 minutes.",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if (secrets.compare_digest(credentials.username, ADMIN_USERNAME) and 
        verify_password(credentials.password, ADMIN_PASSWORD_HASHED)):
        
        await reset_login_attempts(credentials.username, mock_request)
        return credentials.username
    
    log_suspicious_activity(mock_request, credentials.username, "failed_login")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

# User states
class UserStatus(str, Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

class UserGender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    ANY = "any"

class SearchCategory(str, Enum):
    ALL = "all"
    DATING = "dating"
    FRIENDS = "friends"
    FUN = "fun"
    ADVICE = "advice"
    GAMES = "games"

# Search parameters model
class SearchParams(BaseModel):
    category: SearchCategory = SearchCategory.ALL
    user_gender: UserGender
    search_gender: UserGender
    topics: List[str] = []

# Admin models
class DisconnectRequest(BaseModel):
    user_id: str

class BroadcastRequest(BaseModel):
    message: str

class MonitorRoomRequest(BaseModel):
    room_id: str

class AudioData(BaseModel):
    user_id: str
    room_id: str
    audio_chunk: str
    timestamp: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, dict] = {}
        self.search_queues: Dict[str, List[str]] = {
            'all': [], 'dating': [], 'friends': [], 'fun': [], 'advice': [], 'games': []
        }
        self.active_rooms: Dict[str, dict] = {}
        self.room_monitors: Dict[str, List[WebSocket]] = {}  # Для мониторинга комнат
        self.lock = asyncio.Lock()
        self.connection_attempts: Dict[str, List[float]] = {}  # Для защиты WebSocket

    async def check_websocket_rate_limit(self, user_id: str) -> bool:
        """Check WebSocket connection rate limit"""
        now = time.time()
        async with self.lock:
            if user_id not in self.connection_attempts:
                self.connection_attempts[user_id] = []
            
            # Clean old attempts (last 60 seconds)
            self.connection_attempts[user_id] = [
                attempt for attempt in self.connection_attempts[user_id] 
                if now - attempt < 60
            ]
            
            # Max 10 connections per minute
            if len(self.connection_attempts[user_id]) >= 10:
                return False
            
            self.connection_attempts[user_id].append(now)
            return True

    def calculate_volume_level(self, audio_bytes: bytes) -> float:
        """Рассчитывает уровень громкости от 0.0 до 1.0"""
        try:
            if len(audio_bytes) < 2:
                return 0.0
            
            # Используем audioop для расчета RMS (Root Mean Square)
            rms = audioop.rms(audio_bytes, 2)  # 2 = 16-bit samples
            
            # Нормализуем к диапазону 0.0 - 1.0
            # Максимальное значение для 16-bit audio: 32767
            max_rms = 32767.0
            normalized = min(rms / max_rms, 1.0)
            
            # Добавляем логарифмическую шкалу для лучшего восприятия
            if normalized > 0:
                log_normalized = np.log10(normalized * 9 + 1)  # 0-1 -> 0-1 с логарифмической шкалой
                return round(log_normalized, 2)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volume level: {e}")
            return 0.0

    async def connect(self, websocket: WebSocket, user_id: str):
        # Check rate limit
        if not await self.check_websocket_rate_limit(user_id):
            await websocket.close(code=1008, reason="Rate limit exceeded")
            return
            
        await websocket.accept()
        async with self.lock:
            self.active_connections[user_id] = websocket
            self.user_sessions[user_id] = {
                "user_id": user_id,
                "status": UserStatus.IDLE,
                "room_id": None,
                "partner_id": None,
                "search_params": None,
                "connected_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
        
        await self.send_personal_message({
            "type": "connected",
            "user_id": user_id,
            "message": "Подключен к голосовому чату",
            "online_stats": self.get_online_stats()
        }, user_id)

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            session = self.user_sessions.get(user_id, {})
            room_id = session.get("room_id")
            partner_id = session.get("partner_id")
            
            # Remove from search queues
            search_params = session.get("search_params")
            if search_params:
                category = search_params.get("category", "all")
                if user_id in self.search_queues[category]:
                    self.search_queues[category].remove(user_id)
            
            if room_id and partner_id:
                asyncio.create_task(self.handle_partner_disconnect(user_id, partner_id))
                
                if room_id in self.active_rooms:
                    del self.active_rooms[room_id]
            
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            
            del self.active_connections[user_id]
            logger.info(f"User {user_id} disconnected")

    async def handle_partner_disconnect(self, disconnected_user_id: str, partner_id: str):
        if partner_id in self.user_sessions:
            partner_session = self.user_sessions[partner_id]
            room_id = partner_session.get("room_id")
            
            partner_session["status"] = UserStatus.IDLE
            partner_session["room_id"] = None
            partner_session["partner_id"] = None
            partner_session["search_params"] = None
            
            if room_id in self.active_rooms:
                del self.active_rooms[room_id]
            
            await self.send_personal_message({
                "type": "partner_disconnected",
                "message": "Your partner has disconnected",
                "online_stats": self.get_online_stats()
            }, partner_id)

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                # Update last activity
                if user_id in self.user_sessions:
                    self.user_sessions[user_id]["last_activity"] = datetime.utcnow().isoformat()
                
                await self.active_connections[user_id].send_text(json.dumps(message, cls=DateTimeEncoder))
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(user_id)

    async def broadcast_message(self, message: dict):
        """Send message to all connected users"""
        disconnected_users = []
        for user_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message, cls=DateTimeEncoder))
            except Exception as e:
                logger.error(f"Error broadcasting to {user_id}: {e}")
                disconnected_users.append(user_id)
        
        # Remove disconnected users
        for user_id in disconnected_users:
            self.disconnect(user_id)

    async def force_disconnect_user(self, user_id: str):
        """Forcefully disconnect a user"""
        if user_id in self.active_connections:
            try:
                await self.send_personal_message({
                    "type": "force_disconnect",
                    "message": "Вы были отключены администратором",
                    "reason": "admin_action"
                }, user_id)
                
                # Close the connection
                await self.active_connections[user_id].close()
            except Exception as e:
                logger.error(f"Error force disconnecting {user_id}: {e}")
            finally:
                self.disconnect(user_id)

    async def start_search(self, user_id: str, search_params: dict):
        if user_id not in self.user_sessions:
            return False
        
        async with self.lock:
            session = self.user_sessions[user_id]
            session["status"] = UserStatus.SEARCHING
            session["search_params"] = search_params
            
            category = search_params.get("category", "all")
            if user_id not in self.search_queues[category]:
                self.search_queues[category].append(user_id)
            
            await self.send_personal_message({
                "type": "search_started",
                "message": "Поиск собеседника...",
                "search_params": search_params,
                "online_stats": self.get_online_stats()
            }, user_id)
        
        # Start matching process
        asyncio.create_task(self.find_match_for_user(user_id))
        return True

    async def find_match_for_user(self, user_id: str):
        """Find a match for the user based on search parameters"""
        max_attempts = 120  # 2 minutes of searching
        attempt = 0
        
        while attempt < max_attempts:
            # Check if user is still searching
            session = self.user_sessions.get(user_id)
            if not session or session.get("status") != UserStatus.SEARCHING:
                break
            
            # Try to find a match
            partner_id = await self.find_compatible_partner(user_id)
            
            if partner_id:
                # Create room and connect users
                await self.create_room(user_id, partner_id)
                break
            
            attempt += 1
            await asyncio.sleep(1)  # Check every second
        
        else:
            # Search timeout
            if user_id in self.user_sessions:
                await self.send_personal_message({
                    "type": "search_timeout",
                    "message": "Не удалось найти подходящего собеседника",
                    "online_stats": self.get_online_stats()
                }, user_id)
                await self.stop_search(user_id)

    async def find_compatible_partner(self, user_id: str) -> Optional[str]:
        """Find a compatible partner based on search parameters"""
        user_session = self.user_sessions.get(user_id)
        if not user_session or not user_session.get("search_params"):
            return None
        
        user_params = user_session["search_params"]
        user_category = user_params.get("category", "all")
        user_gender = user_params.get("user_gender")
        search_gender = user_params.get("search_gender")
        
        async with self.lock:
            for potential_partner_id in self.search_queues[user_category]:
                if potential_partner_id == user_id:
                    continue
                
                partner_session = self.user_sessions.get(potential_partner_id)
                if not partner_session or partner_session.get("status") != UserStatus.SEARCHING:
                    continue
                
                partner_params = partner_session.get("search_params", {})
                partner_gender = partner_params.get("user_gender")
                partner_search_gender = partner_params.get("search_gender")
                
                # Check gender compatibility
                if self.check_gender_compatibility(user_gender, search_gender, partner_gender, partner_search_gender):
                    return potential_partner_id
        
        return None

    def check_gender_compatibility(self, user_gender: str, user_search_gender: str, 
                                 partner_gender: str, partner_search_gender: str) -> bool:
        """Check if genders are compatible for matching"""
        # If both are searching for any gender
        if user_search_gender == "any" and partner_search_gender == "any":
            return True
        
        # If user is looking for specific gender and partner matches
        if user_search_gender != "any":
            if user_search_gender != partner_gender:
                return False
        
        # If partner is looking for specific gender and user matches
        if partner_search_gender != "any":
            if partner_search_gender != user_gender:
                return False
        
        return True

    async def create_room(self, user1_id: str, user2_id: str):
        room_id = str(uuid.uuid4())
        
        async with self.lock:
            self.user_sessions[user1_id]["status"] = UserStatus.CONNECTED
            self.user_sessions[user1_id]["room_id"] = room_id
            self.user_sessions[user1_id]["partner_id"] = user2_id
            
            self.user_sessions[user2_id]["status"] = UserStatus.CONNECTED
            self.user_sessions[user2_id]["room_id"] = room_id
            self.user_sessions[user2_id]["partner_id"] = user1_id
            
            # Remove from search queues
            user1_params = self.user_sessions[user1_id].get("search_params", {})
            user2_params = self.user_sessions[user2_id].get("search_params", {})
            
            user1_category = user1_params.get("category", "all")
            user2_category = user2_params.get("category", "all")
            
            if user1_id in self.search_queues[user1_category]:
                self.search_queues[user1_category].remove(user1_id)
            if user2_id in self.search_queues[user2_category]:
                self.search_queues[user2_category].remove(user2_id)
            
            self.active_rooms[room_id] = {
                "room_id": room_id,
                "participants": [user1_id, user2_id],
                "created_at": datetime.utcnow().isoformat()
            }
        
        await self.send_personal_message({
            "type": "match_found",
            "room_id": room_id,
            "partner_id": user2_id,
            "message": "Собеседник найден! Можете начинать общение.",
            "online_stats": self.get_online_stats()
        }, user1_id)
        
        await self.send_personal_message({
            "type": "match_found",
            "room_id": room_id,
            "partner_id": user1_id,
            "message": "Собеседник найден! Можете начинать общение.",
            "online_stats": self.get_online_stats()
        }, user2_id)

    async def stop_search(self, user_id: str):
        async with self.lock:
            session = self.user_sessions.get(user_id)
            if not session:
                return
            
            search_params = session.get("search_params")
            if search_params:
                category = search_params.get("category", "all")
                if user_id in self.search_queues[category]:
                    self.search_queues[category].remove(user_id)
            
            session["status"] = UserStatus.IDLE
            session["search_params"] = None
            
            await self.send_personal_message({
                "type": "search_stopped",
                "message": "Поиск остановлен",
                "online_stats": self.get_online_stats()
            }, user_id)

    async def next_partner(self, user_id: str):
        if user_id not in self.user_sessions:
            return
        
        session = self.user_sessions[user_id]
        partner_id = session.get("partner_id")
        room_id = session.get("room_id")
        
        if partner_id:
            await self.handle_partner_disconnect(user_id, partner_id)
        
        if room_id and room_id in self.active_rooms:
            del self.active_rooms[room_id]
        
        session["status"] = UserStatus.IDLE
        session["room_id"] = None
        session["partner_id"] = None
        
        # Restart search with previous parameters
        search_params = session.get("search_params")
        if search_params:
            await self.start_search(user_id, search_params)

    async def handle_webrtc_signal(self, user_id: str, signal_data: dict):
        if user_id not in self.user_sessions:
            return
        
        session = self.user_sessions[user_id]
        partner_id = session.get("partner_id")
        
        if partner_id:
            await self.send_personal_message({
                "type": "webrtc_signal",
                "from_user": user_id,
                "signal": signal_data
            }, partner_id)

    # Мониторинг комнат
    async def add_room_monitor(self, room_id: str, websocket: WebSocket):
        """Добавить монитор для комнаты"""
        async with self.lock:
            if room_id not in self.room_monitors:
                self.room_monitors[room_id] = []
            self.room_monitors[room_id].append(websocket)

    async def remove_room_monitor(self, room_id: str, websocket: WebSocket):
        """Удалить монитор для комнаты"""
        async with self.lock:
            if room_id in self.room_monitors:
                if websocket in self.room_monitors[room_id]:
                    self.room_monitors[room_id].remove(websocket)
                if not self.room_monitors[room_id]:
                    del self.room_monitors[room_id]

    async def broadcast_to_room_monitors(self, room_id: str, message: dict):
        """Отправить сообщение всем мониторам комнаты"""
        if room_id in self.room_monitors:
            disconnected_monitors = []
            for monitor_ws in self.room_monitors[room_id]:
                try:
                    await monitor_ws.send_text(json.dumps(message, cls=DateTimeEncoder))
                except Exception as e:
                    logger.error(f"Error sending to room monitor: {e}")
                    disconnected_monitors.append(monitor_ws)
            
            # Remove disconnected monitors
            for monitor_ws in disconnected_monitors:
                await self.remove_room_monitor(room_id, monitor_ws)

    async def handle_audio_data(self, user_id: str, room_id: str, audio_chunk: str):
        """Обработать аудиоданные и отправить мониторам комнаты"""
        session = self.user_sessions.get(user_id)
        if session and session.get("room_id") == room_id:
            try:
                # Декодируем аудиоданные из base64
                audio_bytes = base64.b64decode(audio_chunk)
                
                # Рассчитываем уровень громкости
                volume_level = self.calculate_volume_level(audio_bytes)
                
                # Отправляем аудиоданные всем мониторам этой комнаты
                await self.broadcast_to_room_monitors(room_id, {
                    "type": "audio_data",
                    "room_id": room_id,
                    "user_id": user_id,
                    "audio_chunk": audio_chunk,
                    "volume_level": volume_level,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")

    def get_monitored_rooms(self) -> List[dict]:
        """Получить список комнат под наблюдением"""
        monitored_info = []
        for room_id, monitors in self.room_monitors.items():
            room_info = self.active_rooms.get(room_id, {})
            monitored_info.append({
                "room_id": room_id,
                "participants": room_info.get("participants", []),
                "monitors_count": len(monitors),
                "created_at": room_info.get("created_at")
            })
        return monitored_info

    def get_online_stats(self) -> dict:
        """Get current online statistics"""
        return {
            "online_users": len(self.active_connections),
            "searching_users": sum(len(queue) for queue in self.search_queues.values()),
            "active_rooms": len(self.active_rooms),
            "monitored_rooms": len(self.room_monitors),
            "queues_by_category": {
                category: len(queue) for category, queue in self.search_queues.items()
            }
        }

    def get_all_connections(self) -> List[dict]:
        """Get all active connections for admin panel"""
        connections = []
        for user_id, session in self.user_sessions.items():
            connections.append({
                "user_id": user_id,
                "status": session.get("status"),
                "room_id": session.get("room_id"),
                "partner_id": session.get("partner_id"),
                "connected_at": session.get("connected_at"),
                "last_activity": session.get("last_activity"),
                "search_params": session.get("search_params")
            })
        return connections

    def get_detailed_stats(self) -> dict:
        """Get detailed statistics for admin panel"""
        stats = self.get_online_stats()
        stats.update({
            "total_connections": len(self.user_sessions),
            "connections": self.get_all_connections(),
            "active_rooms_details": [
                {
                    "room_id": room["room_id"],
                    "participants": room["participants"],
                    "created_at": room["created_at"]
                }
                for room in self.active_rooms.values()
            ],
            "monitored_rooms_details": self.get_monitored_rooms()
        })
        return stats

manager = ConnectionManager()

# Admin endpoints с авторизацией
@admin_router.get("/stats")
async def admin_stats(request: Request, username: str = Depends(authenticate_admin)):
    """Get detailed statistics for admin panel"""
    return manager.get_detailed_stats()

@admin_router.get("/connections")
async def admin_connections(request: Request, username: str = Depends(authenticate_admin)):
    """Get all active connections"""
    return manager.get_all_connections()

@admin_router.post("/disconnect")
async def admin_disconnect(request: DisconnectRequest, username: str = Depends(authenticate_admin)):
    """Force disconnect a user"""
    await manager.force_disconnect_user(request.user_id)
    return {"status": "success", "message": f"User {request.user_id} disconnected"}

@admin_router.post("/broadcast")
async def admin_broadcast(request: BroadcastRequest, username: str = Depends(authenticate_admin)):
    """Broadcast message to all users"""
    await manager.broadcast_message({
        "type": "system_message",
        "message": request.message,
        "timestamp": datetime.utcnow().isoformat(),
        "from_admin": True
    })
    return {"status": "success", "message": "Broadcast sent"}

@admin_router.post("/emergency-stop")
async def emergency_stop(username: str = Depends(authenticate_admin)):
    """Emergency stop - disconnect all users"""
    for user_id in list(manager.active_connections.keys()):
        await manager.force_disconnect_user(user_id)
    return {"status": "success", "message": "All users disconnected"}

@admin_router.get("/health")
async def health_check(username: str = Depends(authenticate_admin)):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "stats": manager.get_online_stats()
    }

# Эндпоинты для мониторинга
@admin_router.post("/monitor/start")
async def start_monitoring_room(request: MonitorRoomRequest, username: str = Depends(authenticate_admin)):
    """Начать мониторинг комнаты"""
    if request.room_id not in manager.active_rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return {
        "status": "success", 
        "message": f"Ready to monitor room {request.room_id}",
        "room_info": manager.active_rooms.get(request.room_id),
        "ws_url": f"/ws/monitor/{request.room_id}"
    }

@admin_router.post("/monitor/stop")
async def stop_monitoring_room(request: MonitorRoomRequest, username: str = Depends(authenticate_admin)):
    """Остановить мониторинг комнаты"""
    return {
        "status": "success", 
        "message": f"Stopped monitoring room {request.room_id}"
    }

@admin_router.get("/monitor/rooms")
async def get_monitored_rooms(username: str = Depends(authenticate_admin)):
    """Получить список комнат под наблюдением"""
    return manager.get_monitored_rooms()

@admin_router.post("/audio")
async def receive_audio_data(request: AudioData, username: str = Depends(authenticate_admin)):
    """Принять аудиоданные для мониторинга"""
    await manager.handle_audio_data(
        request.user_id, 
        request.room_id, 
        request.audio_chunk
    )
    return {"status": "success", "message": "Audio data received"}

# WebSocket endpoint для мониторинга комнат
@app.websocket("/ws/monitor/{room_id}")
async def monitor_websocket(websocket: WebSocket, room_id: str):
    """WebSocket для мониторинга комнаты в реальном времени"""
    
    await websocket.accept()
    
    try:
        # Проверяем существует ли комната
        if room_id not in manager.active_rooms:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Room {room_id} not found"
            }))
            await websocket.close()
            return
        
        # Добавляем монитор комнаты
        await manager.add_room_monitor(room_id, websocket)
        
        # Отправляем информацию о комнате
        room_info = manager.active_rooms.get(room_id, {})
        await websocket.send_text(json.dumps({
            "type": "monitor_started",
            "room_id": room_id,
            "room_info": room_info,
            "message": "Мониторинг комнаты начат",
            "participants": room_info.get("participants", [])
        }, cls=DateTimeEncoder))
        
        # Отправляем текущее состояние комнаты
        participants_info = []
        for user_id in room_info.get("participants", []):
            session = manager.user_sessions.get(user_id, {})
            participants_info.append({
                "user_id": user_id,
                "status": session.get("status", "unknown"),
                "connected_at": session.get("connected_at")
            })
        
        await websocket.send_text(json.dumps({
            "type": "room_status",
            "room_id": room_id,
            "participants": participants_info,
            "active": True
        }, cls=DateTimeEncoder))
        
        # Ждем сообщений от администратора
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "stop_monitoring":
                break
            elif message["type"] == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"Monitor disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"Monitor error: {e}")
    finally:
        # Удаляем монитор
        await manager.remove_room_monitor(room_id, websocket)
        try:
            await websocket.close()
        except:
            pass

# WebSocket endpoint для пользователей с упрощенной аутентификацией
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint с упрощенной аутентификацией"""
    
    # Упрощенная аутентификация - проверяем только rate limit
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_search":
                search_params = message.get("search_params", {})
                await manager.start_search(user_id, search_params)
            elif message["type"] == "stop_search":
                await manager.stop_search(user_id)
            elif message["type"] == "next_partner":
                await manager.next_partner(user_id)
            elif message["type"] == "webrtc_signal":
                await manager.handle_webrtc_signal(user_id, message["signal"])
            elif message["type"] == "disconnect":
                await manager.handle_partner_disconnect(user_id, message.get("partner_id"))
            elif message["type"] == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "online_stats": manager.get_online_stats()
                }, user_id)
            elif message["type"] == "get_online_stats":
                await manager.send_personal_message({
                    "type": "online_stats",
                    "online_stats": manager.get_online_stats()
                }, user_id)
            elif message["type"] == "audio_data":
                # Перехватываем аудиоданные для мониторинга
                session = manager.user_sessions.get(user_id, {})
                room_id = session.get("room_id")
                if room_id and message.get("audio_chunk"):
                    await manager.handle_audio_data(
                        user_id, 
                        room_id, 
                        message["audio_chunk"]
                    )
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)

# Public endpoints
@api_router.get("/stats")
async def get_stats(request: Request):
    return manager.get_online_stats()

@api_router.get("/health")
async def public_health_check(request: Request):
    """Public health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Include routers
app.include_router(api_router)
app.include_router(admin_router)

# Эндпоинт для админ-панели
@app.get("/admin")
async def serve_admin_panel(request: Request):
    """Serve the admin panel HTML"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Панель управления голосовым чатом</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .bg-gray-750:hover { background-color: #374151; }
            .login-form { backdrop-filter: blur(10px); }
            .audio-visualizer { 
                height: 20px; 
                background: linear-gradient(90deg, #4F46E5, #EC4899);
                border-radius: 10px;
                transition: all 0.1s ease;
            }
            .participant-active { border-left: 4px solid #10B981; }
        </style>
    </head>
    <body class="min-h-screen bg-gray-900 text-white">
        <!-- Форма логина -->
        <div id="loginForm" class="fixed inset-0 flex items-center justify-center login-form bg-black bg-opacity-50 z-50">
            <div class="bg-gray-800 p-8 rounded-xl shadow-2xl w-96">
                <h2 class="text-2xl font-bold mb-6 text-center">Авторизация</h2>
                <form id="loginFormElement" onsubmit="handleLogin(event); return false;">
                    <div class="mb-4">
                        <label class="block text-sm font-medium mb-2">Логин</label>
                        <input type="text" id="username" required
                            class="w-full bg-gray-700 border border-gray-600 rounded px-4 py-3 focus:outline-none focus:border-purple-500"
                            placeholder="Введите логин" value="admin"/>
                    </div>
                    <div class="mb-6">
                                                 <label class="block text-sm font-medium mb-2">Пароль</label>
                        <input type="password" id="password" required
                            class="w-full bg-gray-700 border border-gray-600 rounded px-4 py-3 focus:outline-none focus:border-purple-500"
                            placeholder="Введите пароль" value="admin123"/>
                    </div>
                    <button type="submit"
                        class="w-full bg-purple-600 hover:bg-purple-700 py-3 px-4 rounded font-medium transition-colors">
                        Войти
                    </button>
                    <div id="loginError" class="text-red-400 text-sm mt-3 hidden">
                        Неверный логин или пароль
                    </div>
                    <div id="rateLimitError" class="text-red-400 text-sm mt-3 hidden">
                        Слишком много попыток. Попробуйте через 5 минут.
                    </div>
                </form>
            </div>
        </div>

        <!-- Основной контент панели -->
        <div id="adminPanel" class="hidden">
            <div class="p-6">
                <div class="max-w-7xl mx-auto">
                    <div class="flex justify-between items-center mb-8">
                        <h1 class="text-3xl font-bold">Панель управления голосовым чатом</h1>
                        <button onclick="logout()" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded">
                            Выйти
                        </button>
                    </div>
                    
                    <!-- Статистика -->
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8" id="statsContainer">
                        <div class="bg-gray-800 p-6 rounded-lg">
                            <h3 class="text-lg font-semibold mb-2">Всего пользователей</h3>
                            <p class="text-3xl font-bold text-purple-400" id="totalUsers">0</p>
                        </div>
                        <div class="bg-gray-800 p-6 rounded-lg">
                            <h3 class="text-lg font-semibold mb-2">Онлайн сейчас</h3>
                            <p class="text-3xl font-bold text-green-400" id="onlineUsers">0</p>
                        </div>
                        <div class="bg-gray-800 p-6 rounded-lg">
                            <h3 class="text-lg font-semibold mb-2">Активных комнат</h3>
                            <p class="text-3xl font-bold text-blue-400" id="activeRooms">0</p>
                        </div>
                        <div class="bg-gray-800 p-6 rounded-lg">
                            <h3 class="text-lg font-semibold mb-2">В поиске</h3>
                            <p class="text-3xl font-bold text-yellow-400" id="searchingUsers">0</p>
                        </div>
                        <div class="bg-gray-800 p-6 rounded-lg">
                            <h3 class="text-lg font-semibold mb-2">Мониторится</h3>
                            <p class="text-3xl font-bold text-pink-400" id="monitoredRooms">0</p>
                        </div>
                    </div>

                    <!-- Мониторинг комнат -->
                    <div class="bg-gray-800 p-6 rounded-lg mb-8">
                        <h2 class="text-xl font-semibold mb-4">🎧 Мониторинг комнат</h2>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                            <div>
                                <h3 class="text-lg font-medium mb-3">Начать прослушивание</h3>
                                <div class="flex gap-2 mb-3">
                                    <input type="text" placeholder="ID комнаты..." 
                                        class="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
                                        id="monitorRoomId"/>
                                    <button onclick="startMonitoring()"
                                        class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded">
                                        ▶️ Слушать
                                    </button>
                                </div>
                                <div class="text-sm text-gray-400 mb-4">
                                    💡 Для получения ID комнаты проверьте активные подключения
                                </div>
                            </div>
                            
                            <div>
                                <h3 class="text-lg font-medium mb-3">Активные мониторы</h3>
                                <div id="activeMonitors" class="text-sm text-gray-300 bg-gray-700 p-3 rounded max-h-32 overflow-y-auto">
                                    Загрузка...
                                </div>
                            </div>
                        </div>
                        
                        <!-- Панель прослушивания -->
                        <div id="audioMonitor" class="hidden bg-gray-900 p-6 rounded-lg border border-blue-500">
                            <div class="flex justify-between items-center mb-4">
                                <h4 class="text-lg font-medium">🎧 Прослушивание комната: 
                                    <span id="currentRoomId" class="font-mono bg-blue-900 px-2 py-1 rounded"></span>
                                </h4>
                                <button onclick="stopMonitoring()"
                                    class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-sm">
                                    ⏹️ Остановить
                                </button>
                            </div>
                            
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                    <div class="flex items-center space-x-3 mb-4">
                                        <div class="w-3 h-3 bg-red-500 rounded-full" id="audioIndicator"></div>
                                        <span class="text-sm" id="monitorStatus">Подключение...</span>
                                        <div class="audio-visualizer" id="audioVisualizer"></div>
                                    </div>
                                    
                                    <div class="bg-gray-800 p-4 rounded">
                                        <h5 class="font-medium mb-2">📊 Активность</h5>
                                        <div id="activityStats">
                                            <div>Получено аудио: <span id="audioChunksCount">0</span></div>
                                            <div>Последнее: <span id="lastAudioTime">-</span></div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div>
                                    <h5 class="font-medium mb-2">👥 Участники комнаты</h5>
                                    <div id="participantsList" class="space-y-2">
                                        <!-- Участники будут здесь -->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Управление -->
                    <div class="bg-gray-800 p-6 rounded-lg mb-8">
                        <h2 class="text-xl font-semibold mb-4">⚙️ Управление системой</h2>
                        
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <h3 class="text-lg font-medium mb-3">📢 Системное сообщение</h3>
                                <div class="flex gap-2">
                                    <input type="text" placeholder="Введите сообщение..."
                                        class="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-purple-500"
                                        id="systemMessage"/>
                                    <button onclick="sendSystemMessage()"
                                        class="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded">
                                        Отправить
                                    </button>
                                </div>
                            </div>
                            
                            <div>
                                <h3 class="text-lg font-medium mb-3">🚨 Действия</h3>
                                <div class="flex gap-2">
                                    <button onclick="emergencyStop()" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded">
                                        🔴 Экстренная остановка
                                    </button>
                                    <button onclick="refreshData()" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded">
                                        🔄 Обновить данные
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Активные подключения -->
                    <div class="bg-gray-800 p-6 rounded-lg">
                        <div class="flex justify-between items-center mb-4">
                            <h2 class="text-xl font-semibold">🔗 Активные подключения</h2>
                            <button onclick="refreshConnections()" class="bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded text-sm">
                                Обновить
                            </button>
                        </div>
                        
                        <div class="overflow-x-auto">
                            <table class="w-full" id="connectionsTable">
                                <thead>
                                    <tr class="border-b border-gray-700">
                                        <th class="text-left py-2">ID пользователя</th>
                                        <th class="text-left py-2">Статус</th>
                                        <th class="text-left py-2">Комната</th>
                                        <th class="text-left py-2">Время подключения</th>
                                        <th class="text-left py-2">Последняя активность</th>
                                        <th class="text-left py-2">Действия</th>
                                    </tr>
                                </thead>
                                <tbody id="connectionsBody">
                                    <!-- Данные будут заполнены через JavaScript -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Автоматическое определение базового URL
            const getBaseUrl = () => {
                return window.location.origin; // Используем текущий origin
            };

            const BACKEND_URL = getBaseUrl();
            let refreshInterval;
            let authToken = '';
            let monitorWebSocket = null;
            let currentMonitoredRoom = null;
            let audioChunksCount = 0;
            let audioVisualizerInterval = null;
            let audioPlayers = {};

            console.log('Using backend URL:', BACKEND_URL);

            // Функция для базовой авторизации
            function getAuthHeaders() {
                if (authToken) {
                    return {
                        'Authorization': `Basic ${authToken}`,
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    };
                }
                return { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                };
            }

            // Обработка логина
            async function handleLogin(event) {
                event.preventDefault();
                
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                
                // Скрываем предыдущие ошибки
                document.getElementById('loginError').classList.add('hidden');
                document.getElementById('rateLimitError').classList.add('hidden');
                
                // Кодируем в base64 для Basic Auth
                authToken = btoa(`${username}:${password}`);
                
                console.log('Trying to login with:', { username, authToken });
                
                // Проверяем авторизацию
                try {
                    const healthUrl = `${BACKEND_URL}/admin/health`;
                    console.log('Making request to:', healthUrl);
                    
                    const response = await fetch(healthUrl, {
                        headers: getAuthHeaders(),
                        credentials: 'include'
                    });
                    
                    console.log('Login response status:', response.status, response.statusText);
                    
                    if (response.ok) {
                        console.log('Login successful');
                        // Успешная авторизация
                        document.getElementById('loginForm').classList.add('hidden');
                        document.getElementById('adminPanel').classList.remove('hidden');
                        localStorage.setItem('adminAuth', authToken);
                        refreshData();
                    } else if (response.status === 429) {
                        document.getElementById('rateLimitError').classList.remove('hidden');
                    } else {
                        document.getElementById('loginError').classList.remove('hidden');
                    }
                } catch (error) {
                    console.error('Login error:', error);
                    document.getElementById('loginError').classList.remove('hidden');
                }
                
                return false;
            }

            // Показать ошибку авторизации
            function showLoginError() {
                document.getElementById('loginError').classList.remove('hidden');
                authToken = '';
            }

            // Выход из системы
            function logout() {
                authToken = '';
                localStorage.removeItem('adminAuth');
                document.getElementById('loginForm').classList.remove('hidden');
                document.getElementById('adminPanel').classList.add('hidden');
                document.getElementById('username').value = '';
                document.getElementById('password').value = '';
                document.getElementById('loginError').classList.add('hidden');
                document.getElementById('rateLimitError').classList.add('hidden');
                
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                }
                stopMonitoring();
            }

            // Проверка сохраненной авторизации при загрузке
            function checkSavedAuth() {
                const savedAuth = localStorage.getItem('adminAuth');
                if (savedAuth) {
                    authToken = savedAuth;
                    console.log('Found saved auth token');
                    
                    // Проверяем валидность токена
                    fetch(`${BACKEND_URL}/admin/health`, {
                        headers: getAuthHeaders(),
                        credentials: 'include'
                    })
                    .then(response => {
                        console.log('Saved auth check status:', response.status);
                        if (response.ok) {
                            document.getElementById('loginForm').classList.add('hidden');
                            document.getElementById('adminPanel').classList.remove('hidden');
                            refreshData();
                        } else if (response.status === 429) {
                            document.getElementById('rateLimitError').classList.remove('hidden');
                            localStorage.removeItem('adminAuth');
                        } else {
                            localStorage.removeItem('adminAuth');
                        }
                    })
                    .catch(error => {
                        console.error('Saved auth check error:', error);
                        localStorage.removeItem('adminAuth');
                    });
                } else {
                    console.log('No saved auth token found');
                }
            }

            // Функции API
            async function fetchStats() {
                try {
                    const response = await fetch(`${BACKEND_URL}/admin/stats`, {
                        headers: getAuthHeaders(),
                        credentials: 'include'
                    });
                    if (response.ok) {
                        const data = await response.json();
                        updateStats(data);
                    } else if (response.status === 401 || response.status === 429) {
                        logout();
                    }
                } catch (error) {
                    console.error('Error fetching stats:', error);
                }
            }

            async function fetchConnections() {
                try {
                    const response = await fetch(`${BACKEND_URL}/admin/connections`, {
                        headers: getAuthHeaders(),
                        credentials: 'include'
                    });
                    if (response.ok) {
                        const data = await response.json();
                        updateConnections(data);
                    } else if (response.status === 401 || response.status === 429) {
                        logout();
                    }
                } catch (error) {
                    console.error('Error fetching connections:', error);
                }
            }

            async function refreshMonitors() {
                try {
                    const response = await fetch(`${BACKEND_URL}/admin/monitor/rooms`, {
                        headers: getAuthHeaders(),
                        credentials: 'include'
                    });
                    
                    if (response.ok) {
                        const monitors = await response.json();
                        updateMonitorsList(monitors);
                    }
                } catch (error) {
                    console.error('Error fetching monitors:', error);
                }
            }

            // Мониторинг комнат
            async function startMonitoring() {
                const roomId = document.getElementById('monitorRoomId').value.trim();
                if (!roomId) {
                    alert('Введите ID комнаты');
                    return;
                }

                try {
                    const response = await fetch(`${BACKEND_URL}/admin/monitor/start`, {
                        method: 'POST',
                        headers: getAuthHeaders(),
                        credentials: 'include',
                        body: JSON.stringify({ room_id: roomId })
                    });

                    if (response.ok) {
                        currentMonitoredRoom = roomId;
                        setupMonitorWebSocket(roomId);
                    } else if (response.status === 429) {
                        alert('Слишком много запросов. Попробуйте позже.');
                    } else {
                        alert('Комната не найдена или ошибка доступа');
                    }
                } catch (error) {
                    console.error('Error starting monitoring:', error);
                    alert('Ошибка начала мониторинга');
                }
            }

            function setupMonitorWebSocket(roomId) {
                try {
                    const wsUrl = `${BACKEND_URL.replace('http', 'ws')}/ws/monitor/${roomId}`;
                    monitorWebSocket = new WebSocket(wsUrl);
                    
                    monitorWebSocket.onopen = function() {
                        console.log('Monitor WebSocket connected');
                        document.getElementById('monitorStatus').textContent = 'Подключено к комнате';
                        document.getElementById('audioIndicator').className = 'w-3 h-3 bg-green-500 rounded-full animate-ping';
                    };

                    monitorWebSocket.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        handleMonitorMessage(data);
                    };

                    monitorWebSocket.onclose = function() {
                        console.log('Monitor connection closed');
                        stopMonitoring();
                    };

                    monitorWebSocket.onerror = function(error) {
                        console.error('Monitor WebSocket error:', error);
                        alert('Ошибка подключения к мониторингу');
                        stopMonitoring();
                    };

                    // Показываем панель мониторинга
                    document.getElementById('audioMonitor').classList.remove('hidden');
                    document.getElementById('currentRoomId').textContent = roomId;
                    
                } catch (error) {
                    console.error('Error setting up monitor:', error);
                    alert('Ошибка подключения к мониторингу');
                }
            }

            function handleMonitorMessage(data) {
                switch (data.type) {
                    case 'audio_data':
                        handleAudioData(data);
                        break;
                    case 'monitor_started':
                        updateParticipantsList(data.participants || []);
                        break;
                    case 'room_status':
                        updateParticipantsList(data.participants || []);
                        break;
                }
            }

            function handleAudioData(data) {
                audioChunksCount++;
                document.getElementById('audioChunksCount').textContent = audioChunksCount;
                document.getElementById('lastAudioTime').textContent = new Date().toLocaleTimeString();
                
                // Анимация индикатора
                const indicator = document.getElementById('audioIndicator');
                indicator.className = 'w-3 h-3 bg-green-500 rounded-full animate-ping';
                
                setTimeout(() => {
                    indicator.className = 'w-3 h-3 bg-green-500 rounded-full';
                }, 300);
            }

            function updateParticipantsList(participants) {
                const container = document.getElementById('participantsList');
                container.innerHTML = '';
                
                participants.forEach(participant => {
                    const userId = participant.user_id || participant;
                    const participantElement = document.createElement('div');
                    participantElement.id = `participant-${userId}`;
                    participantElement.className = 'bg-gray-700 p-2 rounded text-sm';
                    participantElement.innerHTML = `
                        <div class="font-medium">👤 ${userId}</div>
                        <div class="text-xs text-gray-400">Статус: connected</div>
                    `;
                    container.appendChild(participantElement);
                });
            }

            async function stopMonitoring() {
                if (monitorWebSocket) {
                    monitorWebSocket.close();
                    monitorWebSocket = null;
                }
                
                if (currentMonitoredRoom) {
                    try {
                        await fetch(`${BACKEND_URL}/admin/monitor/stop`, {
                            method: 'POST',
                            headers: getAuthHeaders(),
                            credentials: 'include',
                            body: JSON.stringify({ room_id: currentMonitoredRoom })
                        });
                    } catch (error) {
                        console.error('Error stopping monitoring:', error);
                    }
                    
                    currentMonitoredRoom = null;
                }
                
                document.getElementById('audioMonitor').classList.add('hidden');
                document.getElementById('monitorRoomId').value = '';
                audioChunksCount = 0;
            }

            function updateMonitorsList(monitors) {
                const container = document.getElementById('activeMonitors');
                
                if (monitors.length === 0) {
                    container.innerHTML = '<div class="text-gray-400">Нет активных мониторов</div>';
                    return;
                }
                
                container.innerHTML = monitors.map(monitor => `
                    <div class="mb-2 p-2 bg-gray-700 rounded">
                        <div class="font-medium">Комната: ${monitor.room_id}</div>
                        <div class="text-xs">Участников: ${monitor.participants?.length || 0}</div>
                        <div class="text-xs">Мониторов: ${monitor.monitors_count}</div>
                    </div>
                `).join('');
            }

            async function forceDisconnect(userId) {
                if (!confirm(`Вы уверены, что хотите отключить пользователя ${userId}?`)) {
                    return;
                }

                try {
                    const response = await fetch(`${BACKEND_URL}/admin/disconnect`, {
                        method: 'POST',
                        headers: getAuthHeaders(),
                        credentials: 'include',
                        body: JSON.stringify({ user_id: userId })
                    });
                    
                    if (response.ok) {
                        alert('Пользователь отключен');
                        fetchConnections();
                    } else if (response.status === 429) {
                        alert('Слишком много запросов. Попробуйте позже.');
                    }
                } catch (error) {
                    console.error('Error disconnecting user:', error);
                    alert('Ошибка отключения пользователя');
                }
            }

            async function sendSystemMessage() {
                const messageInput = document.getElementById('systemMessage');
                const message = messageInput.value.trim();
                
                if (!message) {
                    alert('Введите сообщение');
                    return;
                }

                try {
                    const response = await fetch(`${BACKEND_URL}/admin/broadcast`, {
                        method: 'POST',
                        headers: getAuthHeaders(),
                        credentials: 'include',
                        body: JSON.stringify({ message })
                    });
                    
                    if (response.ok) {
                        alert('Сообщение отправлено');
                        messageInput.value = '';
                    } else if (response.status === 429) {
                        alert('Слишком много запросов. Попробуйте позже.');
                    }
                } catch (error) {
                    console.error('Error sending message:', error);
                    alert('Ошибка отправки сообщения');
                }
            }

            async function emergencyStop() {
                if (confirm('Вы уверены, что хотите экстренно остановить систему? Все пользователи будут отключены.')) {
                    try {
                        const response = await fetch(`${BACKEND_URL}/admin/emergency-stop`, {
                            method: 'POST',
                            headers: getAuthHeaders(),
                            credentials: 'include'
                        });
                        
                        if (response.ok) {
                            alert('Система остановлена');
                            refreshData();
                        } else if (response.status === 429) {
                            alert('Слишком много запросов. Попробуйте позже.');
                        }
                    } catch (error) {
                        console.error('Error emergency stop:', error);
                    }
                }
            }

            function updateStats(data) {
                document.getElementById('totalUsers').textContent = data.total_connections || 0;
                document.getElementById('onlineUsers').textContent = data.online_users || 0;
                document.getElementById('activeRooms').textContent = data.active_rooms || 0;
                document.getElementById('searchingUsers').textContent = data.searching_users || 0;
                document.getElementById('monitoredRooms').textContent = data.monitored_rooms || 0;
            }

            function updateConnections(connections) {
                const tbody = document.getElementById('connectionsBody');
                tbody.innerHTML = '';

                if (!connections || connections.length === 0) {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="6" class="py-4 text-center text-gray-400">
                                Нет активных подключений
                            </td>
                        </tr>
                    `;
                    return;
                }

                connections.forEach(conn => {
                    const row = document.createElement('tr');
                    row.className = 'border-b border-gray-700 hover:bg-gray-750';
                    
                    const statusClass = conn.status === 'connected' ? 'bg-green-600' :
                                      conn.status === 'searching' ? 'bg-yellow-600' : 'bg-gray-600';
                    
                    const connectedAt = conn.connected_at ? new Date(conn.connected_at).toLocaleTimeString() : '-';
                    const lastActivity = conn.last_activity ? new Date(conn.last_activity).toLocaleTimeString() : '-';
                    
                    row.innerHTML = `
                        <td class="py-3 font-mono text-sm">${conn.user_id}</td>
                        <td class="py-3">
                            <span class="px-2 py-1 rounded text-xs ${statusClass}">
                                ${conn.status}
                            </span>
                        </td>
                        <td class="py-3 font-mono text-sm">${conn.room_id || '-'}</td>
                        <td class="py-3 text-sm">${connectedAt}</td>
                        <td class="py-3 text-sm">${lastActivity}</td>
                        <td class="py-3">
                            <button onclick="forceDisconnect('${conn.user_id}')"
                                class="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm mr-2">
                                Отключить
                            </button>
                            ${conn.room_id ? `
                            <button onclick="document.getElementById('monitorRoomId').value='${conn.room_id}'"
                                class="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm">
                                Слушать
                            </button>
                            ` : ''}
                        </td>
                    `;
                    
                    tbody.appendChild(row);
                });
            }

            async function refreshData() {
                await Promise.all([fetchStats(), fetchConnections(), refreshMonitors()]);
            }

            function refreshConnections() {
                fetchConnections();
            }

            // Загрузка данных при загрузке страницы
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Admin panel loaded, backend URL:', BACKEND_URL);
                checkSavedAuth();
                
                // Обновление данных каждые 5 секунд
                refreshInterval = setInterval(refreshData, 5000);
            });

            window.addEventListener('beforeunload', function() {
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                }
                stopMonitoring();
            });

            // Глобальные функции
            window.forceDisconnect = forceDisconnect;
            window.sendSystemMessage = sendSystemMessage;
            window.emergencyStop = emergencyStop;
            window.refreshData = refreshData;
            window.refreshConnections = refreshConnections;
            window.handleLogin = handleLogin;
            window.logout = logout;
            window.startMonitoring = startMonitoring;
            window.stopMonitoring = stopMonitoring;
        </script>
    </body>
    </html>
    """)

# Эндпоинт для проверки работы сервера
@app.get("/health")
async def health_check(request: Request):
    return {"status": "ok", "message": "Server is running", "timestamp": datetime.utcnow().isoformat()}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.0.102:8000", "http://localhost:8000", "http://192.168.0.102:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.102", port=8000, reload=True)