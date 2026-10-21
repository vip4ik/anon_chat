# ============================================================
# ИМПОРТЫ
# ============================================================
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, Query, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import json
import logging
import asyncio
import certifi
from pathlib import Path
import zipfile
import shutil
from fastapi import UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Set
from bson import ObjectId
from datetime import datetime, timedelta
import uuid
from enum import Enum
import secrets
import base64
import numpy as np
import audioop
import bcrypt
import time
import ssl
import psutil
import hashlib
import hmac
import traceback
from urllib.parse import urlencode
from starlette.requests import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import aiohttp
from livekit import api as livekit_api
import threading
from datetime import datetime as dt_cls

# ============================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# ============================================================
# КОНФИГУРАЦИЯ ОПЛАТЫ
# ============================================================
class PaymentConfig:
    def __init__(self):
        self.MORUNE_PAYMENT_URL = os.environ.get("MORUNE_PAYMENT_URL", "https://api.morune.com/v1")
        self.BACKEND_URL = os.environ.get("BACKEND_URL", "http://192.168.0.103:8000")
        self.MORUNE_API_KEY = os.environ.get("MORUNE_API_KEY", "")
        self.MORUNE_SECRET_KEY = os.environ.get("MORUNE_SECRET_KEY", "")
        self.MORUNE_SHOP_ID = os.environ.get("MORUNE_SHOP_ID", "")
        
        if not all([self.MORUNE_API_KEY, self.MORUNE_SECRET_KEY, self.MORUNE_SHOP_ID]):
            print("=" * 80)
            print("⚠️  ВНИМАНИЕ: Используются ТЕСТОВЫЕ ключи Morune")
            print("=" * 80)
            self.MORUNE_API_KEY = "test_api_key_demo_123456"
            self.MORUNE_SECRET_KEY = "test_secret_key_demo_789012"
            self.MORUNE_SHOP_ID = "test_shop_demo_456789"
            self.TEST_MODE = True
        else:
            self.TEST_MODE = False
        
        self.CURRENCY = "RUB"
        self.PAYMENT_TIMEOUT = 3600
        self.SUCCESS_URL = f"{self.BACKEND_URL}/api/payment/success"
        self.FAIL_URL = f"{self.BACKEND_URL}/api/payment/fail"
        self.CALLBACK_URL = f"{self.BACKEND_URL}/api/payment/callback"

PAYMENT_CONFIG = PaymentConfig()

CONFIG = type('Config', (), {
    'MORUNE_PAYMENT_URL': PAYMENT_CONFIG.MORUNE_PAYMENT_URL,
    'BACKEND_URL': PAYMENT_CONFIG.BACKEND_URL,
    'MORUNE_API_KEY': PAYMENT_CONFIG.MORUNE_API_KEY,
    'MORUNE_SECRET_KEY': PAYMENT_CONFIG.MORUNE_SECRET_KEY,
    'MORUNE_SHOP_ID': PAYMENT_CONFIG.MORUNE_SHOP_ID
})()

# ============================================================
# КОНФИГУРАЦИЯ LIVEKIT
# ============================================================
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://192.168.0.103:7880")
# ============================================================

class LiveKitRoomManager:
    """
    Хранит активные LiveKit-комнаты и их заполненность.
    Нужен, чтобы клиенты попадали в существующую комнату с местом,
    а не создавали каждый раз новую.
    """
    def __init__(self):
        self.rooms = {}  # {room_name: {"count": int, "max": int, "users": set, "created": datetime}}
        self.lock = threading.Lock()
        self.MAX_PARTICIPANTS = 6
        self.ROOM_TTL_SECONDS = 3600  # час на комнату, если в ней никого

    def find_or_create_room(self, user_id: str, room_type: str = "group6") -> str:
        """
        Возвращает имя комнаты:
        - если есть комната с местом → возвращает её
        - если нет → создаёт новую
        """
        with self.lock:
            now = datetime.utcnow()
            self._cleanup_locked(now)

            # Определяем лимит по типу
            max_size = 2 if room_type == "single" else 6

            # Ищем комнату с местом ТОГО ЖЕ типа
            for room_name, info in self.rooms.items():
                if info.get("type") != room_type:
                    continue
                if info["count"] < info["max"]:
                    if user_id not in info["users"]:
                        info["users"].add(user_id)
                        info["count"] = len(info["users"])
                    print(f"🎥 [LK-ROOM] User {user_id} → existing room {room_name} ({info['count']}/{info['max']})")
                    return room_name

            # Не нашли — создаём новую
            new_room = f"{room_type}_{uuid.uuid4().hex[:12]}"
            self.rooms[new_room] = {
                "count": 1,
                "max": max_size,
                "users": {user_id},
                "created": now,
                "type": room_type,
            }
            print(f"🎥 [LK-ROOM] Created NEW room {new_room} for {user_id} (type={room_type})")
            return new_room

    def leave_room(self, user_id: str, room_name: str):
        """Убирает пользователя из комнаты. Если комната пуста — удаляет."""
        with self.lock:
            if room_name not in self.rooms:
                return
            info = self.rooms[room_name]
            info["users"].discard(user_id)
            info["count"] = len(info["users"])
            print(f"🎥 [LK-ROOM] User {user_id} left {room_name} ({info['count']}/{info['max']})")
            if info["count"] == 0:
                del self.rooms[room_name]
                print(f"🎥 [LK-ROOM] Room {room_name} deleted (empty)")

    def _cleanup_locked(self, now):
        """Удаляет пустые комнаты сразу и старые комнаты по TTL."""
        to_delete = []
        for room_name, info in self.rooms.items():
            # Пустые комнаты удаляем сразу
            if info["count"] == 0:
                to_delete.append(room_name)
                print(f"🎥 [LK-ROOM] Cleanup empty room {room_name}")
                continue
            # Комнаты старше TTL тоже удаляем
            age = (now - info["created"]).total_seconds()
            if age > self.ROOM_TTL_SECONDS:
                to_delete.append(room_name)
                print(f"🎥 [LK-ROOM] Cleanup old room {room_name} (age {age/60:.0f} min)")
        for room_name in to_delete:
            del self.rooms[room_name]

    def get_stats(self):
        with self.lock:
            return {
                "total_rooms": len(self.rooms),
                "rooms": [
                    {
                        "name": name,
                        "count": info["count"],
                        "max": info["max"],
                        "type": info.get("type", "group6"),
                    }
                    for name, info in self.rooms.items()
                ]
            }

livekit_room_manager = LiveKitRoomManager()

# АДМИНСКИЙ ЧАТ (ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ)
# ============================================================
admin_chat_messages = []
ADMIN_CHAT_MAX_MESSAGES = 100

# ============================================================
# СОЗДАНИЕ ПРИЛОЖЕНИЯ (РАНЬШЕ, ЧТОБЫ БЫЛО ДОСТУПНО)
# ============================================================
app = FastAPI(
    title="Cosmic Voice",
    version="1.0.0",
    docs_url=None,      # 🆕 отключает /docs
    redoc_url=None,     # 🆕 отключает /redoc
    openapi_url=None,   # 🆕 отключает /openapi.json (без него /docs тоже не работает)
)

# ============================================================
# МЕНЕДЖЕР АДМИНСКОГО ЧАТА
# ============================================================
class AdminChatManager:
    def __init__(self):
        self.active_connections = set()
    
    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"Admin chat WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Admin chat WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message):
        """Отправка сообщения всем подключенным клиентам"""
        if not self.active_connections:
            print("⚠️ [CHAT] No active connections to broadcast")
            return
        
        disconnected = []
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                print(f"⚠️ [CHAT] Broadcast error: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
        
        print(f"✅ [CHAT] Broadcast to {len(self.active_connections)} clients")

admin_chat_manager = AdminChatManager()

# ============================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С ЧАТОМ (ИСПРАВЛЕНЫ)
# ============================================================
async def save_chat_message(message: dict):
    """Сохранение сообщения чата в БД"""
    try:
        # Проверяем наличие collections
        if not hasattr(app.state, 'chat_messages_collection') or app.state.chat_messages_collection is None:
            print("⚠️ [CHAT] Chat messages collection not available")
            return
        
        message_data = {
            "id": message.get("id"),
            "sender": message.get("sender"),
            "sender_name": message.get("sender_name"),
            "text": message.get("text"),
            "data": message.get("data"),
            "name": message.get("name"),
            "mime": message.get("mime"),
            "duration": message.get("duration"),
            "type": message.get("type", "message"),
            "timestamp": datetime.utcnow(),
            "timestamp_str": message.get("timestamp")
        }
        await app.state.chat_messages_collection.insert_one(message_data)
        
        # Ограничиваем количество сообщений
        count = await app.state.chat_messages_collection.count_documents({})
        if count > 1000:
            oldest = await app.state.chat_messages_collection.find().sort("timestamp", 1).limit(1).to_list(1)
            if oldest:
                await app.state.chat_messages_collection.delete_one({"_id": oldest[0]["_id"]})
    except Exception as e:
        print(f"⚠️ [CHAT] Error saving message to DB: {e}")

async def load_chat_history(limit: int = 100):
    """Загрузка истории чата из БД"""
    try:
        if not hasattr(app.state, 'chat_messages_collection') or app.state.chat_messages_collection is None:
            print("⚠️ [CHAT] Chat messages collection not available")
            return []
        
        cursor = app.state.chat_messages_collection.find().sort("timestamp", -1).limit(limit)
        messages = await cursor.to_list(length=limit)
        messages.reverse()
        return messages
    except Exception as e:
        print(f"⚠️ [CHAT] Error loading history from DB: {e}")
        return []

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def serialize_mongo_doc(doc):
    if doc is None:
        return None
    if isinstance(doc, list):
        return [serialize_mongo_doc(item) for item in doc]
    if isinstance(doc, dict):
        result = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, datetime):
                moscow_offset = timedelta(hours=3)
                moscow_time = value + moscow_offset
                result[key] = moscow_time.isoformat()
            elif isinstance(value, (list, dict)):
                result[key] = serialize_mongo_doc(value)
            else:
                result[key] = value
        return result
    return doc

MAX_LOGIN_ATTEMPTS = 5
LOGIN_BLOCK_TIME = 300
MAX_REPORTS_PER_USER = 5
REPORT_TIME_WINDOW = 3600
AUTO_BLOCK_THRESHOLD = 3
MODERATOR_NOTIFICATION_THRESHOLD = 2

login_attempts = {}
lock = asyncio.Lock()

# ============================================================
# МОДЕЛИ ДАННЫХ
# ============================================================
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

class ReportReason(str, Enum):
    SPAM = "spam"
    HARASSMENT = "harassment"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    VOICE_ABUSE = "voice_abuse"
    OTHER = "other"

class ReportRequest(BaseModel):
    reason: ReportReason
    description: str
    partner_id: str
    room_id: str
    user_id: str

class SearchParams(BaseModel):
    category: SearchCategory = SearchCategory.ALL
    user_gender: UserGender
    search_gender: UserGender
    # 🆕 Гео и возраст
    country: Optional[str] = "any"
    user_age: Optional[str] = "any"
    search_age: Optional[str] = "any"
    topics: List[str] = []

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

class SystemSettingsRequest(BaseModel):
    maintenance_mode: Optional[bool] = None
    max_users: Optional[int] = None
    allow_registration: Optional[bool] = None
    system_message: Optional[str] = None
    muffle_enabled: Optional[bool] = None
    maintenance_message: Optional[str] = None

class ClearQueuesRequest(BaseModel):
    categories: Optional[List[str]] = None

class MaintenanceRequest(BaseModel):
    enabled: bool
    message: Optional[str] = None

class BlockUserRequest(BaseModel):
    user_id: str
    reason: str
    duration_hours: Optional[int] = 24

class LiveKitTokenRequest(BaseModel):
    user_id: str
    room_name: Optional[str] = None  # если None — бэк сам найдёт/создаст
    nickname: Optional[str] = "User"
    room_type: Optional[str] = "group6"  # group6 / single

# ============================================================
# МОДЕЛИ ДЛЯ УПРАВЛЕНИЯ ПЕРСОНАЛОМ
# ============================================================
ALL_MODERATOR_PERMISSIONS = [
    "view",              # просмотр панели
    "view_reports",      # просмотр жалоб
    "warn",              # предупреждения
    "mute",              # заглушение
    "kick",              # выгнать
    "ban",               # бан
    "manage_users",      # управление пользователями
    "manage_settings",   # изменение настроек системы
    "manage_ads",        # управление рекламой
    "manage_payment",    # управление оплатой
    "manage_premium",    # управление премиумом
    "manage_staff",      # управление персоналом
    "view_logs",         # просмотр логов
    "advanced_ban",      # расширенный бан (IP/browser/fingerprint/subnet)
]

class StaffCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)
    role: str = Field("moderator")  # "admin" или "moderator"
    permissions: List[str] = []
    is_active: bool = True

class StaffUpdateRequest(BaseModel):
    password: Optional[str] = None
    role: Optional[str] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None

class ChangePasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=4, max_length=100)        

# ============================================================
# ФУНКЦИИ БЕЗОПАСНОСТИ
# ============================================================
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except (ValueError, TypeError):
        bcrypt.checkpw(b"dummy_password", hashed_password.encode('utf-8'))
        return False

# ============================================================
# ФУНКЦИИ МОДЕРАЦИИ
# ============================================================
MODERATION_ACTION_TRANSLATIONS = {
    "warn": "Предупреждение",
    "mute": "Заглушение", 
    "unmute": "Снятие заглушения",
    "kick": "Выгон",
    "ban": "Бан",
    "unban": "Разбан",
    "advanced_ban": "Продвинутый бан",
    "ip_ban": "Бан по IP",
    "subnet_ban": "Бан подсети",
    "report_resolved": "Решение жалобы",
    "create_staff": "Создание персонала",
    "update_staff": "Обновление персонала",
    "delete_staff": "Удаление персонала",
    "change_staff_password": "Смена пароля персонала"

}

REASON_TRANSLATIONS = {
    "spam": "Спам",
    "harassment": "Харассмент",
    "inappropriate_content": "Неуместный контент",
    "voice_abuse": "Злоупотребление голосом",
    "ddos": "DDOS атака",
    "multiple_accounts": "Мультиаккаунт",
    "other": "Другое",
    "no_action": "Без действия",
    "manual_unmute": "Ручное снятие заглушения",
    "manual_unban": "Ручной разбан",
    "automatic": "Автоматическое действие"
}

def translate_reason(reason: str) -> str:
    """Переводит код причины на русский. Если не найдено — возвращает как есть."""
    if not reason:
        return "Нарушение правил"
    return REASON_TRANSLATIONS.get(reason, reason)

async def parse_moderation_request(request: Request):
    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty request body")
        try:
            data = await request.json()
            return data
        except:
            try:
                import json
                data = json.loads(body.decode('utf-8'))
                return data
            except:
                text = body.decode('utf-8')
                import re
                json_match = re.search(r'\{.*\}', text)
                if json_match:
                    import json
                    data = json.loads(json_match.group())
                    return data
                raise HTTPException(status_code=400, detail="Invalid JSON format")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error parsing request: {e}")
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

async def get_user_ip(request: Request) -> str:
    if request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[0]
    return request.client.host

async def get_ip_info(ip: str) -> dict:
    if not ip or ip == "unknown":
        return {"query": ip, "status": "fail", "message": "No IP address"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,region,regionName,city,zip,lat,lon,timezone,isp,org,as,query", 
                timeout=2
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
    except:
        pass
    return {"query": ip, "status": "fail"}

async def safe_db_operation(operation, fallback_value=None, operation_name="unknown"):
    try:
        # ИСПРАВЛЕНО: проверяем на None
        if app.state.db is None:
            logger.warning(f"Database not connected for operation: {operation_name}")
            return fallback_value
        result = await operation()
        return result
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        logger.error(f"MongoDB connection error in {operation_name}: {e}")
        return fallback_value
    except Exception as e:
        logger.error(f"Database operation failed in {operation_name}: {e}")
        return fallback_value

async def check_report_spam(reporter_id: str) -> bool:
    async def operation():
        now = datetime.utcnow()
        time_threshold = now - timedelta(seconds=REPORT_TIME_WINDOW)
        report_count = await app.state.reports_collection.count_documents({
            "reporter_id": reporter_id,
            "timestamp": {"$gte": time_threshold}
        })
        return report_count >= MAX_REPORTS_PER_USER
    return await safe_db_operation(operation, False, "check_report_spam")

async def get_user_report_count(reported_user_id: str) -> int:
    async def operation():
        return await app.state.reports_collection.count_documents({
            "reported_user_id": reported_user_id
        })
    return await safe_db_operation(operation, 0, "get_user_report_count")

async def is_user_blocked(user_id: str) -> bool:
    async def operation():
        blocked_user = await app.state.blocked_users_collection.find_one({
            "user_id": user_id,
            "$or": [
                {"expires_at": {"$gt": datetime.utcnow()}},
                {"expires_at": None}
            ]
        })
        return blocked_user is not None
    return await safe_db_operation(operation, False, "is_user_blocked")

async def is_user_fully_blocked(user_id: str, ip_address: str = None, browser_id: str = None, fingerprint: str = None) -> dict:
    """Проверка бана по всем параметрам. Возвращает {blocked: bool, reason: str, ban_type: str}"""
    result = {"blocked": False, "reason": None, "ban_type": None}
    
    now = datetime.utcnow()
    active_filter = {
        "$or": [
            {"is_permanent": True},
            {"expires_at": {"$gt": now}},
            {"expires_at": None}
        ]
    }
    
    # ===== 1. Проверка по user_id =====
    try:
        if hasattr(app.state, 'blocked_users_collection') and app.state.blocked_users_collection is not None:
            blocked_user = await app.state.blocked_users_collection.find_one({
                "user_id": user_id,
                **active_filter
            })
            if blocked_user:
                reason = blocked_user.get('reason', 'нарушение правил')
                return {"blocked": True, "reason": f"Ваш аккаунт заблокирован: {reason}", "ban_type": "user"}
    except Exception as e:
        print(f"⚠️ [BAN-CHECK] user_id check error: {e}")
    
    # ===== 2. Проверка по IP =====
    if ip_address and ip_address not in ("unknown", "0.0.0.0", "127.0.0.1"):
        try:
            if hasattr(app.state, 'blocked_ips_collection') and app.state.blocked_ips_collection is not None:
                blocked_ip = await app.state.blocked_ips_collection.find_one({
                    "ip": ip_address,
                    **active_filter
                })
                if blocked_ip:
                    reason = blocked_ip.get('reason', 'нарушение правил')
                    return {"blocked": True, "reason": f"Ваш IP заблокирован: {reason}", "ban_type": "ip"}
        except Exception as e:
            print(f"⚠️ [BAN-CHECK] IP check error: {e}")
    
    # ===== 3. Проверка по Browser ID =====
    if browser_id and browser_id != "unknown":
        try:
            if hasattr(app.state, 'blocked_browsers_collection') and app.state.blocked_browsers_collection is not None:
                blocked_browser = await app.state.blocked_browsers_collection.find_one({
                    "browser_id": browser_id,
                    **active_filter
                })
                if blocked_browser:
                    reason = blocked_browser.get('reason', 'нарушение правил')
                    return {"blocked": True, "reason": f"Ваш браузер заблокирован: {reason}", "ban_type": "browser"}
        except Exception as e:
            print(f"⚠️ [BAN-CHECK] Browser check error: {e}")
    
    # ===== 4. Проверка по Fingerprint =====
    if fingerprint and fingerprint != "unknown":
        try:
            if hasattr(app.state, 'blocked_fingerprints_collection') and app.state.blocked_fingerprints_collection is not None:
                blocked_fp = await app.state.blocked_fingerprints_collection.find_one({
                    "fingerprint": fingerprint,
                    **active_filter
                })
                if blocked_fp:
                    reason = blocked_fp.get('reason', 'нарушение правил')
                    return {"blocked": True, "reason": f"Ваше устройство заблокировано: {reason}", "ban_type": "fingerprint"}
        except Exception as e:
            print(f"⚠️ [BAN-CHECK] Fingerprint check error: {e}")
    
    # ===== 5. Проверка по подсети =====
    if ip_address and ip_address not in ("unknown", "0.0.0.0", "127.0.0.1"):
        try:
            ip_parts = ip_address.split('.')
            if len(ip_parts) >= 3 and ip_parts[0] != '0':
                subnet = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"
                if hasattr(app.state, 'blocked_subnets_collection') and app.state.blocked_subnets_collection is not None:
                    blocked_subnet = await app.state.blocked_subnets_collection.find_one({
                        "subnet": subnet,
                        **active_filter
                    })
                    if blocked_subnet:
                        reason = blocked_subnet.get('reason', 'нарушение правил')
                        return {"blocked": True, "reason": f"Ваша подсеть заблокирована: {subnet} ({reason})", "ban_type": "subnet"}
        except Exception as e:
            print(f"⚠️ [BAN-CHECK] Subnet check error: {e}")
    
    return result    

async def block_user(user_id: str, reason: str, duration_hours: int = 24):
    async def operation():
        is_permanent = (duration_hours == 0)
        expires_at = None if is_permanent else (datetime.utcnow() + timedelta(hours=duration_hours))
        await app.state.blocked_users_collection.update_one(
            {"user_id": user_id},
            {"$set": {
                "user_id": user_id,
                "reason": reason,
                "blocked_at": datetime.utcnow(),
                "expires_at": expires_at,
                "is_permanent": is_permanent,
                "ban_type": "user"
            }},
            upsert=True
        )
        await notify_moderators(f"Пользователь {user_id} был заблокирован. Причина: {reason}. Срок: {'перманентно' if is_permanent else f'{duration_hours} часов'}")
    await safe_db_operation(operation, None, "block_user")

async def notify_moderators(message: str):
    async def operation():
        moderators = await app.state.moderators_collection.find({"is_active": True}).to_list(None)
        for moderator in moderators:
            logger.info(f"Moderator notification for {moderator['username']}: {message}")
        logger.info(f"Moderator notification: {message}")
    await safe_db_operation(operation, None, "notify_moderators")

async def log_moderation_action(moderator, action: str, target_user: str, reason: str, duration: Optional[int] = None):
    async def operation():
        if moderator is None:
            moderator_username = "system"
        elif isinstance(moderator, dict):
            moderator_username = moderator.get('username', 'system')
        else:
            moderator_username = str(moderator)
        
        action_ru = MODERATION_ACTION_TRANSLATIONS.get(action, action)
        reason_ru = REASON_TRANSLATIONS.get(reason, reason)
        
        log_entry = {
            "moderator": moderator_username,
            "action": action,
            "action_ru": action_ru,
            "target_user": target_user,
            "reason": reason,
            "reason_ru": reason_ru,
            "duration": duration,
            "duration_text": f"{duration} минут" if duration else None,
            "timestamp": datetime.utcnow(),
            "ip_address": "moderation_panel"
        }
        
        if hasattr(app.state, 'moderation_logs_collection') and app.state.moderation_logs_collection is not None:
            await app.state.moderation_logs_collection.insert_one(log_entry)
            print(f"📝 [LOG] Moderation action logged: {moderator_username} {action_ru} {target_user}")
        else:
            print(f"⚠️ [LOG] Moderation logs collection not available, action not logged")
    await safe_db_operation(operation, None, "log_moderation_action")

async def get_user_warnings(user_id: str) -> List[dict]:
    async def operation():
        try:
            warnings = await app.state.warnings_collection.find({
                "user_id": user_id,
                "expires_at": {"$gt": datetime.utcnow()}
            }).sort("created_at", -1).to_list(100)
            return serialize_mongo_doc(warnings)
        except Exception as e:
            logger.error(f"Error getting warnings for {user_id}: {e}")
            return []
    return await safe_db_operation(operation, [], "get_user_warnings")

async def get_user_mutes(user_id: str) -> List[dict]:
    async def operation():
        try:
            mutes = await app.state.mutes_collection.find({
                "user_id": user_id,
                "expires_at": {"$gt": datetime.utcnow()}
            }).sort("created_at", -1).to_list(100)
            return serialize_mongo_doc(mutes)
        except Exception as e:
            logger.error(f"Error getting mutes for {user_id}: {e}")
            return []
    return await safe_db_operation(operation, [], "get_user_mutes")

async def is_user_muted(user_id: str) -> bool:
    async def operation():
        mute = await app.state.mutes_collection.find_one({
            "user_id": user_id,
            "expires_at": {"$gt": datetime.utcnow()}
        })
        return mute is not None
    return await safe_db_operation(operation, False, "is_user_muted")

async def warn_user(user_id: str, reason: str, severity: str = "low", moderator: str = None):
    if moderator is None:
        moderator_username = "system"
    elif isinstance(moderator, dict):
        moderator_username = moderator.get('username', 'system')
    else:
        moderator_username = str(moderator)
    
    # 🆕 ЕДИНСТВЕННОЕ ЧТО ДОБАВИЛОСЬ — одна строчка
    reason_ru = translate_reason(reason)
    
    async def operation():
        warning_data = {
            "user_id": user_id,
            "reason": reason_ru,       # ← теперь "Спам" вместо "spam"
            "severity": severity,
            "moderator": moderator_username,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=30)
        }
        if hasattr(app.state, 'warnings_collection') and app.state.warnings_collection is not None:
            await app.state.warnings_collection.insert_one(warning_data)
    await safe_db_operation(operation, None, "warn_user")
    await log_moderation_action(moderator_username, "warn", user_id, reason_ru)  # ← и здесь reason_ru
    return True

async def mute_user(user_id: str, reason: str, duration_minutes: int = 60, moderator: str = None):
    if moderator is None:
        moderator_username = "system"
    elif isinstance(moderator, dict):
        moderator_username = moderator.get('username', 'system')
    else:
        moderator_username = str(moderator)
    
    # 🆕 Переводим причину
    reason_ru = translate_reason(reason)

    async def operation():
        expires_at = datetime.utcnow() + timedelta(minutes=duration_minutes)
        mute_data = {
            "user_id": user_id,
            "reason": reason_ru,  # 🆕
            "moderator": moderator_username,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "duration_minutes": duration_minutes
        }
        if hasattr(app.state, 'mutes_collection') and app.state.mutes_collection is not None:
            await app.state.mutes_collection.insert_one(mute_data)
    await safe_db_operation(operation, None, "mute_user")
    await log_moderation_action(moderator_username, "mute", user_id, reason, duration_minutes)
    return True

async def unmute_user(user_id: str, moderator: str = "system"):
    async def operation():
        await app.state.mutes_collection.update_many(
            {"user_id": user_id, "expires_at": {"$gt": datetime.utcnow()}},
            {"$set": {"expires_at": datetime.utcnow()}}
        )
    await safe_db_operation(operation, None, "unmute_user")
    await log_moderation_action(moderator, "unmute", user_id, "Снятие заглушки")

async def kick_user(user_id: str, reason: str, moderator: str = "system"):
    async def operation():
        kick_data = {
            "user_id": user_id,
            "reason": reason_ru,
            "moderator": moderator,
            "created_at": datetime.utcnow()
        }
        if hasattr(app.state, 'kicks_collection') and app.state.kicks_collection is not None:
            await app.state.kicks_collection.insert_one(kick_data)
    await safe_db_operation(operation, None, "kick_user")
    await log_moderation_action(moderator, "kick", user_id, reason)

async def ban_user(user_id: str, reason: str, duration_hours: int = 24, moderator: str = None):
    if moderator is None:
        moderator_username = "system"
    elif isinstance(moderator, dict):
        moderator_username = moderator.get('username', 'system')
    else:
        moderator_username = str(moderator)
        reason_ru = translate_reason(reason)
    
    async def operation():
        expires_at = datetime.utcnow() + timedelta(hours=duration_hours) if duration_hours > 0 else None
        ban_data = {
            "user_id": user_id,
            "reason": reason_ru,
            "moderator": moderator_username,
            "banned_at": datetime.utcnow(),
            "expires_at": expires_at,
            "is_permanent": duration_hours == 0,
            "type": "user_ban",
            "ban_type": "user"
        }
        if hasattr(app.state, 'blocked_users_collection') and app.state.blocked_users_collection is not None:
            await app.state.blocked_users_collection.update_one(
                {"user_id": user_id},
                {"$set": ban_data},
                upsert=True
            )
    await safe_db_operation(operation, None, "ban_user")
    await log_moderation_action(moderator_username, "ban", user_id, reason, duration_hours * 60)
    return True

async def unban_user(user_id: str, moderator: str = "system"):
    async def operation():
        result = await app.state.blocked_users_collection.delete_one({"user_id": user_id})
        return result.deleted_count > 0
    success = await safe_db_operation(operation, False, "unban_user")
    if success:
        await log_moderation_action(moderator, "unban", user_id, "Ручной разбан")
    return success

async def get_user_moderation_status(user_id: str) -> dict:
    try:
        warnings = await get_user_warnings(user_id)
        mutes = await get_user_mutes(user_id)
        is_banned = await is_user_blocked(user_id)
        return {
            "user_id": user_id,
            "is_muted": len(mutes) > 0,
            "is_banned": is_banned,
            "warnings_count": len(warnings),
            "active_warnings": warnings,
            "active_mutes": mutes,
            "total_reports": await get_user_report_count(user_id)
        }
    except Exception as e:
        logger.error(f"Error in get_user_moderation_status for {user_id}: {e}")
        return {
            "user_id": user_id,
            "is_muted": False,
            "is_banned": False,
            "warnings_count": 0,
            "active_warnings": [],
            "active_mutes": [],
            "total_reports": 0
        }

async def save_user_fingerprint(user_id: str, ip_address: str, fingerprint: str = None):
    try:
        if hasattr(app.state, 'users_collection') and app.state.users_collection is not None:
            await app.state.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {
                    "ip_address": ip_address,
                    "browser_fingerprint": fingerprint,
                    "last_connection": datetime.utcnow()
                }},
                upsert=True
            )
    except Exception as e:
        print(f"⚠️ Error saving fingerprint: {e}")

# ============================================================
# ПОДКЛЮЧЕНИЕ К MONGODB
# ============================================================
async def setup_mongodb_connection():
    atlas_url = os.environ.get('MONGO_URL', '')
    db_name = os.environ.get('DB_NAME', 'voice_chat')
    
    if not atlas_url:
        print("❌ [MongoDB] MONGO_URL не найден в .env!")
        return None, None, {}
    
    # Убираем лишние кавычки
    atlas_url = atlas_url.strip('"').strip("'")
    
    print(f"🔗 [MongoDB] Подключение к: {atlas_url[:80]}...")
    print(f"📁 [MongoDB] Имя БД: {db_name}")
    
    # ===== ПРОБУЕМ ПОДКЛЮЧЕНИЕ СРАЗУ (БЕЗ СПИСКА) =====
    configs_to_try = [
        # 1. Минимальная
        {},
        # 2. С сертификатом
        {'tlsCAFile': certifi.where()},
        # 3. Без проверки сертификата
        {
            'tlsAllowInvalidCertificates': True,
            'tlsAllowInvalidHostnames': True
        },
        # 4. Полная конфигурация
        {
            'tlsCAFile': certifi.where(),
            'tlsAllowInvalidCertificates': False,
            'serverSelectionTimeoutMS': 30000,
            'connectTimeoutMS': 30000,
            'socketTimeoutMS': 30000,
            'retryWrites': True,
            'w': 'majority'
        }
    ]
    
    last_error = None
    
    for i, kwargs in enumerate(configs_to_try, 1):
        try:
            print(f"🔄 [MongoDB] Попытка {i} с параметрами: {list(kwargs.keys()) if kwargs else 'минимальная'}")
            
            client = AsyncIOMotorClient(atlas_url, **kwargs)
            await client.admin.command('ping')
            print(f"✅ [MongoDB] ПОДКЛЮЧЕНИЕ УСПЕШНО! (попытка {i})")
            
            db = client[db_name]
            
            # Создаем коллекции
            collections = {
                "reports_collection": db["reports"],
                "blocked_users_collection": db["blocked_users"],
                "moderators_collection": db["moderators"],
                "users_collection": db["users"],
                "connections_collection": db["connections"],
                "rooms_collection": db["rooms"],
                "warnings_collection": db["warnings"],
                "mutes_collection": db["mutes"],
                "kicks_collection": db["kicks"],
                "moderation_logs_collection": db["moderation_logs"],
                "video_orders_collection": db["video_orders"],
                "system_settings_collection": db["system_settings"],
                "video_tariffs_collection": db["video_tariffs"],
                "messages_log_collection": db["messages_log"],
                "audio_ads_collection": db["audio_ads"],
                "video_ads_collection": db["video_ads"],
                "premium_collection": db["premium_users"],
                "premium_settings_collection": db["premium_settings"],
                "chat_messages_collection": db["chat_messages"],
                "blocked_ips_collection": db["blocked_ips"],
                "blocked_browsers_collection": db["blocked_browsers"],
                "blocked_fingerprints_collection": db["blocked_fingerprints"],
                "blocked_subnets_collection": db["blocked_subnets"],
                "pending_notifications_collection": db["pending_notifications"],
            }
            
            await setup_mongodb_indexes(collections)
            await initialize_default_moderator(collections["moderators_collection"])
            
            app.state.db_type = "MongoDB Atlas"
            app.state.db_url = atlas_url
            
            return client, db, collections
            
        except Exception as e:
            print(f"   ❌ Ошибка: {str(e)[:150]}")
            last_error = e
            continue
    
    print("=" * 80)
    print("❌ [MongoDB] НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ!")
    print(f"❌ Последняя ошибка: {last_error}")
    print("=" * 80)
    
    # Пустые коллекции
    empty_collections = {}
    collection_names = [
        "reports_collection", "blocked_users_collection", "moderators_collection",
        "users_collection", "connections_collection", "rooms_collection",
        "warnings_collection", "mutes_collection", "kicks_collection",
        "moderation_logs_collection", "video_orders_collection",
        "system_settings_collection", "video_tariffs_collection",
        "messages_log_collection", "audio_ads_collection", "video_ads_collection",
        "premium_collection", "premium_settings_collection", "chat_messages_collection",
        "blocked_ips_collection", "blocked_browsers_collection",
        "blocked_fingerprints_collection", "blocked_subnets_collection",
        "pending_notifications_collection"
    ]
    for name in collection_names:
        empty_collections[name] = None
    
    return None, None, empty_collections

async def setup_mongodb_indexes(collections):
    try:
        if collections.get("reports_collection") is not None:
            await collections["reports_collection"].create_index("timestamp")
            await collections["reports_collection"].create_index("reporter_id")
            await collections["reports_collection"].create_index("reported_user_id")
            await collections["reports_collection"].create_index([("reporter_id", 1), ("timestamp", 1)])
        
        if collections.get("users_collection") is not None:
            await collections["users_collection"].create_index("user_id")
            await collections["users_collection"].create_index("last_activity")
        
        if collections.get("connections_collection") is not None:
            await collections["connections_collection"].create_index("session_id")
            await collections["connections_collection"].create_index("timestamp")
        
        if collections.get("rooms_collection") is not None:
            await collections["rooms_collection"].create_index("room_id")
        
        if collections.get("blocked_users_collection") is not None:
            await collections["blocked_users_collection"].create_index("user_id")
            await collections["blocked_users_collection"].create_index("expires_at")
        
        if collections.get("video_orders_collection") is not None:
            await collections["video_orders_collection"].create_index("order_id", unique=True)
            await collections["video_orders_collection"].create_index("user_id")
            await collections["video_orders_collection"].create_index("status")
            await collections["video_orders_collection"].create_index("created_at")
        
        if collections.get("video_tariffs_collection") is not None:
            await collections["video_tariffs_collection"].create_index("id", unique=True)
            await collections["video_tariffs_collection"].create_index("is_active")
            await collections["video_tariffs_collection"].create_index("created_at")

        if collections.get("blocked_ips_collection") is not None:
            await collections["blocked_ips_collection"].create_index("ip")
            await collections["blocked_ips_collection"].create_index("expires_at")

        if collections.get("blocked_browsers_collection") is not None:
            await collections["blocked_browsers_collection"].create_index("browser_id")
            await collections["blocked_browsers_collection"].create_index("expires_at")

        if collections.get("blocked_fingerprints_collection") is not None:
            await collections["blocked_fingerprints_collection"].create_index("fingerprint")
            await collections["blocked_fingerprints_collection"].create_index("expires_at")

        if collections.get("blocked_subnets_collection") is not None:
            await collections["blocked_subnets_collection"].create_index("subnet")
            await collections["blocked_subnets_collection"].create_index("expires_at")                
        
        logger.info("✅ MongoDB indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")

async def initialize_default_moderator(moderators_collection):
    if moderators_collection is None:
        logger.warning("Moderators collection is not available - skipping default moderator creation")
        return
    try:
        existing_moderator = await moderators_collection.find_one({"username": "admin"})
        if not existing_moderator:
            default_password = "admin123"
            password_hash = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            moderator_data = {
                "username": "admin",
                "password_hash": password_hash,
                "role": "admin",
                "permissions": ["view", "warn", "mute", "kick", "ban", "manage_users", "view_reports", "manage_settings"],
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            await moderators_collection.insert_one(moderator_data)
            logger.info("✅ Default admin moderator created")
        else:
            logger.info("✅ Admin moderator already exists")
    except Exception as e:
        logger.error(f"Error initializing default moderator: {e}")

async def sync_user_data_to_db():
    print("🔄 [SYNC] Starting data synchronization to MongoDB...")
    async def operation():
        if not hasattr(app.state, 'users_collection') or app.state.users_collection is None:
            return {"error": "Users collection not available"}
        if not hasattr(app.state, 'connections_collection') or app.state.connections_collection is None:
            return {"error": "Connections collection not available"}
        if not hasattr(app.state, 'rooms_collection') or app.state.rooms_collection is None:
            return {"error": "Rooms collection not available"}
        
        results = {"users_updated": 0, "connections_created": 0, "rooms_updated": 0, "errors": []}
        
        for user_id, session in manager.user_sessions.items():
            try:
                user_data = {
                    "user_id": user_id,
                    "browser_id": session.get("browser_id"),
                    "ip_address": session.get("ip_address"),
                    "status": session.get("status"),
                    "room_id": session.get("room_id"),
                    "partner_id": session.get("partner_id"),
                    "last_activity": datetime.utcnow(),
                    "connected_at": session.get("connected_at", datetime.utcnow()),
                    "updated_at": datetime.utcnow()
                }
                await app.state.users_collection.update_one(
                    {"user_id": user_id},
                    {"$set": user_data},
                    upsert=True
                )
                results["users_updated"] += 1
            except Exception as e:
                results["errors"].append(f"Error syncing user {user_id}: {e}")
        
        for user_id, session in manager.user_sessions.items():
            try:
                connection_data = {
                    "user_id": user_id,
                    "session_id": session.get("session_id", str(uuid.uuid4())),
                    "browser_id": session.get("browser_id"),
                    "ip_address": session.get("ip_address"),
                    "status": session.get("status"),
                    "room_id": session.get("room_id"),
                    "connected_at": session.get("connected_at", datetime.utcnow()),
                    "last_activity": datetime.utcnow(),
                    "sync_timestamp": datetime.utcnow()
                }
                await app.state.connections_collection.insert_one(connection_data)
                results["connections_created"] += 1
            except Exception as e:
                results["errors"].append(f"Error syncing connection for {user_id}: {e}")
        
        for room_id, room_info in manager.active_rooms.items():
            try:
                room_data = {
                    "room_id": room_id,
                    "participants": room_info.get("participants", []),
                    "created_at": room_info.get("created_at", datetime.utcnow()),
                    "updated_at": datetime.utcnow(),
                    "sync_timestamp": datetime.utcnow()
                }
                await app.state.rooms_collection.update_one(
                    {"room_id": room_id},
                    {"$set": room_data},
                    upsert=True
                )
                results["rooms_updated"] += 1
            except Exception as e:
                results["errors"].append(f"Error syncing room {room_id}: {e}")
        
        return results
    return await safe_db_operation(operation, {"error": "Database operation failed"}, "sync_user_data_to_db")

async def periodic_sync():
    while True:
        try:
            await sync_user_data_to_db()
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Periodic sync error: {e}")
            await asyncio.sleep(300)

async def periodic_cleanup():
    while True:
        try:
            await cleanup_expired_actions()
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(3600)

async def cleanup_expired_actions():
    async def operation():
        now = datetime.utcnow()
        
        # ===== Warnings =====
        try:
            if hasattr(app.state, 'warnings_collection') and app.state.warnings_collection is not None:
                result = await app.state.warnings_collection.delete_many({"expires_at": {"$lt": now}})
                if result.deleted_count > 0:
                    logger.info(f"[CLEANUP] Deleted {result.deleted_count} expired warnings")
        except Exception as e:
            logger.error(f"[CLEANUP] warnings error: {e}")
        
        # ===== Mutes =====
        try:
            if hasattr(app.state, 'mutes_collection') and app.state.mutes_collection is not None:
                result = await app.state.mutes_collection.delete_many({"expires_at": {"$lt": now}})
                if result.deleted_count > 0:
                    logger.info(f"[CLEANUP] Deleted {result.deleted_count} expired mutes")
        except Exception as e:
            logger.error(f"[CLEANUP] mutes error: {e}")
        
        # ===== Blocked users =====
        try:
            if hasattr(app.state, 'blocked_users_collection') and app.state.blocked_users_collection is not None:
                result = await app.state.blocked_users_collection.delete_many({
                    "expires_at": {"$lt": now},
                    "is_permanent": {"$ne": True}
                })
                if result.deleted_count > 0:
                    logger.info(f"[CLEANUP] Deleted {result.deleted_count} expired user bans")
        except Exception as e:
            logger.error(f"[CLEANUP] blocked_users error: {e}")
        
        # ===== Blocked IPs =====
        try:
            if hasattr(app.state, 'blocked_ips_collection') and app.state.blocked_ips_collection is not None:
                result = await app.state.blocked_ips_collection.delete_many({
                    "expires_at": {"$lt": now},
                    "is_permanent": {"$ne": True}
                })
                if result.deleted_count > 0:
                    logger.info(f"[CLEANUP] Deleted {result.deleted_count} expired IP bans")
        except Exception as e:
            logger.error(f"[CLEANUP] blocked_ips error: {e}")
        
        # ===== Blocked browsers =====
        try:
            if hasattr(app.state, 'blocked_browsers_collection') and app.state.blocked_browsers_collection is not None:
                result = await app.state.blocked_browsers_collection.delete_many({
                    "expires_at": {"$lt": now},
                    "is_permanent": {"$ne": True}
                })
                if result.deleted_count > 0:
                    logger.info(f"[CLEANUP] Deleted {result.deleted_count} expired browser bans")
        except Exception as e:
            logger.error(f"[CLEANUP] blocked_browsers error: {e}")
        
        # ===== Blocked fingerprints =====
        try:
            if hasattr(app.state, 'blocked_fingerprints_collection') and app.state.blocked_fingerprints_collection is not None:
                result = await app.state.blocked_fingerprints_collection.delete_many({
                    "expires_at": {"$lt": now},
                    "is_permanent": {"$ne": True}
                })
                if result.deleted_count > 0:
                    logger.info(f"[CLEANUP] Deleted {result.deleted_count} expired fingerprint bans")
        except Exception as e:
            logger.error(f"[CLEANUP] blocked_fingerprints error: {e}")
        
        # ===== Blocked subnets =====
        try:
            if hasattr(app.state, 'blocked_subnets_collection') and app.state.blocked_subnets_collection is not None:
                result = await app.state.blocked_subnets_collection.delete_many({
                    "expires_at": {"$lt": now},
                    "is_permanent": {"$ne": True}
                })
                if result.deleted_count > 0:
                    logger.info(f"[CLEANUP] Deleted {result.deleted_count} expired subnet bans")
        except Exception as e:
            logger.error(f"[CLEANUP] blocked_subnets error: {e}")
        
        logger.info("✅ Cleanup completed for all moderation actions")
    
    await safe_db_operation(operation, None, "cleanup_expired_actions")

def get_system_stats():
    try:
        # ===== Uptime =====
        if hasattr(app, 'startup_time'):
            uptime_seconds = time.time() - app.startup_time
        elif hasattr(app.state, 'startup_time'):
            uptime_seconds = time.time() - app.state.startup_time
        else:
            uptime_seconds = 0
            
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        
        if hours > 0:
            uptime_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            uptime_str = f"{minutes}m {seconds}s"
        else:
            uptime_str = f"{seconds}s"
        
        # ===== Процесс =====
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = round(memory_info.rss / 1024 / 1024, 1)
        except Exception as e:
            logger.warning(f"psutil process error: {e}")
            memory_usage_mb = 0
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except:
            cpu_percent = 0
        
        # ===== Системная память =====
        try:
            system_memory = psutil.virtual_memory()
            system_memory_used = round(system_memory.used / 1024 / 1024, 1)
            system_memory_total = round(system_memory.total / 1024 / 1024, 1)
            system_memory_percent = system_memory.percent
        except:
            system_memory_used = 0
            system_memory_total = 0
            system_memory_percent = 0
        
        # ===== Диск =====
        try:
            disk_usage = psutil.disk_usage('/')
            disk_used = round(disk_usage.used / 1024 / 1024 / 1024, 1)
            disk_total = round(disk_usage.total / 1024 / 1024 / 1024, 1)
            disk_percent = disk_usage.percent
        except:
            disk_used = 0
            disk_total = 0
            disk_percent = 0
        
        return {
            "uptime": uptime_str,
            "uptime_seconds": uptime_seconds,
            "memory_usage": memory_usage_mb,
            "cpu_usage": cpu_percent,
            "system_memory_used": system_memory_used,
            "system_memory_total": system_memory_total,
            "system_memory_percent": system_memory_percent,
            "disk_used": disk_used,
            "disk_total": disk_total,
            "disk_percent": disk_percent,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"get_system_stats error: {e}")
        return {
            "uptime": "0s",
            "uptime_seconds": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
            "system_memory_used": 0,
            "system_memory_total": 0,
            "system_memory_percent": 0,
            "disk_used": 0,
            "disk_total": 0,
            "disk_percent": 0,
            "error": str(e)
        }

# ============================================================
# ПРЕМИУМ ФУНКЦИИ
# ============================================================
async def get_premium_settings() -> dict:
    default_settings = {
        "premium_price": 499.0,
        "premium_duration_days": 30,
        "disable_ads": True,
        "priority_support": True,
        "extended_session_time": 3600
    }
    if hasattr(app.state, 'premium_settings_collection') and app.state.premium_settings_collection:
        settings_doc = await app.state.premium_settings_collection.find_one({"key": "premium_settings"})
        if settings_doc:
            return {
                "premium_price": settings_doc.get("premium_price", 499.0),
                "premium_duration_days": settings_doc.get("premium_duration_days", 30),
                "disable_ads": settings_doc.get("disable_ads", True),
                "priority_support": settings_doc.get("priority_support", True),
                "extended_session_time": settings_doc.get("extended_session_time", 3600)
            }
    return default_settings

async def activate_premium(user_id: str, duration_days: int, order_id: str, source: str):
    print(f"👑 [ACTIVATE-PREMIUM] Начало: user={user_id}, days={duration_days}, order={order_id}")
    
    try:
        end_date = datetime.utcnow() + timedelta(days=duration_days)
        premium_data = {
            "user_id": user_id,
            "is_active": True,
            "start_date": datetime.utcnow(),
            "end_date": end_date,
            "order_id": order_id,
            "duration_days": int(duration_days),
            "source": source,
            "created_at": datetime.utcnow()
        }
        
        collection = getattr(app.state, 'premium_collection', None)
        if collection is not None:
            # Деактивируем старые
            try:
                await collection.update_many(
                    {"user_id": user_id, "is_active": True},
                    {"$set": {"is_active": False, "replaced_by": order_id, "deactivated_at": datetime.utcnow()}}
                )
                print(f"👑 [ACTIVATE-PREMIUM] Старые деактивированы")
            except Exception as e:
                print(f"⚠️ [ACTIVATE-PREMIUM] Ошибка деактивации старых: {e}")
            
            # Вставляем новый
            try:
                result = await collection.insert_one(premium_data)
                print(f"✅ [ACTIVATE-PREMIUM] Вставлено, id={result.inserted_id}")
            except Exception as e:
                print(f"❌ [ACTIVATE-PREMIUM] Ошибка вставки: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"❌ [ACTIVATE-PREMIUM] Коллекция = None!")
            raise Exception("premium_collection is None")
        
        # Обновляем video_orders (не критично если упадёт)
        try:
            orders_collection = getattr(app.state, 'video_orders_collection', None)
            if orders_collection is not None:
                await orders_collection.update_one(
                    {"order_id": order_id},
                    {"$set": {"status": "completed", "type": "premium", "paid_at": datetime.utcnow()}}
                )
        except Exception as e:
            print(f"⚠️ [ACTIVATE-PREMIUM] Не удалось обновить orders: {e}")
        
        # Уведомляем пользователя
        try:
            if user_id in manager.active_connections:
                await manager.send_personal_message({
                    "type": "premium_activated",
                    "duration_days": int(duration_days),
                    "end_date": end_date.isoformat(),
                    "message": f"✅ Премиум активирован на {duration_days} дней! Реклама отключена."
                }, user_id)
        except Exception as e:
            print(f"⚠️ [ACTIVATE-PREMIUM] Не удалось уведомить: {e}")
        
        print(f"✅ [ACTIVATE-PREMIUM] Успех")
        return True
    except Exception as e:
        print(f"❌ [ACTIVATE-PREMIUM] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise

async def is_premium_user(user_id: str) -> bool:
    if hasattr(app.state, 'premium_collection') and app.state.premium_collection:
        premium = await app.state.premium_collection.find_one({
            "user_id": user_id,
            "is_active": True,
            "$or": [
                {"end_date": {"$gt": datetime.utcnow()}},
                {"end_date": None}
            ]
        })
        return premium is not None
    return False

# ============================================================
# ОПЛАТА
# ============================================================
async def test_morune_connection():
    try:
        print("🔗 [PAYMENT] Testing connection to Morune...")
        if PAYMENT_CONFIG.TEST_MODE:
            return {
                "status": "success",
                "message": "✅ Тестовый режим активирован",
                "mode": "test",
                "shop_id": PAYMENT_CONFIG.MORUNE_SHOP_ID,
                "test_key": True
            }
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {PAYMENT_CONFIG.MORUNE_API_KEY}",
                "Content-Type": "application/json"
            }
            async with session.get(
                f"{PAYMENT_CONFIG.MORUNE_PAYMENT_URL}/health",
                headers=headers,
                timeout=10
            ) as response:
                if response.status in [200, 204]:
                    return {
                        "status": "success",
                        "message": "✅ Успешное подключение к Morune API",
                        "mode": "live",
                        "api_status": "active"
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"❌ Ошибка подключения: HTTP {response.status}",
                        "mode": "live"
                    }
    except asyncio.TimeoutError:
        return {"status": "error", "message": "❌ Таймаут подключения к Morune"}
    except Exception as e:
        return {"status": "error", "message": f"❌ Ошибка: {str(e)}"}

def generate_morune_signature(shop_id: str, amount: float, currency: str, order_id: str) -> str:
    sign_string = f"{shop_id}:{amount}:{currency}:{order_id}:{PAYMENT_CONFIG.MORUNE_SECRET_KEY}"
    signature = hmac.new(
        PAYMENT_CONFIG.MORUNE_SECRET_KEY.encode('utf-8'),
        sign_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

async def create_morune_payment(order_data: dict) -> dict:
    try:
        if not PAYMENT_CONFIG.MORUNE_SHOP_ID:
            return {"status": "error", "message": "Shop ID is not configured"}
        
        signature = generate_morune_signature(
            shop_id=PAYMENT_CONFIG.MORUNE_SHOP_ID,
            amount=order_data["amount"],
            currency=order_data["currency"],
            order_id=order_data["order_id"]
        )
        order_data["signature"] = signature
        
        order_data["success_url"] = f"{PAYMENT_CONFIG.BACKEND_URL}/api/payment/success?order_id={order_data['order_id']}"
        order_data["fail_url"] = f"{PAYMENT_CONFIG.BACKEND_URL}/api/payment/fail?order_id={order_data['order_id']}"
        order_data["callback_url"] = f"{PAYMENT_CONFIG.BACKEND_URL}/api/payment/callback"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {PAYMENT_CONFIG.MORUNE_API_KEY}",
                "Content-Type": "application/json"
            }
            async with session.post(
                f"{PAYMENT_CONFIG.MORUNE_PAYMENT_URL}/payments/create",
                json=order_data,
                headers=headers,
                timeout=30
            ) as response:
                if response.status == 200:
                    payment_response = await response.json()
                    return {
                        "status": "success",
                        "payment_url": payment_response.get("payment_url") or payment_response.get("url"),
                        "order_id": order_data["order_id"],
                        "payment_id": payment_response.get("payment_id")
                    }
                elif response.status in [201, 204]:
                    payment_url = response.headers.get("Location") or response.headers.get("X-Payment-Url")
                    return {
                        "status": "success",
                        "payment_url": payment_url or f"https://morune.com/payment/{order_data['order_id']}",
                        "order_id": order_data["order_id"]
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "message": f"HTTP {response.status}: {error_text[:200]}",
                        "order_id": order_data["order_id"]
                    }
    except Exception as e:
        print(f"❌ [PAYMENT] Error creating payment: {e}")
        return {"status": "error", "message": str(e)}


    

    async def send_or_queue_notification(user_id: str, notification: dict):
        try:
            if user_id in manager.active_connections:
                await manager.send_personal_message(notification, user_id)
                return True

            if hasattr(app.state, 'pending_notifications_collection') and app.state.pending_notifications_collection is not None:
                await app.state.pending_notifications_collection.insert_one({
                    "user_id": user_id,
                    "type": notification.get("type", "info"),
                    "icon": notification.get("icon", "ℹ️"),
                    "title": notification.get("title", ""),
                    "message": notification.get("message", ""),
                    "submessage": notification.get("submessage", ""),
                    "timeout": notification.get("timeout", 8000),
                    "created_at": datetime.utcnow(),
                    "delivered": False
                })
                print(f"📬 [NOTIFY] Queued for offline user: {user_id}")
                return True

            return False
        except Exception as e:
            print(f"⚠️ [NOTIFY] Error: {e}")
            return False

# ============================================================
# УПРАВЛЕНИЕ WEBSOCKET СОЕДИНЕНИЯМИ
# ============================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, dict] = {}
        self.search_queues: Dict[str, List[str]] = {
            'all': [], 'dating': [], 'friends': [], 'fun': [], 'advice': [], 'games': []
        }
        self.active_rooms: Dict[str, dict] = {}
        self.room_monitors: Dict[str, List[WebSocket]] = {}
        self.lock = asyncio.Lock()
        self.connection_attempts: Dict[str, List[float]] = {}
        self.browser_sessions: Dict[str, str] = {}
        self.sessions: Dict[str, dict] = {}
        self.user_ips: Dict[str, str] = {}
        self.ip_info_cache: Dict[str, dict] = {}
        self.user_rooms: Dict[str, str] = {}
        self.active_audio_monitoring: Dict[str, Dict] = {}
        
        # ВИДЕО-ФУНКЦИИ
        self.video_mode_users: Dict[str, dict] = {}
        self.video_orders: Dict[str, dict] = {}
        self.pending_payments = {}
        self.video_tariffs = []
        self.video_mode_enabled: bool = True
        self.payment_check_interval = 300
        self.payment_timeout = 3600
        
        # ВИДЕО-КОМНАТЫ
        self.video_rooms: Dict[str, dict] = {}
        self.video_search_queues: Dict[str, List[str]] = {
            'single': [],
            'group6': []
        }
        
        if not self.video_tariffs:
            self.video_tariffs = self.get_default_tariffs()

    def get_default_tariffs(self):
        return [
            {"id": "tariff_1", "name": "15 минут", "minutes": 15, "price": 100.0, "is_active": True, "description": "15 минут видео-общения", "created_by": "system"},
            {"id": "tariff_2", "name": "30 минут", "minutes": 30, "price": 180.0, "is_active": True, "description": "30 минут видео-общения", "created_by": "system"},
            {"id": "tariff_3", "name": "60 минут", "minutes": 60, "price": 300.0, "is_active": True, "description": "60 минут видео-общения", "created_by": "system"},
            {"id": "tariff_4", "name": "Навсегда", "minutes": 0, "price": 1000.0, "is_active": True, "description": "Бессрочный видео-режим", "created_by": "system"}
        ]

    # ===== ВИДЕО-ФУНКЦИИ =====
    async def add_video_search_user(self, user_id: str, mode: str, room_id: str):
        if mode == 'single' or mode == '1on1':
            if user_id not in self.video_search_queues['single']:
                self.video_search_queues['single'].append(user_id)
                print(f"🎥 [VIDEO-SEARCH] User {user_id} added to single search queue")
        elif mode == 'group6' or mode == 'group' or mode == '6':
            if user_id not in self.video_search_queues['group6']:
                self.video_search_queues['group6'].append(user_id)
                print(f"🎥 [VIDEO-SEARCH] User {user_id} added to group6 search queue")
        else:
            print(f"⚠️ [VIDEO-SEARCH] Unknown mode: {mode}")
    
    async def remove_video_search_user(self, user_id: str, mode: str = None):
        if mode is None or mode == 'single' or mode == '1on1':
            if user_id in self.video_search_queues['single']:
                self.video_search_queues['single'].remove(user_id)
                print(f"🎥 [VIDEO-SEARCH] User {user_id} removed from single queue")
        if mode is None or mode == 'group6' or mode == 'group' or mode == '6':
            if user_id in self.video_search_queues['group6']:
                self.video_search_queues['group6'].remove(user_id)
                print(f"🎥 [VIDEO-SEARCH] User {user_id} removed from group6 queue")
    
    async def find_video_match(self, user_id: str, mode: str):
        if mode == 'single' or mode == '1on1':
            queue = self.video_search_queues['single']
            for candidate in queue:
                if candidate != user_id:
                    if candidate in self.user_sessions:
                        session = self.user_sessions[candidate]
                        if session.get('status') == UserStatus.SEARCHING:
                            return candidate
            return None
        elif mode == 'group6' or mode == 'group' or mode == '6':
            queue = self.video_search_queues['group6']
            partners = []
            for candidate in queue:
                if candidate != user_id and len(partners) < 5:
                    if candidate in self.user_sessions:
                        session = self.user_sessions[candidate]
                        if session.get('status') == UserStatus.SEARCHING:
                            partners.append(candidate)
            return partners if len(partners) > 0 else None
        return None
    
    async def create_video_room(self, user_id: str, mode: str, partners: list) -> str:
        room_id = str(uuid.uuid4())
        participants = [user_id] + (partners if isinstance(partners, list) else [partners])
        
        self.video_rooms[room_id] = {
            "room_id": room_id,
            "mode": mode,
            "participants": participants,
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True
        }
        
        print(f"🎥 [VIDEO-ROOM] Created video room {room_id} with {len(participants)} participants, mode: {mode}")
        
        for pid in participants:
            if pid in self.user_sessions:
                self.user_sessions[pid]["status"] = UserStatus.CONNECTED
                self.user_sessions[pid]["room_id"] = room_id
        
        await self.remove_video_search_user(user_id, mode)
        for partner in (partners if isinstance(partners, list) else [partners]):
            await self.remove_video_search_user(partner, mode)
        
        return room_id
    
    async def get_video_room_participants(self, room_id: str) -> List[str]:
        if room_id in self.video_rooms:
            return self.video_rooms[room_id].get("participants", [])
        return []
    
    async def get_video_room_info(self, room_id: str) -> Optional[dict]:
        return self.video_rooms.get(room_id)
    
    async def leave_video_room(self, user_id: str, room_id: str):
        if room_id in self.video_rooms:
            room = self.video_rooms[room_id]
            if user_id in room["participants"]:
                room["participants"].remove(user_id)
                print(f"🎥 [VIDEO-ROOM] User {user_id} left room {room_id}")
                
                if user_id in self.user_sessions:
                    self.user_sessions[user_id]["status"] = UserStatus.IDLE
                    self.user_sessions[user_id]["room_id"] = None
                
                if len(room["participants"]) == 0:
                    del self.video_rooms[room_id]
                    print(f"🎥 [VIDEO-ROOM] Room {room_id} deleted (empty)")
                else:
                    for pid in room["participants"]:
                        if pid in self.active_connections:
                            await self.send_personal_message({
                                "type": "video_participant_left",
                                "participant_id": user_id,
                                "room_id": room_id,
                                "participants_left": len(room["participants"])
                            }, pid)
    
    async def broadcast_to_video_room(self, room_id: str, message: dict, exclude_user: str = None):
        if room_id in self.video_rooms:
            for participant in self.video_rooms[room_id]["participants"]:
                if participant != exclude_user and participant in self.active_connections:
                    await self.send_personal_message(message, participant)
    
    async def find_match_for_video_user(self, user_id: str, mode: str, room_id: str):
        max_attempts = 120
        attempt = 0
        
        while attempt < max_attempts:
            if room_id not in self.video_rooms:
                print(f"🎥 [VIDEO] Room {room_id} no longer exists")
                break
            
            room = self.video_rooms[room_id]
            target = 2 if mode == 'single' else 6
            
            if len(room["participants"]) >= target:
                print(f"🎥 [VIDEO] Room {room_id} already full ({len(room['participants'])} participants)")
                break
            
            if user_id not in room["participants"]:
                print(f"🎥 [VIDEO] User {user_id} left room {room_id}")
                break
            
            compatible_partners = await self.find_compatible_video_partners(user_id, mode, room_id)
            
            if compatible_partners:
                for partner in compatible_partners:
                    queue_key = 'single' if mode == 'single' else 'group6'
                    if partner in self.video_search_queues[queue_key]:
                        self.video_search_queues[queue_key].remove(partner)
                    
                    if partner not in room["participants"]:
                        room["participants"].append(partner)
                        if partner in self.user_sessions:
                            self.user_sessions[partner]["status"] = UserStatus.CONNECTED
                            self.user_sessions[partner]["room_id"] = room_id
                        
                        if partner in self.active_connections:
                            await self.send_personal_message({
                                "type": "video_participant_joined",
                                "participant_id": partner,
                                "participant_name": self.user_sessions.get(partner, {}).get("user_id", "User"),
                                "room_id": room_id,
                                "participants_count": len(room["participants"])
                            }, partner)
                            
                            await self.send_personal_message({
                                "type": "video_room_participants",
                                "participants": room["participants"],
                                "room_id": room_id,
                                "mode": room.get("mode", "single")
                            }, partner)
                
                await self.broadcast_to_video_room(room_id, {
                    "type": "video_room_updated",
                    "participants": room["participants"],
                    "participants_count": len(room["participants"]),
                    "room_id": room_id
                })
                
                if len(room["participants"]) >= target:
                    print(f"✅ [VIDEO] Room {room_id} is full with {len(room['participants'])} participants")
                    await self.broadcast_to_video_room(room_id, {
                        "type": "video_room_ready",
                        "room_id": room_id,
                        "participants": room["participants"],
                        "message": "Комната готова! Все участники собраны."
                    })
                    break
            
            attempt += 1
            await asyncio.sleep(1)
        
        if room_id in self.video_rooms:
            room = self.video_rooms[room_id]
            target = 2 if mode == 'single' else 6
            if len(room["participants"]) < target:
                print(f"⏰ [VIDEO] Search timeout for room {room_id}, found {len(room['participants'])} participants")
                await self.broadcast_to_video_room(room_id, {
                    "type": "video_search_timeout",
                    "room_id": room_id,
                    "participants_found": len(room["participants"]),
                    "message": "Поиск завершен. Участников недостаточно."
                })
    
    async def find_compatible_video_partners(self, user_id: str, mode: str, room_id: str) -> List[str]:
        if mode == 'single':
            queue = self.video_search_queues['single']
            compatible = []
            for candidate in queue:
                if candidate != user_id and len(compatible) < 1:
                    if candidate in self.user_sessions:
                        session = self.user_sessions[candidate]
                        if session.get('status') == UserStatus.SEARCHING:
                            compatible.append(candidate)
            return compatible
        
        elif mode == 'group6':
            queue = self.video_search_queues['group6']
            compatible = []
            for candidate in queue:
                if candidate != user_id and len(compatible) < 5:
                    if candidate in self.user_sessions:
                        session = self.user_sessions[candidate]
                        if session.get('status') == UserStatus.SEARCHING:
                            compatible.append(candidate)
            return compatible
        
        return []

    # ===== СТАНДАРТНЫЕ ФУНКЦИИ =====
    async def check_pending_payments(self):
        if not hasattr(self, 'pending_payments'):
            return
        expired_payments = []
        now = time.time()
        for user_id, payment_data in self.pending_payments.items():
            if now - payment_data.get('created_at', now) > self.payment_timeout:
                expired_payments.append(user_id)
                if user_id in self.active_connections:
                    await self.send_personal_message({
                        "type": "payment_expired",
                        "order_id": payment_data.get('order_id'),
                        "message": "Время на оплату истекло. Пожалуйста, создайте новый заказ."
                    }, user_id)
        for user_id in expired_payments:
            del self.pending_payments[user_id]

    async def check_pending_payments_periodically(self):
        while True:
            try:
                await self.check_pending_payments()
                await asyncio.sleep(self.payment_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️ [PAYMENT] Error in periodic check: {e}")
                await asyncio.sleep(60)

    async def add_pending_payment(self, user_id: str, payment_data: dict):
        if not hasattr(self, 'pending_payments'):
            self.pending_payments = {}
        complete_payment_data = {
            'order_id': payment_data.get('order_id'),
            'user_id': user_id,
            'price': payment_data.get('price', 100.0),
            'tariff_minutes': payment_data.get('tariff_minutes', 15),
            'room_id': payment_data.get('room_id'),
            'partner_id': payment_data.get('partner_id'),
            'created_at': time.time(),
            'last_check': time.time(),
            'status': payment_data.get('status', 'waiting_payment')
        }
        for key, value in payment_data.items():
            if key not in complete_payment_data:
                complete_payment_data[key] = value
        self.pending_payments[user_id] = complete_payment_data
        print(f"💰 [PAYMENT] Added pending payment for {user_id}: {complete_payment_data.get('order_id')}")

    async def remove_pending_payment(self, user_id: str):
        if hasattr(self, 'pending_payments') and user_id in self.pending_payments:
            order_id = self.pending_payments[user_id].get('order_id')
            del self.pending_payments[user_id]
            print(f"✅ [PAYMENT] Removed pending payment for {user_id}: {order_id}")

    async def get_pending_payment(self, user_id: str):
        if hasattr(self, 'pending_payments') and user_id in self.pending_payments:
            return self.pending_payments[user_id]
        return None

    def get_online_stats(self) -> dict:
        try:
            SERVICE_IDS = ['chat', 'moderation', 'admin', 'system']
            real_users = 0
            for user_id in self.active_connections.keys():
                if user_id in SERVICE_IDS:
                    continue
                if user_id.startswith(('chat_', 'mod_', 'admin_')):
                    continue
                real_users += 1
            
            online_users = real_users
            
            unique_browsers = set()
            for session in self.user_sessions.values():
                browser_id = session.get("browser_id")
                if browser_id:
                    unique_browsers.add(browser_id)
            
            active_rooms = len(self.active_rooms)
            searching_users = sum(len(queue) for queue in self.search_queues.values())
            
            return {
                "online_users": online_users,
                "unique_users": len(unique_browsers),
                "active_rooms": active_rooms,
                "searching_users": searching_users,
                "server_status": "online",
                "queues_by_category": {category: len(queue) for category, queue in self.search_queues.items()}
            }
        except Exception as e:
            print(f"❌ Ошибка в get_online_stats: {e}")
            return {
                "online_users": 0,
                "unique_users": 0,
                "active_rooms": 0,
                "searching_users": 0,
                "server_status": "error",
                "queues_by_category": {}
            }

    async def toggle_global_video_mode(self, enabled: bool):
        self.video_mode_enabled = enabled
        await self.broadcast_message({
            "type": "global_video_mode_changed",
            "enabled": enabled,
            "timestamp": datetime.utcnow().isoformat()
        })
        if not enabled:
            for user_id in list(self.video_mode_users.keys()):
                await self.deactivate_video_mode(user_id)

    async def is_video_mode_available(self, user_id: str) -> bool:
        if not self.video_mode_enabled:
            return False
        session = self.user_sessions.get(user_id)
        if not session or session.get("status") != UserStatus.CONNECTED:
            return False
        return True

    async def check_websocket_rate_limit(self, user_id: str) -> bool:
        now = time.time()
        async with self.lock:
            if user_id not in self.connection_attempts:
                self.connection_attempts[user_id] = []
            self.connection_attempts[user_id] = [
                attempt for attempt in self.connection_attempts[user_id] 
                if now - attempt < 60
            ]
            if len(self.connection_attempts[user_id]) >= 10:
                return False
            self.connection_attempts[user_id].append(now)
            return True

    async def connect(self, websocket: WebSocket, user_id: str, browser_id: str = None, ip_address: str = None):
        print(f"🔌 Новое подключение: {user_id}, browser: {browser_id}, IP: {ip_address}")
        
        # ===== ПОЛУЧАЕМ FINGERPRINT ИЗ QUERY PARAMS =====
        fingerprint = None
        try:
            query_params = dict(websocket.query_params)
            fingerprint = query_params.get('fingerprint', None)
            print(f"🔑 [CONNECT] Fingerprint: {fingerprint}")
        except Exception as e:
            print(f"⚠️ [CONNECT] Error parsing query params: {e}")
        
        # ===== ПРОВЕРКА БАНА ПО ВСЕМ ПАРАМЕТРАМ =====
        try:
            ban_check = await is_user_fully_blocked(user_id, ip_address, browser_id, fingerprint)
            if ban_check.get("blocked"):
                ban_type = ban_check.get("ban_type", "unknown")
                ban_reason = ban_check.get("reason", "Ваш аккаунт заблокирован")
                print(f"🚫 [BAN] User {user_id} blocked by {ban_type}: {ban_reason}")
                await websocket.accept()
                await websocket.send_text(json.dumps({
                    "type": "blocked",
                    "reason": ban_reason,
                    "ban_type": ban_type
                }))
                await websocket.close(code=1008, reason=f"Blocked by {ban_type}")
                return
        except Exception as e:
            print(f"⚠️ Error checking ban: {e}")
        
        # ===== RATE LIMIT =====
        if not await self.check_websocket_rate_limit(user_id):
            print(f"🚫 [RATE-LIMIT] User {user_id} exceeded rate limit")
            await websocket.close(code=1008, reason="Rate limit exceeded")
            return
        
        # ===== СОХРАНЯЕМ IP =====
        if ip_address:
            self.user_ips[user_id] = ip_address
        
        # ===== ЗАКРЫВАЕМ ПРЕДЫДУЩУЮ СЕССИЮ ТОГО ЖЕ БРАУЗЕРА =====
        if browser_id and browser_id in self.browser_sessions:
            old_user_id = self.browser_sessions[browser_id]
            if old_user_id in self.active_connections:
                logger.info(f"Closing previous session for browser {browser_id}, user {old_user_id}")
                await self.force_disconnect_user(old_user_id)
        
        # ===== ПРИНИМАЕМ СОЕДИНЕНИЕ =====
        await websocket.accept()
        
        async with self.lock:
            if browser_id:
                self.browser_sessions[browser_id] = user_id
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "browser_id": browser_id,
                "ip_address": ip_address,
                "connected_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
            self.active_connections[user_id] = websocket
            self.user_sessions[user_id] = {
                "user_id": user_id,
                "session_id": session_id,
                "browser_id": browser_id,
                "fingerprint": fingerprint,
                "ip_address": ip_address,
                "status": UserStatus.IDLE,
                "room_id": None,
                "partner_id": None,
                "search_params": None,
                "connected_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
        
        print(f"✅ Пользователь {user_id} успешно подключен. Всего подключений: {len(self.active_connections)}")
        
        # ===== ОТПРАВЛЯЕМ ПОДТВЕРЖДЕНИЕ =====
        await self.send_personal_message({
            "type": "connected",
            "user_id": user_id,
            "session_id": session_id,
            "message": "Подключен к голосовому чату",
            "online_stats": self.get_online_stats()
        }, user_id)
        
        # ===== ПРОВЕРЯЕМ НЕЗАВЕРШЁННЫЕ ПЛАТЕЖИ =====
        await self.check_and_notify_pending_payments(user_id)
        await self.deliver_pending_notifications(user_id)
        
        logger.info(f"User {user_id} connected from {ip_address}")        

    async def check_and_notify_pending_payments(self, user_id: str):
        if user_id in self.pending_payments:
            payment_data = self.pending_payments[user_id]
            if time.time() - payment_data.get('created_at', 0) < self.payment_timeout:
                await self.send_personal_message({
                    "type": "payment_waiting_notification",
                    "order_id": payment_data.get('order_id'),
                    "message": "У вас есть незавершенный платеж. Он доступен для продолжения.",
                    "room_id": payment_data.get('room_id'),
                    "current_room": self.user_sessions.get(user_id, {}).get('room_id'),
                    "can_resume": True
                }, user_id)

    async def deliver_pending_notifications(self, user_id: str):
        """Доставляет накопившиеся офлайн-уведомления при подключении."""
        try:
            if not hasattr(app.state, 'pending_notifications_collection') or app.state.pending_notifications_collection is None:
                return
            pending = await app.state.pending_notifications_collection.find({
                "user_id": user_id,
                "delivered": False
            }).sort("created_at", 1).to_list(50)
            
            if not pending:
                return
            
            print(f"📬 [NOTIFY] Delivering {len(pending)} pending notifications to {user_id}")
            
            for notif in pending:
                await self.send_personal_message({
                    "type": notif.get("type", "info"),
                    "icon": notif.get("icon", "ℹ️"),
                    "title": notif.get("title", ""),
                    "message": notif.get("message", ""),
                    "submessage": notif.get("submessage", ""),
                    "timeout": notif.get("timeout", 8000)
                }, user_id)
                
                await app.state.pending_notifications_collection.update_one(
                    {"_id": notif["_id"]},
                    {"$set": {"delivered": True, "delivered_at": datetime.utcnow()}}
                )
                await asyncio.sleep(0.3)  # Пауза, чтобы уведомления не слипались
        except Exception as e:
            print(f"⚠️ [NOTIFY] deliver error: {e}")                

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            session = self.user_sessions.get(user_id, {})
            browser_id = session.get("browser_id")
            session_id = session.get("session_id")
            
            if browser_id and browser_id in self.browser_sessions:
                del self.browser_sessions[browser_id]
            if session_id and session_id in self.sessions:
                del self.sessions[session_id]
            if user_id in self.user_ips:
                del self.user_ips[user_id]
            
            search_params = session.get("search_params")
            if search_params:
                category = search_params.get("category", "all")
                if user_id in self.search_queues[category]:
                    self.search_queues[category].remove(user_id)
            
            room_id = session.get("room_id")
            partner_id = session.get("partner_id")
            
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
            await self.check_and_notify_pending_payments(partner_id)

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                if user_id in self.user_sessions:
                    self.user_sessions[user_id]["last_activity"] = datetime.utcnow().isoformat()
                    session_id = self.user_sessions[user_id].get("session_id")
                    if session_id and session_id in self.sessions:
                        self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
                await self.active_connections[user_id].send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(user_id)

    async def broadcast_message(self, message: dict):
        disconnected_users = []
        for user_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Error broadcasting to {user_id}: {e}")
                disconnected_users.append(user_id)
        for user_id in disconnected_users:
            self.disconnect(user_id)

    async def force_disconnect_user(self, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.send_personal_message({
                    "type": "force_disconnect",
                    "message": "Вы были отключены администратором",
                    "reason": "admin_action"
                }, user_id)
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
        await self.check_and_notify_pending_payments(user_id)
        asyncio.create_task(self.find_match_for_user(user_id))
        return True

    async def find_match_for_user(self, user_id: str):
        max_attempts = 120
        attempt = 0
        while attempt < max_attempts:
            session = self.user_sessions.get(user_id)
            if not session or session.get("status") != UserStatus.SEARCHING:
                break
            partner_id = await self.find_compatible_partner(user_id)
            if partner_id:
                await self.create_room(user_id, partner_id)
                break
            attempt += 1
            await asyncio.sleep(1)
        else:
            if user_id in self.user_sessions:
                await self.send_personal_message({
                    "type": "search_timeout",
                    "message": "Не удалось найти подходящего собеседника",
                    "online_stats": self.get_online_stats()
                }, user_id)
                await self.stop_search(user_id)

    async def find_compatible_partner(self, user_id: str):
        user_session = self.user_sessions.get(user_id)
        if not user_session or not user_session.get("search_params"):
            return None
        user_params = user_session["search_params"]
        user_category = user_params.get("category", "all")
        user_gender = user_params.get("user_gender")
        search_gender = user_params.get("search_gender")
        # 🆕 Гео и возраст
        user_country = user_params.get("country", "any")
        user_age = user_params.get("user_age", "any")
        search_age = user_params.get("search_age", "any")

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
                partner_country = partner_params.get("country", "any")
                partner_user_age = partner_params.get("user_age", "any")

                if user_country and user_country != "any":
                    if partner_country and partner_country != "any" and partner_country != user_country:
                        continue

                # 🆕 Фильтр по возрасту (взаимный)
                if search_age and search_age != "any":
                    if partner_user_age and partner_user_age != "any" and partner_user_age != search_age:
                        continue
                partner_search_age = partner_params.get("search_age", "any")
                if partner_search_age and partner_search_age != "any":
                    if user_age and user_age != "any" and user_age != partner_search_age:
                        continue

                # Проверка пола
                if self.check_gender_compatibility(user_gender, search_gender, partner_gender, partner_search_gender):
                    return potential_partner_id
        return None

    def check_gender_compatibility(self, user_gender: str, user_search_gender: str, partner_gender: str, partner_search_gender: str) -> bool:
        if user_search_gender == "any" and partner_search_gender == "any":
            return True
        if user_search_gender != "any":
            if user_search_gender != partner_gender:
                return False
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
            await self.update_payment_room_context(user1_id, room_id)
            await self.update_payment_room_context(user2_id, room_id)
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
        await self.check_and_notify_pending_payments(user1_id)
        await self.check_and_notify_pending_payments(user2_id)

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
        search_params = session.get("search_params")
        if search_params:
            await self.start_search(user_id, search_params)
            await self.check_and_notify_pending_payments(user_id)

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

    async def update_user_activity(self, user_id: str):
        if user_id in self.user_sessions:
            self.user_sessions[user_id]["last_activity"] = datetime.utcnow().isoformat()

    async def update_payment_room_context(self, user_id: str, new_room_id: str = None):
        if user_id in self.pending_payments:
            payment_data = self.pending_payments[user_id]
            old_room_id = payment_data.get('room_id')
            if old_room_id != new_room_id and new_room_id:
                print(f"🔄 [PAYMENT] Обновление комнаты для платежа {user_id}: {old_room_id} → {new_room_id}")
                self.pending_payments[user_id]['room_id'] = new_room_id
                if hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection:
                    try:
                        await app.state.video_orders_collection.update_one(
                            {"order_id": payment_data.get('order_id')},
                            {"$set": {"room_id": new_room_id}}
                        )
                    except Exception as e:
                        print(f"⚠️ [PAYMENT] Ошибка обновления комнаты в БД: {e}")

    async def add_room_monitor(self, room_id: str, websocket: WebSocket):
        async with self.lock:
            if room_id not in self.room_monitors:
                self.room_monitors[room_id] = []
            self.room_monitors[room_id].append(websocket)

    async def remove_room_monitor(self, room_id: str, websocket: WebSocket):
        async with self.lock:
            if room_id in self.room_monitors:
                if websocket in self.room_monitors[room_id]:
                    self.room_monitors[room_id].remove(websocket)
                if not self.room_monitors[room_id]:
                    del self.room_monitors[room_id]

    async def broadcast_to_room_monitors(self, room_id: str, message: dict):
        if room_id in self.room_monitors:
            disconnected_monitors = []
            for monitor_ws in self.room_monitors[room_id]:
                try:
                    await monitor_ws.send_text(json.dumps(message, default=str))
                except Exception as e:
                    logger.error(f"Error sending to room monitor: {e}")
                    disconnected_monitors.append(monitor_ws)
            for monitor_ws in disconnected_monitors:
                await self.remove_room_monitor(room_id, monitor_ws)

    async def handle_audio_data(self, user_id: str, room_id: str, audio_chunk: str):
        try:
            session = self.user_sessions.get(user_id)
            if not session or session.get("room_id") != room_id:
                return
            try:
                audio_bytes = base64.b64decode(audio_chunk)
            except Exception as e:
                print(f"⚠️ [AUDIO] Failed to decode audio for {user_id}: {e}")
                return
            volume_level = self.calculate_volume_level(audio_bytes)
            if room_id in self.room_monitors:
                audio_message = {
                    "type": "audio_data",
                    "room_id": room_id,
                    "user_id": user_id,
                    "audio_chunk": audio_chunk,
                    "volume_level": volume_level,
                    "timestamp": datetime.utcnow().isoformat()
                }
                disconnected_monitors = []
                for monitor_ws in self.room_monitors[room_id]:
                    try:
                        await monitor_ws.send_text(json.dumps(audio_message, default=str))
                    except Exception as e:
                        print(f"⚠️ [AUDIO] Failed to send to monitor: {e}")
                        disconnected_monitors.append(monitor_ws)
                for monitor_ws in disconnected_monitors:
                    await self.remove_room_monitor(room_id, monitor_ws)
        except Exception as e:
            print(f"❌ [AUDIO] Error processing audio data: {e}")

    def calculate_volume_level(self, audio_bytes: bytes) -> float:
        try:
            if len(audio_bytes) < 2:
                return 0.0
            rms = audioop.rms(audio_bytes, 2)
            max_rms = 32767.0
            normalized = min(rms / max_rms, 1.0)
            if normalized > 0:
                import math
                log_normalized = math.log10(normalized * 9 + 1)
                return round(log_normalized, 2)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating volume level: {e}")
            return 0.0

    async def load_video_tariffs_from_db(self):
        try:
            print("🔄 [TARIFFS] Loading tariffs from database...")
            default_tariffs = [
                {"id": "tariff_1", "name": "15 минут", "minutes": 15, "price": 100.0, "is_active": True},
                {"id": "tariff_2", "name": "30 минут", "minutes": 30, "price": 180.0, "is_active": True},
                {"id": "tariff_3", "name": "60 минут", "minutes": 60, "price": 300.0, "is_active": True},
                {"id": "tariff_4", "name": "Навсегда", "minutes": 0, "price": 1000.0, "is_active": True}
            ]
            if not hasattr(app.state, 'video_tariffs_collection') or app.state.video_tariffs_collection is None:
                print("⚠️ [TARIFFS] Collection not available, using default tariffs")
                self.video_tariffs = default_tariffs
                return self.video_tariffs
            cursor = app.state.video_tariffs_collection.find({"is_active": True})
            db_tariffs = await cursor.to_list(length=20)
            if db_tariffs and len(db_tariffs) > 0:
                self.video_tariffs = []
                for tariff in db_tariffs:
                    self.video_tariffs.append({
                        "id": str(tariff.get("_id", tariff.get("id", f"t{len(self.video_tariffs)}"))),
                        "name": tariff.get("name", "Без названия"),
                        "minutes": tariff.get("minutes", 15),
                        "price": float(tariff.get("price", 100.0)),
                        "is_active": tariff.get("is_active", True),
                        "description": tariff.get("description", ""),
                        "created_at": tariff.get("created_at"),
                        "created_by": tariff.get("created_by", "system")
                    })
                print(f"✅ [TARIFFS] Loaded {len(self.video_tariffs)} tariffs from DB")
            else:
                print("📭 [TARIFFS] No tariffs in DB, using defaults")
                self.video_tariffs = default_tariffs
            return self.video_tariffs
        except Exception as e:
            print(f"❌ [TARIFFS] Error loading tariffs: {e}")
            self.video_tariffs = self.get_default_tariffs()
            return self.video_tariffs

    def get_monitored_rooms(self) -> List[dict]:
        try:
            monitored_info = []
            for room_id, monitors in self.room_monitors.items():
                try:
                    room_info = self.active_rooms.get(room_id, {})
                    participants = room_info.get("participants", [])
                    participants_info = []
                    for user_id in participants:
                        session = self.user_sessions.get(user_id, {})
                        participants_info.append({
                            "user_id": user_id,
                            "status": session.get("status", "unknown"),
                            "connected_at": session.get("connected_at")
                        })
                    monitored_info.append({
                        "room_id": room_id,
                        "participants": participants_info,
                        "participants_count": len(participants),
                        "monitors_count": len(monitors),
                        "created_at": room_info.get("created_at"),
                        "is_active": room_id in self.active_rooms
                    })
                except Exception as e:
                    print(f"⚠️ Ошибка обработки мониторинга комнаты {room_id}: {e}")
                    continue
            return monitored_info
        except Exception as e:
            print(f"❌ Ошибка в get_monitored_rooms: {e}")
            return []

    def get_all_connections(self):
        connections = []
        SERVICE_IDS = ['chat', 'moderation', 'admin', 'system']
        
        for user_id, session in self.user_sessions.items():
            if user_id in SERVICE_IDS:
                continue
            if user_id.startswith(('chat_', 'mod_', 'admin_')):
                continue
            
            ws_active = user_id in self.active_connections
            ip_info = {}
            ip_address = session.get('ip_address')
            if ip_address and ip_address in self.ip_info_cache:
                ip_info = self.ip_info_cache[ip_address]

            fingerprint = session.get('fingerprint')
            if not fingerprint:
                fingerprint = self.user_sessions.get(user_id, {}).get('fingerprint')    
            
            connection_data = {
                "user_id": user_id,
                "browser_id": session.get("browser_id", "unknown"),
                "fingerprint": fingerprint or "unknown",
                "ip_address": ip_address or "unknown",
                "ip_info": ip_info,
                "status": session.get("status", "unknown"),
                "room_id": session.get("room_id"),
                "partner_id": session.get("partner_id"),
                "ws_active": ws_active,
                "connected_at": session.get("connected_at", ""),
                "last_activity": session.get("last_activity", ""),
                "search_params": session.get("search_params")
            }
            connections.append(connection_data)
        
        connections.sort(key=lambda x: x.get('last_activity', ''), reverse=True)
        return connections

    def get_video_time_left(self, user_id: str):
        if user_id in self.video_mode_users:
            video_data = self.video_mode_users[user_id]
            if video_data.get("minutes", 0) == 0:
                return None
            end_time = video_data.get("end_time")
            if end_time:
                time_left = (end_time - datetime.utcnow()).total_seconds() / 60
                return max(0, int(time_left))
        return 0

    async def activate_video_mode(self, user_id: str, minutes: int):
        end_time = None
        if minutes > 0:
            end_time = datetime.utcnow() + timedelta(minutes=minutes)
        self.video_mode_users[user_id] = {
            "minutes": minutes,
            "activated_at": datetime.utcnow(),
            "end_time": end_time,
            "is_active": True
        }
        await self.send_personal_message({
            "type": "video_mode_activated",
            "minutes": minutes,
            "end_time": end_time.isoformat() if end_time else None
        }, user_id)

    async def deactivate_video_mode(self, user_id: str):
        if user_id in self.video_mode_users:
            del self.video_mode_users[user_id]
            await self.send_personal_message({
                "type": "video_mode_deactivated",
                "message": "Видео-режим деактивирован"
            }, user_id)

    def is_video_mode_active(self, user_id: str) -> bool:
        if user_id in self.video_mode_users:
            video_data = self.video_mode_users[user_id]
            if video_data.get("minutes", 0) == 0:
                return True
            end_time = video_data.get("end_time")
            if end_time and datetime.utcnow() < end_time:
                return True
            else:
                asyncio.create_task(self.deactivate_video_mode(user_id))
                return False
        return False

    def get_video_stats(self) -> dict:
        active_users = 0
        permanent_users = 0
        temporary_users = 0
        for user_id, video_data in self.video_mode_users.items():
            if self.is_video_mode_active(user_id):
                active_users += 1
                if video_data.get("minutes", 0) == 0:
                    permanent_users += 1
                else:
                    temporary_users += 1
        return {
            "video_mode_enabled": self.video_mode_enabled,
            "active_video_users": active_users,
            "permanent_video_users": permanent_users,
            "temporary_video_users": temporary_users,
            "total_tariffs": len(self.video_tariffs),
            "active_tariffs": len([t for t in self.video_tariffs if t.get("is_active", True)])
        }

    async def update_ip_info_cache(self):
        print("🔍 [DEBUG] Updating IP info cache")
        updated_count = 0
        update_tasks = []
        for user_id, ip_address in self.user_ips.items():
            if ip_address and ip_address not in self.ip_info_cache:
                task = asyncio.create_task(self._update_single_ip_info(ip_address))
                update_tasks.append(task)
        if update_tasks:
            try:
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                updated_count = len([r for r in results if not isinstance(r, Exception)])
            except Exception as e:
                print(f"⚠️ [DEBUG] Error gathering IP info tasks: {e}")
        print(f"✅ [DEBUG] IP cache updated: {updated_count} new entries")

    async def _update_single_ip_info(self, ip_address: str):
        try:
            ip_info = await get_ip_info(ip_address)
            if ip_info.get("status") == "success":
                self.ip_info_cache[ip_address] = {
                    "country": ip_info.get("country", "Unknown"),
                    "city": ip_info.get("city", "Unknown"),
                    "isp": ip_info.get("isp", "Unknown"),
                    "query": ip_info.get("query", ip_address)
                }
                return True
            else:
                self.ip_info_cache[ip_address] = {
                    "country": "Unknown",
                    "city": "Unknown",
                    "isp": "Unknown",
                    "query": ip_address
                }
                return False
        except Exception as e:
            print(f"⚠️ [IP] Error updating IP info for {ip_address}: {e}")
            self.ip_info_cache[ip_address] = {
                "country": "Error",
                "city": "Error",
                "isp": "Error",
                "query": ip_address
            }
            return False

    async def clear_search_queues(self, categories: List[str] = None):
        async with self.lock:
            if categories is None:
                categories = list(self.search_queues.keys())
            cleared_count = 0
            for category in categories:
                if category in self.search_queues:
                    for user_id in self.search_queues[category]:
                        if user_id in self.user_sessions:
                            session = self.user_sessions[user_id]
                            session["status"] = UserStatus.IDLE
                            session["search_params"] = None
                            await self.send_personal_message({
                                "type": "search_stopped",
                                "message": "Поиск остановлен администратором",
                                "online_stats": self.get_online_stats()
                            }, user_id)
                    cleared_count += len(self.search_queues[category])
                    self.search_queues[category] = []
            return cleared_count

    async def create_default_tariffs(self) -> List[dict]:
        print("🔄 [TARIFFS] Creating default tariffs...")
        default_tariffs_data = [
            {"id": "tariff_1", "name": "15 минут", "minutes": 15, "price": 100.0, "is_active": True, "description": "15 минут видео-общения", "created_by": "system"},
            {"id": "tariff_2", "name": "30 минут", "minutes": 30, "price": 180.0, "is_active": True, "description": "30 минут видео-общения", "created_by": "system"},
            {"id": "tariff_3", "name": "60 минут", "minutes": 60, "price": 300.0, "is_active": True, "description": "60 минут видео-общения", "created_by": "system"},
            {"id": "tariff_4", "name": "Навсегда", "minutes": 0, "price": 1000.0, "is_active": True, "description": "Бессрочный видео-режим", "created_by": "system"}
        ]
        tariffs_list = []
        now = datetime.utcnow()
        for tariff_data in default_tariffs_data:
            tariff = {**tariff_data, "created_at": now, "updated_at": now}
            tariffs_list.append(tariff)
            if hasattr(app.state, 'video_tariffs_collection') and app.state.video_tariffs_collection is not None:
                try:
                    result = await app.state.video_tariffs_collection.update_one(
                        {"id": tariff["id"]},
                        {"$set": tariff},
                        upsert=True
                    )
                    if result.upserted_id:
                        print(f"✅ [TARIFFS] Created tariff in DB: {tariff['name']}")
                    elif result.modified_count > 0:
                        print(f"✅ [TARIFFS] Updated tariff in DB: {tariff['name']}")
                    else:
                        print(f"✅ [TARIFFS] Tariff already exists: {tariff['name']}")
                except Exception as e:
                    print(f"❌ [TARIFFS] Error saving tariff to DB: {e}")
        self.video_tariffs = tariffs_list
        print(f"✅ [TARIFFS] Created {len(tariffs_list)} default tariffs")
        return tariffs_list

manager = ConnectionManager()

async def start_payment_checking():
    await manager.check_pending_payments_periodically()

async def initialize_video_tariffs_on_startup():
    print("🔄 Инициализация тарифов видео-режима...")
    if not hasattr(manager, 'video_tariffs') or len(manager.video_tariffs) == 0:
        print("🔄 [TARIFFS] Loading tariffs from DB via manager...")
        await manager.load_video_tariffs_from_db()
        logger.info(f"✅ Тарифы загружены: {len(manager.video_tariffs)} шт.")
    else:
        logger.info(f"ℹ️ Тарифы уже загружены: {len(manager.video_tariffs)} шт.")

class ModerationConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        async with self.lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            for connection in disconnected:
                await self.disconnect(connection)

moderation_manager = ModerationConnectionManager()

# ============================================================
# АУТЕНТИФИКАЦИЯ
# ============================================================
security = HTTPBasic()

async def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    print(f"🔐 [AUTH] Attempting login for: {credentials.username}")
    if not hasattr(app.state, 'moderators_collection') or app.state.moderators_collection is None:
        print("❌ [AUTH] Moderators collection not available")
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        moderator = await app.state.moderators_collection.find_one({
            "username": credentials.username,
            "is_active": True
        })
        if moderator:
            if verify_password(credentials.password, moderator["password_hash"]):
                print(f"✅ [AUTH] Successful admin login: {credentials.username}")
                return moderator
            else:
                print(f"❌ [AUTH] Invalid password for: {credentials.username}")
        else:
            print(f"❌ [AUTH] Moderator not found: {credentials.username}")
    except Exception as e:
        print(f"❌ [AUTH] Database error: {e}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

async def authenticate_moderator(credentials: HTTPBasicCredentials = Depends(security)):
    print(f"🔐 [AUTH_MOD] Attempting login for: {credentials.username}")
    if not hasattr(app.state, 'moderators_collection') or app.state.moderators_collection is None:
        print("❌ [AUTH_MOD] Moderators collection is not available")
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        moderator = await app.state.moderators_collection.find_one({
            "username": credentials.username,
            "is_active": True
        })
        if moderator:
            print(f"✅ [AUTH_MOD] Moderator found: {moderator['username']}")
            if verify_password(credentials.password, moderator["password_hash"]):
                print(f"✅ [AUTH_MOD] Password verified for: {credentials.username}")
                return moderator
            else:
                print(f"❌ [AUTH_MOD] Invalid password for: {credentials.username}")
        else:
            print(f"❌ [AUTH_MOD] Moderator not found: {credentials.username}")
    except Exception as e:
        print(f"❌ [AUTH_MOD] Database error: {e}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

# ============================================================
# LIFESPAN (ПЕРЕМЕЩЕН ПОСЛЕ СОЗДАНИЯ APP)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Application starting up...")
    app.state.startup_time = time.time()
    app.startup_time = time.time()
    app.state.system_settings = {
        "maintenance_mode": False,
        "max_users": 1000,
        "allow_registration": True,
        "system_message": "Добро пожаловать в голосовой чат!",
        "muffle_enabled": False,
        "maintenance_message": "Ведутся технические работы. Приносим извинения за временные неудобства.",
        "auto_moderation_enabled": True,
        "max_warnings_before_ban": 3,
        "mute_duration_minutes": 60,
        "ban_duration_hours": 24
    }
    app.state.admin_sessions = {}
    
    try:
        logger.info("🔗 Connecting to MongoDB...")
        client, db, collections = await setup_mongodb_connection()
        app.state.client = client
        app.state.db = db
        
        collections_to_init = [
            "reports_collection", "blocked_users_collection", "moderators_collection",
            "users_collection", "connections_collection", "rooms_collection",
            "warnings_collection", "mutes_collection", "kicks_collection", 
            "moderation_logs_collection", "video_orders_collection",
            "audio_ads_collection", "video_ads_collection", "premium_collection",
            "premium_settings_collection", "chat_messages_collection",
            "system_settings_collection", "video_tariffs_collection",
            "messages_log_collection",
            "blocked_ips_collection",
            "blocked_browsers_collection", 
            "blocked_fingerprints_collection",
            "blocked_subnets_collection",
            "pending_notifications_collection"
        ]
        for collection_name in collections_to_init:
            try:
                setattr(app.state, collection_name, collections.get(collection_name))
                if getattr(app.state, collection_name) is None:
                    logger.warning(f"⚠️ Collection {collection_name} is None")
                else:
                    logger.info(f"✅ Collection {collection_name} initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {collection_name}: {e}")
                setattr(app.state, collection_name, None)
        
        if app.state.moderators_collection is not None:
            await initialize_default_moderator(app.state.moderators_collection)
        
        if app.state.db is not None:
            logger.info("🔄 Performing initial data synchronization...")
            try:
                await sync_user_data_to_db()
                logger.info("✅ Initial data synchronization completed")
                payment_check_task = asyncio.create_task(start_payment_checking())
                tariffs_init_task = asyncio.create_task(initialize_video_tariffs_on_startup())
                logger.info("✅ Фоновые задачи запущены")
                app.state.payment_check_task = payment_check_task
                app.state.tariffs_init_task = tariffs_init_task
            except Exception as e:
                logger.error(f"❌ Initial data synchronization failed: {e}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        logger.info("⚠️ Running in limited mode - data will not persist between restarts")
        app.state.client = None
        app.state.db = None
        app.state.reports_collection = None
        app.state.blocked_users_collection = None
        app.state.moderators_collection = None
        app.state.users_collection = None
        app.state.connections_collection = None
        app.state.rooms_collection = None
        app.state.warnings_collection = None
        app.state.mutes_collection = None
        app.state.kicks_collection = None
        app.state.moderation_logs_collection = None
        app.state.video_orders_collection = None
        app.state.audio_ads_collection = None
        app.state.video_ads_collection = None
        app.state.premium_collection = None
        app.state.premium_settings_collection = None
        app.state.chat_messages_collection = None
        app.state.system_settings_collection = None
        app.state.video_tariffs_collection = None
        app.state.messages_log_collection = None

    try:
        logger.info("💳 Загрузка настроек оплаты из базы данных...")
        if hasattr(app.state, 'system_settings_collection') and app.state.system_settings_collection is not None:
            payment_settings = await app.state.system_settings_collection.find_one({"key": "video_payment_settings"})
            if payment_settings:
                print("=" * 80)
                print("💳 НАЙДЕНЫ СОХРАНЕННЫЕ НАСТРОЙКИ ОПЛАТЫ В БАЗЕ:")
                print(f"   Shop ID: {payment_settings.get('morune_shop_id', 'НЕ УКАЗАН')}")
                print(f"   API Key: {payment_settings.get('morune_api_key', 'НЕ УКАЗАН')}")
                print(f"   Режим: {'ТЕСТОВЫЙ' if payment_settings.get('test_mode', True) else 'РАБОЧИЙ'}")
                print("=" * 80)
                PAYMENT_CONFIG.MORUNE_SHOP_ID = payment_settings.get("morune_shop_id", PAYMENT_CONFIG.MORUNE_SHOP_ID)
                PAYMENT_CONFIG.MORUNE_API_KEY = payment_settings.get("morune_api_key", PAYMENT_CONFIG.MORUNE_API_KEY)
                PAYMENT_CONFIG.MORUNE_SECRET_KEY = payment_settings.get("morune_secret_key", PAYMENT_CONFIG.MORUNE_SECRET_KEY)
                PAYMENT_CONFIG.MORUNE_PAYMENT_URL = payment_settings.get("morune_payment_url", PAYMENT_CONFIG.MORUNE_PAYMENT_URL)
                PAYMENT_CONFIG.TEST_MODE = payment_settings.get("test_mode", PAYMENT_CONFIG.TEST_MODE)
                CONFIG.MORUNE_SHOP_ID = PAYMENT_CONFIG.MORUNE_SHOP_ID
                CONFIG.MORUNE_API_KEY = PAYMENT_CONFIG.MORUNE_API_KEY
                logger.info("✅ Настройки оплаты загружены из базы данных")
            else:
                logger.info("ℹ️ Настроек оплаты в базе нет, используются значения по умолчанию")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки настроек оплаты: {e}")
    
    try:
        sync_task = asyncio.create_task(periodic_sync())
        cleanup_task = asyncio.create_task(periodic_cleanup())
        logger.info("✅ Фоновые задачи запущены")
        yield
    finally:
        logger.info("🛑 Application shutting down...")
        try:
            if hasattr(app.state, 'payment_check_task'):
                app.state.payment_check_task.cancel()
            if hasattr(app.state, 'tariffs_init_task'):
                app.state.tariffs_init_task.cancel()
            await asyncio.gather(
                app.state.payment_check_task,
                app.state.tariffs_init_task,
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"❌ Ошибка остановки фоновых задач: {e}")
        if hasattr(app.state, 'client') and app.state.client:
            app.state.client.close()
            logger.info("🔌 MongoDB connection closed")

# ПРИМЕНЯЕМ LIFESPAN
app.router.lifespan_context = lifespan

# ============================================================
# НАСТРОЙКА CORS И MIDDLEWARE
# ============================================================
limiter = Limiter(key_func=get_remote_address, default_limits=["10000/minute"])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

origins = [
    "https://cosmic-voice.ru",
    "https://cosmic-voice.ru:8000",
    "https://cosmic-voice.ru:3000",
    "https://cosmic-voice.ru:8001",
    "http://192.168.0.103:8000",
    "http://192.168.0.103:3000",
    "http://192.168.0.103:8001",
    "http://localhost:8000",
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization", "X-Recaptcha-Token"],
    expose_headers=["*"]
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    # Пропускаем WebSocket
    if request.url.path.startswith("/ws/"):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "600"
        return response
    
    response = await call_next(request)
    
    if request.method == "OPTIONS":
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, X-Requested-With"
        response.headers["Access-Control-Max-Age"] = "600"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        if "origin" in request.headers:
            response.headers["Access-Control-Allow-Origin"] = request.headers["origin"]
        return response
    
    if "origin" in request.headers:
        response.headers["Access-Control-Allow-Origin"] = request.headers["origin"]
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "microphone=(), camera=()"
    
    return response

@app.middleware("http")
async def admin_cors_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/admin") or request.url.path.startswith("/moderation"):
        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please try again later."},
        headers={"Retry-After": "60"}
    )

    

# ============================================================
# СОЗДАНИЕ РОУТЕРОВ
# ============================================================
api_router = APIRouter(prefix="/api")
admin_router = APIRouter(prefix="/admin")
moderation_router = APIRouter(prefix="/moderation")

# ============================================================
# API ЭНДПОИНТЫ (все из оригинала, с исправлениями)
# ============================================================
@api_router.post("/report")
async def submit_report(request: ReportRequest):
    if await check_report_spam(request.user_id):
        raise HTTPException(status_code=429, detail="Too many reports. Please try again later.")
    
    async def check_duplicate_report():
        existing_report = await app.state.reports_collection.find_one({
            "reporter_id": request.user_id,
            "reported_user_id": request.partner_id,
            "resolved": {"$ne": True}
        })
        return existing_report is not None
    
    has_reported_before = await safe_db_operation(check_duplicate_report, False, "check_duplicate_report")
    if has_reported_before:
        raise HTTPException(status_code=409, detail="Вы уже отправляли жалобу на этого пользователя.")
    
    reported_user_session = manager.user_sessions.get(request.partner_id)
    if not reported_user_session:
        raise HTTPException(status_code=404, detail="Reported user not found")
    
    # 🆕 ОПРЕДЕЛЯЕМ report_data прямо здесь (на уровне submit_report)
    report_data = {
        "reporter_id": request.user_id,
        "reported_user_id": request.partner_id,
        "room_id": request.room_id,
        "reason": request.reason,
        "description": request.description,
        "timestamp": datetime.utcnow(),
        "ip_address": manager.user_ips.get(request.user_id, "unknown"),
        "resolved": False,
        "resolved_by": None,
        "resolved_at": None,
        "moderator_action": None,
        "moderator_notes": None,
        "reporter_notified": False,
        "reported_notified": False
    }
    
    # 🆕 Сохраняем отчёт в БД
    try:
        result = await app.state.reports_collection.insert_one(report_data)
    except Exception as e:
        logger.error(f"[REPORT] Insert error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save report: {e}")
    
    # 🆕 УВЕДОМЛЯЕМ ТОГО, НА КОГО ЖАЛУЮТСЯ
    reported_user_id = request.partner_id
    reason_text = translate_reason(request.reason)
    
    if reported_user_id in manager.active_connections:
        await manager.send_personal_message({
            "type": "warning",
            "icon": "⚠️",
            "title": "На вас поступила жалоба",
            "message": f"Причина: {reason_text}",
            "submessage": "Модераторы рассмотрят её в ближайшее время",
            "timeout": 10000
        }, reported_user_id)
        await app.state.reports_collection.update_one(
            {"_id": result.inserted_id},
            {"$set": {"reported_notified": True}}
        )
    
    # 🆕 УВЕДОМЛЯЕМ ТОГО, КТО ПОДАЛ ЖАЛОБУ
    reporter_id = request.user_id
    if reporter_id in manager.active_connections:
        await manager.send_personal_message({
            "type": "report_accepted",
            "icon": "✅",
            "title": "Жалоба принята",
            "message": f"Ваша жалоба на пользователя {reported_user_id[:8]}... принята",
            "submessage": f"Причина: {reason_text}",
            "timeout": 6000
        }, reporter_id)
        await app.state.reports_collection.update_one(
            {"_id": result.inserted_id},
            {"$set": {"reporter_notified": True}}
        )
    
    # 🆕 Уведомляем модераторов
    await moderation_manager.broadcast({
        "type": "new_report",
        "report_id": str(result.inserted_id),
        "reporter_id": request.user_id,
        "reported_id": request.partner_id,
        "reason": request.reason
    })
    
    # 🆕 Автобан при превышении порога
    total_reports = await get_user_report_count(request.partner_id)
    if total_reports >= MODERATOR_NOTIFICATION_THRESHOLD:
        await notify_moderators(f"Пользователь {request.partner_id} получил {total_reports} жалоб.")
    if total_reports >= AUTO_BLOCK_THRESHOLD:
        await block_user(request.partner_id, f"Автоматическая блокировка после {total_reports} жалоб", 24)
        if request.partner_id in manager.active_connections:
            await manager.force_disconnect_user(request.partner_id)
        return {"status": "success", "message": "Отправлен отчет. Пользователь заблокирован.", "user_blocked": True}
    
    return {"status": "success", "message": "Отчет успешно отправлен", "user_blocked": False}

@api_router.post("/create-video-order")
async def create_video_order(request: dict):
    try:
        print("💰 [PAYMENT] Creating video order...")
        user_id = request.get("user_id")
        tariff_minutes = request.get("tariff_minutes")
        price = request.get("price")
        room_id = request.get("room_id")
        partner_id = request.get("partner_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        if tariff_minutes is None:
            raise HTTPException(status_code=400, detail="Tariff minutes is required")
        if price is None:
            raise HTTPException(status_code=400, detail="Price is required")
        order_id = f"video_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        order_data = {
            "order_id": order_id,
            "user_id": user_id,
            "partner_id": partner_id,
            "room_id": room_id,
            "tariff_minutes": tariff_minutes,
            "tariff_name": f"{tariff_minutes} минут" if tariff_minutes > 0 else "Навсегда",
            "price": float(price),
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        video_orders_collection = getattr(app.state, 'video_orders_collection', None)
        if video_orders_collection is not None:
            try:
                await video_orders_collection.insert_one(order_data)
                print(f"✅ [PAYMENT] Order saved to database: {order_id}")
            except Exception as e:
                print(f"❌ [PAYMENT] Error saving order to DB: {e}")
        if PAYMENT_CONFIG.TEST_MODE:
            if video_orders_collection is not None:
                try:
                    await video_orders_collection.update_one(
                        {"order_id": order_id},
                        {"$set": {"status": "completed", "paid_at": datetime.utcnow(), "payment_method": "test_mode", "test_mode": True}}
                    )
                except Exception as e:
                    print(f"⚠️ [PAYMENT] Error updating order: {e}")
            if user_id in manager.active_connections:
                await manager.send_personal_message({
                    "type": "video_mode_activated",
                    "minutes": tariff_minutes,
                    "order_id": order_id,
                    "test_mode": True,
                    "message": f"✅ Видео-режим активирован на {tariff_minutes if tariff_minutes > 0 else 'всегда'} минут (тестовый режим)"
                }, user_id)
            if hasattr(manager, 'activate_video_mode'):
                await manager.activate_video_mode(user_id, tariff_minutes)
            if hasattr(manager, 'pending_payments') and user_id in manager.pending_payments:
                del manager.pending_payments[user_id]
            return {
                "status": "success",
                "order_id": order_id,
                "payment_url": f"{PAYMENT_CONFIG.BACKEND_URL}/api/payment/success?order_id={order_id}&user_id={user_id}",
                "test_mode": True,
                "message": "✅ Видео-режим активирован (тестовый режим)"
            }
        payment_data = {
            "shop_id": PAYMENT_CONFIG.MORUNE_SHOP_ID,
            "amount": float(price),
            "currency": PAYMENT_CONFIG.CURRENCY,
            "order_id": order_id,
            "description": f"Видео-режим на {tariff_minutes} минут" if tariff_minutes > 0 else "Видео-режим навсегда",
            "success_url": f"{PAYMENT_CONFIG.BACKEND_URL}/api/payment/success?order_id={order_id}&user_id={user_id}",
            "fail_url": f"{PAYMENT_CONFIG.BACKEND_URL}/api/payment/fail?order_id={order_id}&user_id={user_id}",
            "callback_url": f"{PAYMENT_CONFIG.BACKEND_URL}/api/payment/callback",
            "metadata": {"user_id": user_id, "tariff_minutes": tariff_minutes, "product_type": "video_mode"}
        }
        payment_result = await create_morune_payment(payment_data)
        if payment_result.get("status") == "success":
            if video_orders_collection is not None:
                await video_orders_collection.update_one(
                    {"order_id": order_id},
                    {"$set": {"payment_id": payment_result.get("payment_id"), "payment_url": payment_result.get("payment_url"), "status": "waiting_payment"}}
                )
            if hasattr(manager, 'add_pending_payment'):
                await manager.add_pending_payment(user_id, {
                    "order_id": order_id,
                    "payment_url": payment_result.get("payment_url"),
                    "tariff_minutes": tariff_minutes,
                    "price": price,
                    "room_id": room_id,
                    "partner_id": partner_id
                })
            return {"status": "success", "order_id": order_id, "payment_url": payment_result.get("payment_url"), "test_mode": False}
        else:
            raise HTTPException(status_code=400, detail=f"Payment error: {payment_result.get('message')}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [PAYMENT] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@api_router.post("/ads/{ad_type}/{ad_id}/played")
async def mark_ad_played(ad_type: str, ad_id: str):
    """Увеличивает счётчик показов"""
    try:
        if ad_type == "audio":
            collection = getattr(app.state, 'audio_ads_collection', None)
        elif ad_type == "video":
            collection = getattr(app.state, 'video_ads_collection', None)
        else:
            raise HTTPException(status_code=400, detail="Invalid ad type")
        
        if collection is None:
            raise HTTPException(status_code=503, detail="Collection not available")
        
        await collection.update_one(
            {"id": ad_id},
            {"$inc": {"play_count": 1}, "$set": {"last_played_at": datetime.utcnow()}}
        )
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        

@api_router.get("/video-tariffs")
async def get_video_tariffs():
    try:
        if not hasattr(manager, 'video_tariffs') or not manager.video_tariffs:
            await manager.load_video_tariffs_from_db()
        active_tariffs = [t for t in manager.video_tariffs if t.get("is_active", True)]
        return {"status": "success", "tariffs": active_tariffs, "video_mode_enabled": manager.video_mode_enabled, "total": len(active_tariffs)}
    except Exception as e:
        print(f"❌ [TARIFFS] Error getting video tariffs: {e}")
        default_tariffs = [
            {"id": "tariff_1", "name": "15 минут", "minutes": 15, "price": 100.0, "is_active": True},
            {"id": "tariff_2", "name": "30 минут", "minutes": 30, "price": 180.0, "is_active": True},
            {"id": "tariff_3", "name": "60 минут", "minutes": 60, "price": 300.0, "is_active": True},
            {"id": "tariff_4", "name": "Навсегда", "minutes": 0, "price": 1000.0, "is_active": True}
        ]
        return {"status": "success", "tariffs": default_tariffs, "video_mode_enabled": True, "total": len(default_tariffs)}

@api_router.get("/ads/active")
async def get_active_ads():
    """Возвращает рекламу, которая сейчас должна показываться"""
    try:
        now = datetime.utcnow()
        
        async def get_active_from(collection):
            if collection is None:
                return []
            # Фильтр: is_active = True И (show_from <= now или None) И (show_until >= now или None)
            query = {
                "is_active": True,
                "$and": [
                    {"$or": [{"show_from": None}, {"show_from": {"$lte": now}}]},
                    {"$or": [{"show_until": None}, {"show_until": {"$gte": now}}]}
                ]
            }
            cursor = collection.find(query, {"_id": 0, "file_data": 0})
            return await cursor.to_list(length=50)
        
        audio_ads = await get_active_from(getattr(app.state, 'audio_ads_collection', None))
        video_ads = await get_active_from(getattr(app.state, 'video_ads_collection', None))
        
        return {
            "audio": serialize_mongo_doc(audio_ads),
            "video": serialize_mongo_doc(video_ads),
            "timestamp": now.isoformat()
        }
    except Exception as e:
        print(f"❌ [ACTIVE-ADS] Ошибка: {e}")
        return {"audio": [], "video": []}        

@api_router.get("/stats")
async def get_stats(request: Request):
    try:
        stats = manager.get_online_stats()
        return {
            "online_users": stats.get("online_users", 0),
            "unique_users": stats.get("unique_users", 0),
            "active_rooms": stats.get("active_rooms", 0),
            "searching_users": stats.get("searching_users", 0),
            "total_connections": stats.get("total_connections", 0),
            "server_status": stats.get("server_status", "online"),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"online_users": 0, "unique_users": 0, "active_rooms": 0, "searching_users": 0, "total_connections": 0, "server_status": "error"}

# ============================================================
# ПУБЛИЧНЫЕ ЭНДПОИНТЫ ДЛЯ ПОКАЗА РЕКЛАМЫ ВО ФРОНТЕНДЕ
# ============================================================

@api_router.get("/ads/active")
async def get_active_ads():
    """Возвращает активную рекламу (без file_data) с учётом периода показа"""
    print("📢 [API-ADS-ACTIVE] Запрос активной рекламы")
    try:
        now = datetime.utcnow()
        
        async def get_active_from(collection, ads_type):
            if collection is None:
                print(f"⚠️ [API-ADS-ACTIVE] Коллекция {ads_type} недоступна")
                return []
            # Фильтр: is_active + show_from <= now <= show_until
            query = {
                "is_active": True,
                "$and": [
                    {"$or": [{"show_from": None}, {"show_from": {"$lte": now}}]},
                    {"$or": [{"show_until": None}, {"show_until": {"$gte": now}}]}
                ]
            }
            cursor = collection.find(query, {"_id": 0, "file_data": 0}).limit(50)
            ads = await cursor.to_list(length=50)
            print(f"📢 [API-ADS-ACTIVE] {ads_type}: найдено {len(ads)}")
            return ads
        
        audio_ads = await get_active_from(getattr(app.state, 'audio_ads_collection', None), 'audio')
        video_ads = await get_active_from(getattr(app.state, 'video_ads_collection', None), 'video')
        
        return {
            "audio": serialize_mongo_doc(audio_ads),
            "video": serialize_mongo_doc(video_ads),
            "timestamp": now.isoformat()
        }
    except Exception as e:
        print(f"❌ [API-ADS-ACTIVE] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return {"audio": [], "video": [], "error": str(e)}


@api_router.get("/ads/{ad_type}/{ad_id}/data")
async def get_ad_data(ad_type: str, ad_id: str):
    """Возвращает file_data для показа рекламы во фронтенде"""
    print(f"📄 [API-ADS-DATA] type={ad_type}, id={ad_id}")
    try:
        if ad_type == "audio":
            collection = getattr(app.state, 'audio_ads_collection', None)
        elif ad_type == "video":
            collection = getattr(app.state, 'video_ads_collection', None)
        else:
            raise HTTPException(status_code=400, detail="Invalid ad type")
        
        if collection is None:
            print(f"❌ [API-ADS-DATA] Коллекция {ad_type} недоступна")
            raise HTTPException(status_code=503, detail="Collection not available")
        
        ad = await collection.find_one(
            {"id": ad_id},
            {"_id": 0, "file_data": 1, "name": 1, "duration": 1}
        )
        
        if not ad:
            print(f"❌ [API-ADS-DATA] Не найдено: {ad_id}")
            raise HTTPException(status_code=404, detail="Ad not found")
        
        file_data_size = len(ad.get("file_data", ""))
        print(f"✅ [API-ADS-DATA] Отдаю {file_data_size} байт ({file_data_size/1024/1024:.2f} MB)")
        
        return {
            "id": ad_id,
            "name": ad.get("name", ""),
            "duration": ad.get("duration", 30),
            "file_data": ad.get("file_data", "")
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [API-ADS-DATA] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/ads/{ad_type}/{ad_id}/played")
async def mark_ad_played(ad_type: str, ad_id: str):
    """Отмечает факт показа рекламы (увеличивает счётчик)"""
    print(f"👁️ [API-ADS-PLAYED] type={ad_type}, id={ad_id}")
    try:
        if ad_type == "audio":
            collection = getattr(app.state, 'audio_ads_collection', None)
        elif ad_type == "video":
            collection = getattr(app.state, 'video_ads_collection', None)
        else:
            raise HTTPException(status_code=400, detail="Invalid ad type")
        
        if collection is None:
            raise HTTPException(status_code=503, detail="Collection not available")
        
        result = await collection.update_one(
            {"id": ad_id},
            {
                "$inc": {"play_count": 1},
                "$set": {"last_played_at": datetime.utcnow()}
            }
        )
        
        print(f"👁️ [API-ADS-PLAYED] matched={result.matched_count}, modified={result.modified_count}")
        
        return {
            "status": "success",
            "matched": result.matched_count,
            "modified": result.modified_count
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [API-ADS-PLAYED] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/user/{user_id}/is-premium")
async def check_user_is_premium(user_id: str):
    """Проверяет, является ли пользователь премиум"""
    try:
        is_prem = await is_premium_user(user_id)
        return {"user_id": user_id, "is_premium": is_prem}
    except Exception as e:
        print(f"❌ [API-PREMIUM-CHECK] ОШИБКА: {e}")
        return {"user_id": user_id, "is_premium": False}

@api_router.post("/livekit/token")
async def get_livekit_token(request: LiveKitTokenRequest):
    """
    Выдаёт JWT-токен для подключения к LiveKit-комнате.
    Если room_name не передан — бэк сам найдёт комнату с местом или создаст новую.
    """
    print(f"🎥 [LIVEKIT] Token request: user={request.user_id}, room={request.room_name}, type={request.room_type}")
    try:
        # Ищем или создаём комнату
        if request.room_name:
            room_name = request.room_name
            # Регистрируем пользователя в комнате (на случай, если фронт пришёл со своим именем)
            with livekit_room_manager.lock:
                if room_name not in livekit_room_manager.rooms:
                    livekit_room_manager.rooms[room_name] = {
                        "count": 0,
                        "max": livekit_room_manager.MAX_PARTICIPANTS,
                        "users": set(),
                        "created": datetime.utcnow(),
                        "type": request.room_type or "group6",
                    }
                info = livekit_room_manager.rooms[room_name]
                if request.user_id not in info["users"]:
                    info["users"].add(request.user_id)
                    info["count"] = len(info["users"])
        else:
            room_name = livekit_room_manager.find_or_create_room(
                user_id=request.user_id,
                room_type=request.room_type or "group6",
            )

        # Генерируем токен
        token = (
            livekit_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(request.user_id)
            .with_name(request.nickname or request.user_id)
            .with_grants(
                livekit_api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )
        print(f"✅ [LIVEKIT] Token issued for {request.user_id} → room {room_name}")
        return {
            "token": token,
            "url": LIVEKIT_URL,
            "room": room_name,
        }
    except Exception as e:
        print(f"❌ [LIVEKIT] Token error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LiveKit token error: {str(e)}")

class LiveKitLeaveRequest(BaseModel):
    user_id: str
    room_name: str

@api_router.post("/livekit/leave")
async def livekit_leave_room(request: LiveKitLeaveRequest):
    """Убирает пользователя из LiveKit-комнаты (учёт заполненности)."""
    print(f"🎥 [LIVEKIT] Leave: user={request.user_id}, room={request.room_name}")
    try:
        livekit_room_manager.leave_room(request.user_id, request.room_name)
        return {"status": "success", "room": request.room_name}
    except Exception as e:
        print(f"❌ [LIVEKIT] Leave error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/livekit/rooms")
async def livekit_rooms_stats():
    """Статистика активных LiveKit-комнат (для отладки/админки)."""
    return livekit_room_manager.get_stats()                        

@api_router.get("/health")
async def public_health_check(request: Request):
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}

@api_router.get("/maintenance-status")
async def get_maintenance_status():
    return {
        "maintenance_mode": app.state.system_settings.get("maintenance_mode", False),
        "muffle_enabled": app.state.system_settings.get("muffle_enabled", False),
        "maintenance_message": app.state.system_settings.get("maintenance_message", ""),
        "server_status": "online"
    }

@api_router.get("/server-status")
async def get_server_status():
    return {"status": "online", "timestamp": datetime.utcnow().isoformat()}

@api_router.get("/user/{user_id}/moderation-status")
async def get_user_moderation_status_public(user_id: str, request: Request):
    try:
        # 🆕 IP из запроса
        ip_address = request.client.host if request.client else None
        if request.headers.get("x-forwarded-for"):
            ip_address = request.headers["x-forwarded-for"].split(",")[0].strip()
        
        # 🆕 Если IP не получен — пробуем взять из query param (fallback)
        if not ip_address:
            ip_address = request.query_params.get("ip")
        
        browser_id = request.query_params.get("browser_id")
        fingerprint = request.query_params.get("fingerprint")
        
        # 🆕 Если browser_id или fingerprint нет — пробуем взять из session
        if not browser_id or not fingerprint:
            session = manager.user_sessions.get(user_id, {})
            if not browser_id:
                browser_id = session.get("browser_id")
            if not fingerprint:
                fingerprint = session.get("fingerprint")
        
        print(f"🔍 [MOD-STATUS] user={user_id}, ip={ip_address}, browser={browser_id}, fp={fingerprint}")
        
        status = await get_user_moderation_status(user_id)
        
        ban_check = await is_user_fully_blocked(user_id, ip_address, browser_id, fingerprint)
        status["is_banned_extended"] = ban_check.get("blocked", False)
        status["ban_type"] = ban_check.get("ban_type")
        status["ban_reason"] = ban_check.get("reason")
        
        print(f"🔍 [MOD-STATUS] result: is_banned={status.get('is_banned')}, is_banned_extended={status.get('is_banned_extended')}, ban_type={status.get('ban_type')}")
        
        return serialize_mongo_doc(status)
    except Exception as e:
        logger.error(f"Error getting user moderation status: {e}")
        return {
            "user_id": user_id,
            "is_muted": False,
            "is_banned": False,
            "is_banned_extended": False,
            "warnings_count": 0,
            "active_warnings": [],
            "active_mutes": [],
            "total_reports": 0
        }

@api_router.get("/active-video-users")
async def get_active_video_users():
    try:
        video_users = []
        for user_id, user_session in manager.user_sessions.items():
            if manager.is_video_mode_active(user_id):
                video_users.append({
                    "user_id": user_id,
                    "room_id": user_session.get("room_id"),
                    "status": user_session.get("status"),
                    "video_mode_active": True,
                    "video_time_left": manager.get_video_time_left(user_id),
                    "last_activity": user_session.get("last_activity")
                })
        return {"video_users": video_users, "total_count": len(video_users)}
    except Exception as e:
        logger.error(f"Error getting active video users: {e}")
        return {"video_users": [], "total_count": 0}

@api_router.get("/payment-status/{order_id}")
async def get_payment_status(order_id: str):
    try:
        if not hasattr(app.state, 'video_orders_collection') or app.state.video_orders_collection is None:
            return {"order_id": order_id, "status": "database_unavailable"}
        order = await app.state.video_orders_collection.find_one({"order_id": order_id})
        if not order:
            return {"order_id": order_id, "status": "not_found"}
        return {"order_id": order_id, "status": order.get("status", "pending"), "user_id": order.get("user_id"), "tariff_minutes": order.get("tariff_minutes"), "test_mode": order.get("test_mode", False)}
    except Exception as e:
        return {"order_id": order_id, "status": "error", "message": str(e)}

@api_router.get("/payment/success")
async def payment_success(request: Request):
    try:
        order_id = request.query_params.get("order_id")
        user_id = request.query_params.get("user_id")
        print(f"💰 [PAYMENT] Payment success page accessed - order: {order_id}, user: {user_id}")
        if order_id and hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection:
            try:
                existing_order = await app.state.video_orders_collection.find_one({"order_id": order_id})
                if existing_order and existing_order.get("status") != "completed":
                    await app.state.video_orders_collection.update_one(
                        {"order_id": order_id},
                        {"$set": {"status": "completed", "paid_at": datetime.utcnow()}}
                    )
            except Exception as e:
                print(f"⚠️ [PAYMENT] Error updating order: {e}")
        if user_id and hasattr(manager, 'pending_payments') and user_id in manager.pending_payments:
            del manager.pending_payments[user_id]
        if user_id and not manager.is_video_mode_active(user_id):
            tariff_minutes = 15
            if order_id and hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection:
                try:
                    order = await app.state.video_orders_collection.find_one({"order_id": order_id})
                    if order:
                        tariff_minutes = order.get("tariff_minutes", 15)
                except:
                    pass
            await manager.activate_video_mode(user_id, tariff_minutes)
        if user_id and user_id in manager.active_connections:
            await manager.send_personal_message({
                "type": "payment_success",
                "order_id": order_id,
                "message": "✅ Оплата прошла успешно! Видео-режим активирован."
            }, user_id)
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"><title>Оплата успешна</title>
        <style>
            body { background: linear-gradient(135deg, #0a0a2a, #1a1a3e); min-height: 100vh; display: flex; align-items: center; justify-content: center; font-family: Arial; color: white; }
            .container { background: rgba(10,20,40,0.95); border-radius: 20px; padding: 40px; text-align: center; border: 1px solid #00ff88; }
            h1 { color: #00ff88; } .close-btn { background: #00d2ff; padding: 10px 20px; border-radius: 10px; color: white; border: none; cursor: pointer; margin-top: 20px; font-size: 16px; }
        </style>
        </head>
        <body><div class="container"><div style="font-size:60px;">✅</div><h1>Оплата успешна!</h1><p>Видео-режим активирован.</p><button class="close-btn" onclick="window.close()">Закрыть окно</button></div>
        <script>setTimeout(function(){window.close();},3000);</script></body></html>
        """)
    except Exception as e:
        print(f"❌ [PAYMENT] Error in payment success page: {e}")
        return HTMLResponse("<h1>Оплата успешна</h1><p>Видео-режим активирован.</p><button onclick='window.close()'>Закрыть</button>")

@api_router.get("/payment/fail")
async def payment_fail(request: Request):
    try:
        order_id = request.query_params.get("order_id")
        user_id = request.query_params.get("user_id")
        print(f"❌ [PAYMENT] Payment fail page accessed - order: {order_id}, user: {user_id}")
        if order_id and hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection:
            try:
                await app.state.video_orders_collection.update_one(
                    {"order_id": order_id},
                    {"$set": {"status": "failed", "failed_at": datetime.utcnow()}}
                )
            except Exception as e:
                print(f"⚠️ [PAYMENT] Error updating order: {e}")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"><title>Оплата не прошла</title>
        <style>
            body { background: linear-gradient(135deg, #0a0a2a, #1a1a3e); min-height: 100vh; display: flex; align-items: center; justify-content: center; font-family: Arial; color: white; }
            .container { background: rgba(10,20,40,0.95); border-radius: 20px; padding: 40px; text-align: center; border: 1px solid #ff4757; }
            h1 { color: #ff4757; } .btn { background: #ff4757; padding: 10px 20px; border-radius: 10px; color: white; border: none; cursor: pointer; margin-top: 20px; font-size: 16px; text-decoration: none; display: inline-block; }
        </style>
        </head>
        <body><div class="container"><div style="font-size:60px;">❌</div><h1>Оплата не прошла</h1><p>Пожалуйста, попробуйте еще раз.</p><a href="/" class="btn">Вернуться</a></div></body></html>
        """)
    except Exception as e:
        print(f"❌ [PAYMENT] Error in payment fail page: {e}")
        return HTMLResponse("<h1>Оплата не прошла</h1><p>Пожалуйста, попробуйте еще раз.</p><a href='/'>Вернуться</a>")

@api_router.post("/payment/callback")
async def payment_callback(request: Request):
    try:
        body = await request.json()
        print(f"💰 [PAYMENT] Callback received: {body}")
        order_id = body.get("order_id")
        status = body.get("status")
        payment_id = body.get("payment_id")
        if not order_id:
            return JSONResponse(status_code=400, content={"error": "Missing order_id"})
        if hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection:
            if status == "completed":
                order = await app.state.video_orders_collection.find_one({"order_id": order_id})
                await app.state.video_orders_collection.update_one(
                    {"order_id": order_id},
                    {"$set": {"status": "completed", "payment_id": payment_id, "paid_at": datetime.utcnow(), "callback_received": True}}
                )
                if order and order.get("user_id"):
                    user_id = order.get("user_id")
                    tariff_minutes = order.get("tariff_minutes", 15)
                    await manager.activate_video_mode(user_id, tariff_minutes)
                    if hasattr(manager, 'pending_payments') and user_id in manager.pending_payments:
                        del manager.pending_payments[user_id]
                    if user_id in manager.active_connections:
                        await manager.send_personal_message({
                            "type": "payment_success",
                            "order_id": order_id,
                            "minutes": tariff_minutes,
                            "message": f"✅ Оплата прошла успешно! Видео-режим активирован на {tariff_minutes if tariff_minutes > 0 else 'всегда'} минут."
                        }, user_id)
                print(f"✅ [PAYMENT] Callback: Order {order_id} completed")
            elif status == "failed":
                await app.state.video_orders_collection.update_one(
                    {"order_id": order_id},
                    {"$set": {"status": "failed", "payment_id": payment_id, "failed_at": datetime.utcnow()}}
                )
                print(f"❌ [PAYMENT] Callback: Order {order_id} failed")
            elif status == "cancelled":
                await app.state.video_orders_collection.update_one(
                    {"order_id": order_id},
                    {"$set": {"status": "cancelled", "payment_id": payment_id, "cancelled_at": datetime.utcnow()}}
                )
                order = await app.state.video_orders_collection.find_one({"order_id": order_id})
                if order and order.get("user_id"):
                    user_id = order.get("user_id")
                    if hasattr(manager, 'pending_payments') and user_id in manager.pending_payments:
                        del manager.pending_payments[user_id]
                print(f"⚠️ [PAYMENT] Callback: Order {order_id} cancelled")
        return JSONResponse(status_code=200, content={"status": "ok"})
    except Exception as e:
        print(f"❌ [PAYMENT] Callback error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============================================================
# АДМИН ЭНДПОИНТЫ
# ============================================================
@admin_router.get("/stats")
async def admin_stats(request: Request, moderator: dict = Depends(authenticate_admin)):
    try:
        basic_stats = manager.get_online_stats()
        connections = manager.get_all_connections()
        moderation_stats = await get_moderation_stats_internal()
        system_stats = get_system_stats()
        video_stats = manager.get_video_stats()
        premium_users_count = 0
        try:
            if hasattr(app.state, 'premium_collection') and app.state.premium_collection:
                premium_users_count = await app.state.premium_collection.count_documents({"is_active": True})
        except:
            pass
        return {
            "online_users": basic_stats.get("online_users", 0),
            "unique_users": basic_stats.get("unique_users", 0),
            "active_rooms": basic_stats.get("active_rooms", 0),
            "searching_users": basic_stats.get("searching_users", 0),
            "total_connections": basic_stats.get("total_connections", 0),
            "queues_by_category": basic_stats.get("queues_by_category", {}),
            "active_reports": moderation_stats.get("active_reports", 0),
            "banned_users": moderation_stats.get("banned_users", 0),
            "muted_users": moderation_stats.get("muted_users", 0),
            "moderation_actions_today": moderation_stats.get("moderation_actions_today", 0),
            "active_warnings": moderation_stats.get("active_warnings", 0),
            "cpu_usage": system_stats.get("cpu_usage", 0),
            "memory_usage": system_stats.get("memory_usage", 0),
            "uptime": system_stats.get("uptime", "0s"),
            "uptime_seconds": system_stats.get("uptime_seconds", 0),
            "system_memory_used": system_stats.get("system_memory_used", 0),
            "system_memory_total": system_stats.get("system_memory_total", 0),
            "system_memory_percent": system_stats.get("system_memory_percent", 0),
            "disk_used": system_stats.get("disk_used", 0),
            "disk_total": system_stats.get("disk_total", 0),
            "disk_percent": system_stats.get("disk_percent", 0),
            "video_mode_enabled": video_stats.get("video_mode_enabled", True),
            "active_video_users": video_stats.get("active_video_users", 0),
            "permanent_video_users": video_stats.get("permanent_video_users", 0),
            "premium_users": premium_users_count,
            "connections": connections[:50],
            "connections_count": len(connections),
            "server_status": "online",
            "db_connected": app.state.db is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in admin stats: {e}")
        return {"online_users": len(manager.active_connections), "unique_users": 0, "active_rooms": len(manager.active_rooms), "searching_users": 0, "server_status": "error", "connections": [], "error": str(e)}

@admin_router.get("/connections")
async def admin_connections(request: Request, moderator: dict = Depends(authenticate_admin)):
    try:
        if hasattr(manager, 'update_ip_info_cache'):
            try:
                await manager.update_ip_info_cache()
            except:
                pass
        connections = manager.get_all_connections()
        return JSONResponse(content=serialize_mongo_doc(connections), status_code=200)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "connections": [], "total": 0})

@admin_router.get("/user-info/{user_id}")
async def get_user_info_for_ban(user_id: str, moderator: dict = Depends(authenticate_admin)):
    """Получить инфо о пользователе для расширенного бана (работает и для offline)"""
    try:
        result = {
            "user_id": user_id,
            "in_online": False,
            "status": "unknown",
            "ip_address": None,
            "browser_id": None,
            "fingerprint": None,
            "is_banned_user": False,
            "is_banned_ip": False,
            "is_banned_browser": False,
            "is_banned_fingerprint": False,
            "is_banned_subnet": False,
        }
        
        # ===== Если пользователь в онлайне — берём из сессии =====
        if user_id in manager.user_sessions:
            session = manager.user_sessions[user_id]
            result["in_online"] = True
            result["status"] = session.get("status", "unknown")
            result["ip_address"] = session.get("ip_address")
            result["browser_id"] = session.get("browser_id")
            result["fingerprint"] = session.get("fingerprint")
        
        # ===== Если offline — берём из БД =====
        if not result["ip_address"] or not result["browser_id"] or not result["fingerprint"]:
            if hasattr(app.state, 'users_collection') and app.state.users_collection is not None:
                try:
                    user_doc = await app.state.users_collection.find_one({"user_id": user_id})
                    if user_doc:
                        if not result["ip_address"]:
                            result["ip_address"] = user_doc.get("ip_address")
                        if not result["browser_id"]:
                            result["browser_id"] = user_doc.get("browser_id")
                        if not result["fingerprint"]:
                            result["fingerprint"] = user_doc.get("fingerprint")
                except Exception as e:
                    print(f"⚠️ [USER-INFO] DB error: {e}")
        
        # ===== Проверяем текущие баны =====
        try:
            if await is_user_blocked(user_id):
                result["is_banned_user"] = True
        except:
            pass
        
        if result["ip_address"]:
            try:
                if hasattr(app.state, 'blocked_ips_collection') and app.state.blocked_ips_collection is not None:
                    blocked = await app.state.blocked_ips_collection.find_one({
                        "ip": result["ip_address"],
                        "$or": [{"expires_at": {"$gt": datetime.utcnow()}}, {"expires_at": None}]
                    })
                    result["is_banned_ip"] = blocked is not None
            except:
                pass
        
        if result["browser_id"]:
            try:
                if hasattr(app.state, 'blocked_browsers_collection') and app.state.blocked_browsers_collection is not None:
                    blocked = await app.state.blocked_browsers_collection.find_one({
                        "browser_id": result["browser_id"],
                        "$or": [{"expires_at": {"$gt": datetime.utcnow()}}, {"expires_at": None}]
                    })
                    result["is_banned_browser"] = blocked is not None
            except:
                pass
        
        if result["fingerprint"]:
            try:
                if hasattr(app.state, 'blocked_fingerprints_collection') and app.state.blocked_fingerprints_collection is not None:
                    blocked = await app.state.blocked_fingerprints_collection.find_one({
                        "fingerprint": result["fingerprint"],
                        "$or": [{"expires_at": {"$gt": datetime.utcnow()}}, {"expires_at": None}]
                    })
                    result["is_banned_fingerprint"] = blocked is not None
            except:
                pass
        
        if result["ip_address"]:
            try:
                ip_parts = result["ip_address"].split('.')
                if len(ip_parts) >= 3 and ip_parts[0] != '0':
                    subnet = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"
                    if hasattr(app.state, 'blocked_subnets_collection') and app.state.blocked_subnets_collection is not None:
                        blocked = await app.state.blocked_subnets_collection.find_one({
                            "subnet": subnet,
                            "$or": [{"expires_at": {"$gt": datetime.utcnow()}}, {"expires_at": None}]
                        })
                        result["is_banned_subnet"] = blocked is not None
            except:
                pass
        
        return result
    except Exception as e:
        logger.error(f"[USER-INFO] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))        

# ============================================================
# РЕКЛАМНЫЕ ЭНДПОИНТЫ
# ============================================================
@admin_router.get("/ads/audio/list")
async def get_audio_ads(moderator: dict = Depends(authenticate_admin)):
    print("📋 [AD-AUDIO-LIST] Загрузка списка аудио-рекламы...")
    try:
        ads = []
        collection = getattr(app.state, 'audio_ads_collection', None)
        
        if collection is not None:
            # 🔧 FIX: НЕ запрашиваем file_data (он огромный base64)
            # 🔧 FIX: НЕ сортируем на стороне MongoDB (это вызывает memory limit)
            cursor = collection.find(
                {},
                {
                    "_id": 0,           # не тянем ObjectId
                    "file_data": 0,     # НЕ тянем сам файл (он в /admin/ads/audio/{id}/data)
                }
            ).limit(100)
            
            ads = await cursor.to_list(length=100)
            print(f"✅ [AD-AUDIO-LIST] Загружено {len(ads)} записей (без file_data)")
            
            # Сортируем на стороне Python (маленький объём данных)
            try:
                ads.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            except Exception:
                pass
        else:
            print("⚠️ [AD-AUDIO-LIST] Коллекция недоступна")
        
        return {"ads": serialize_mongo_doc(ads)}
    except Exception as e:
        print(f"❌ [AD-AUDIO-LIST] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return {"ads": [], "error": str(e)}

# ============= GET ФАЙЛА АУДИО-РЕКЛАМЫ =============
# ============= GET FILE_DATA АУДИО-РЕКЛАМЫ =============
@admin_router.get("/ads/audio/{ad_id}/data")
async def get_audio_ad_data(ad_id: str, moderator: dict = Depends(authenticate_admin)):
    print(f"📄 [AD-AUDIO-DATA] ad_id={ad_id}")
    try:
        collection = getattr(app.state, 'audio_ads_collection', None)
        if collection is None:
            print("❌ [AD-AUDIO-DATA] Коллекция недоступна")
            raise HTTPException(status_code=503, detail="Collection not available")
        
        ad = await collection.find_one(
            {"id": ad_id},
            {"_id": 0, "file_data": 1, "name": 1, "duration": 1}
        )
        if not ad:
            print(f"❌ [AD-AUDIO-DATA] Не найдено: {ad_id}")
            raise HTTPException(status_code=404, detail="Ad not found")
        
        print(f"✅ [AD-AUDIO-DATA] Отдаю {len(ad.get('file_data', ''))} байт")
        return {
            "id": ad_id,
            "name": ad.get("name", ""),
            "duration": ad.get("duration", 30),
            "file_data": ad.get("file_data", "")
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AD-AUDIO-DATA] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============= ОБНОВИТЬ ПЕРИОД ПОКАЗА АУДИО =============
@admin_router.put("/ads/audio/{ad_id}/schedule")
async def update_audio_ad_schedule(ad_id: str, request: dict, moderator: dict = Depends(authenticate_admin)):
    print(f"⏰ [AD-AUDIO-SCHEDULE] ad_id={ad_id}, preset={request.get('period_preset')}")
    try:
        collection = getattr(app.state, 'audio_ads_collection', None)
        if collection is None:
            raise HTTPException(status_code=503, detail="Collection not available")
        
        # Собираем поля
        period_preset = request.get("period_preset", "always")
        show_from = request.get("show_from")  # ISO строка или None
        show_until = request.get("show_until")  # ISO строка или None
        is_active = request.get("is_active", True)
        
        # Парсим даты
        def parse_date(s):
            if not s:
                return None
            try:
                # Поддерживаем формат 'YYYY-MM-DDTHH:MM'
                return datetime.fromisoformat(s.replace('Z', ''))
            except Exception:
                return None
        
        show_from_dt = parse_date(show_from)
        show_until_dt = parse_date(show_until)
        
        update_data = {
            "period_preset": period_preset,
            "show_from": show_from_dt,
            "show_until": show_until_dt,
            "is_active": bool(is_active),
            "updated_at": datetime.utcnow(),
            "updated_by": moderator["username"]
        }
        
        print(f"⏰ [AD-AUDIO-SCHEDULE] Обновляем: {update_data}")
        
        result = await collection.update_one(
            {"id": ad_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Ad not found")
        
        return {
            "status": "success",
            "message": "Период показа обновлён",
            "period_preset": period_preset,
            "show_from": show_from_dt.isoformat() if show_from_dt else None,
            "show_until": show_until_dt.isoformat() if show_until_dt else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AD-AUDIO-SCHEDULE] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============= ОБНОВИТЬ ПЕРИОД ПОКАЗА ВИДЕО =============
@admin_router.put("/ads/video/{ad_id}/schedule")
async def update_video_ad_schedule(ad_id: str, request: dict, moderator: dict = Depends(authenticate_admin)):
    print(f"⏰ [AD-VIDEO-SCHEDULE] ad_id={ad_id}, preset={request.get('period_preset')}")
    try:
        collection = getattr(app.state, 'video_ads_collection', None)
        if collection is None:
            raise HTTPException(status_code=503, detail="Collection not available")
        
        period_preset = request.get("period_preset", "always")
        show_from = request.get("show_from")
        show_until = request.get("show_until")
        is_active = request.get("is_active", True)
        
        def parse_date(s):
            if not s:
                return None
            try:
                return datetime.fromisoformat(s.replace('Z', ''))
            except Exception:
                return None
        
        show_from_dt = parse_date(show_from)
        show_until_dt = parse_date(show_until)
        
        update_data = {
            "period_preset": period_preset,
            "show_from": show_from_dt,
            "show_until": show_until_dt,
            "is_active": bool(is_active),
            "updated_at": datetime.utcnow(),
            "updated_by": moderator["username"]
        }
        
        result = await collection.update_one(
            {"id": ad_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Ad not found")
        
        return {
            "status": "success",
            "message": "Период показа обновлён",
            "period_preset": period_preset,
            "show_from": show_from_dt.isoformat() if show_from_dt else None,
            "show_until": show_until_dt.isoformat() if show_until_dt else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AD-VIDEO-SCHEDULE] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))        


# ============= АНАЛОГИЧНО ДЛЯ ВИДЕО =============
@admin_router.get("/ads/video/{ad_id}/data")
async def get_video_ad_data(ad_id: str, moderator: dict = Depends(authenticate_admin)):
    """Возвращает file_data для конкретной видео-рекламы"""
    try:
        collection = getattr(app.state, 'video_ads_collection', None)
        if collection is None:
            raise HTTPException(status_code=503, detail="Collection not available")
        
        ad = await collection.find_one({"id": ad_id}, {"file_data": 1, "name": 1, "_id": 0})
        if not ad:
            raise HTTPException(status_code=404, detail="Ad not found")
        
        return {
            "id": ad_id,
            "name": ad.get("name", ""),
            "file_data": ad.get("file_data", "")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        

# ============= GET FILE_DATA ВИДЕО-РЕКЛАМЫ =============
@admin_router.get("/ads/video/{ad_id}/data")
async def get_video_ad_data(ad_id: str, moderator: dict = Depends(authenticate_admin)):
    print(f"📄 [AD-VIDEO-DATA] ad_id={ad_id}")
    try:
        collection = getattr(app.state, 'video_ads_collection', None)
        if collection is None:
            print("❌ [AD-VIDEO-DATA] Коллекция недоступна")
            raise HTTPException(status_code=503, detail="Collection not available")
        
        ad = await collection.find_one(
            {"id": ad_id},
            {"_id": 0, "file_data": 1, "name": 1, "duration": 1}
        )
        if not ad:
            print(f"❌ [AD-VIDEO-DATA] Не найдено: {ad_id}")
            raise HTTPException(status_code=404, detail="Ad not found")
        
        print(f"✅ [AD-VIDEO-DATA] Отдаю {len(ad.get('file_data', ''))} байт")
        return {
            "id": ad_id,
            "name": ad.get("name", ""),
            "duration": ad.get("duration", 30),
            "file_data": ad.get("file_data", "")
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AD-VIDEO-DATA] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============= GET СПИСОК ВИДЕО-РЕКЛАМЫ =============
@admin_router.get("/ads/video/list")
async def get_video_ads(moderator: dict = Depends(authenticate_admin)):
    print("📋 [AD-VIDEO-LIST] Загрузка списка видео-рекламы...")
    try:
        ads = []
        collection = getattr(app.state, 'video_ads_collection', None)
        
        if collection is not None:
            cursor = collection.find(
                {},
                {"_id": 0, "file_data": 0}   # НЕ тянем файл
            ).limit(100)
            
            ads = await cursor.to_list(length=100)
            print(f"✅ [AD-VIDEO-LIST] Загружено {len(ads)} записей (без file_data)")
            
            try:
                ads.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
            except Exception:
                pass
        else:
            print("⚠️ [AD-VIDEO-LIST] Коллекция недоступна")
        
        return {"ads": serialize_mongo_doc(ads)}
    except Exception as e:
        print(f"❌ [AD-VIDEO-LIST] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return {"ads": [], "error": str(e)}        

@admin_router.get("/ads/settings")
async def get_ad_settings(moderator: dict = Depends(authenticate_admin)):
    try:
        default_settings = {"audio_enabled": True, "video_enabled": True, "audio_interval_seconds": 300, "video_interval_seconds": 600}
        if hasattr(app.state, 'system_settings_collection') and app.state.system_settings_collection:
            settings_doc = await app.state.system_settings_collection.find_one({"key": "ad_settings"})
            if settings_doc:
                return {
                    "audio_enabled": settings_doc.get("audio_enabled", True),
                    "video_enabled": settings_doc.get("video_enabled", True),
                    "audio_interval_seconds": settings_doc.get("audio_interval_seconds", 300),
                    "video_interval_seconds": settings_doc.get("video_interval_seconds", 600)
                }
        return default_settings
    except Exception as e:
        return default_settings

@admin_router.get("/chat/history")
async def get_chat_history(
    moderator: dict = Depends(authenticate_admin),
    limit: int = Query(100, ge=1, le=500)
):
    """Получить историю чата"""
    try:
        messages = admin_chat_messages[-limit:] if admin_chat_messages else []
        return {
            "messages": messages,
            "total": len(admin_chat_messages)
        }
    except Exception as e:
        return {"messages": [], "total": 0, "error": str(e)}

@admin_router.post("/ads/settings")
async def update_ad_settings(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        settings = {
            "audio_enabled": request.get("audio_enabled", True),
            "video_enabled": request.get("video_enabled", True),
            "audio_interval_seconds": request.get("audio_interval_seconds", 300),
            "video_interval_seconds": request.get("video_interval_seconds", 600),
            "updated_at": datetime.utcnow(),
            "updated_by": moderator["username"]
        }
        if hasattr(app.state, 'system_settings_collection') and app.state.system_settings_collection:
            await app.state.system_settings_collection.update_one({"key": "ad_settings"}, {"$set": settings}, upsert=True)
        return {"status": "success", "settings": settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.post("/advanced-ban")
async def advanced_ban(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        user_id = request.get("user_id")
        reason = request.get("reason", "Нарушение правил")
        reason = translate_reason(reason)
        duration_hours = request.get("duration_hours", 24)
        ban_types = request.get("ban_types", {})
        
        print(f"🔒 [ADVANCED-BAN] Request: user_id={user_id}, ban_types={ban_types}, duration_hours={duration_hours}")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        is_permanent = (duration_hours == 0)
        expires_at = None if is_permanent else (datetime.utcnow() + timedelta(hours=duration_hours))
        
        results = []
        banned_items = []
        
        # ===== 1. Бан по ID пользователя =====
        if ban_types.get("user", False):
            try:
                if not hasattr(app.state, 'blocked_users_collection') or app.state.blocked_users_collection is None:
                    if app.state.db is not None:
                        app.state.blocked_users_collection = app.state.db["blocked_users"]
                
                if app.state.blocked_users_collection is not None:
                    await app.state.blocked_users_collection.update_one(
                        {"user_id": user_id},
                        {"$set": {
                            "user_id": user_id,
                            "reason": reason,
                            "banned_at": datetime.utcnow(),
                            "expires_at": expires_at,
                            "is_permanent": is_permanent,
                            "banned_by": moderator["username"],
                            "ban_type": "user"
                        }},
                        upsert=True
                    )
                    results.append("✅ ID пользователя заблокирован")
                    banned_items.append(f"user:{user_id}")
                    print(f"✅ [ADVANCED-BAN] User {user_id} banned by ID (permanent={is_permanent})")
            except Exception as e:
                results.append(f"⚠️ Ошибка бана ID: {str(e)}")
                print(f"❌ [ADVANCED-BAN] User ban error: {e}")
        
        # ===== Получаем данные пользователя =====
        ip_address = None
        browser_id = None
        fingerprint = None
        
        if user_id in manager.user_sessions:
            session = manager.user_sessions[user_id]
            ip_address = session.get("ip_address")
            browser_id = session.get("browser_id")
            fingerprint = session.get("fingerprint")
            print(f"📋 [ADVANCED-BAN] Session data: ip={ip_address}, browser={browser_id}, fp={fingerprint}")
        
        if not ip_address or not browser_id or not fingerprint:
            if hasattr(app.state, 'users_collection') and app.state.users_collection:
                try:
                    user_doc = await app.state.users_collection.find_one({"user_id": user_id})
                    if user_doc:
                        if not ip_address:
                            ip_address = user_doc.get("ip_address")
                        if not browser_id:
                            browser_id = user_doc.get("browser_id")
                        if not fingerprint:
                            fingerprint = user_doc.get("fingerprint")
                        print(f"📋 [ADVANCED-BAN] DB data: ip={ip_address}, browser={browser_id}, fp={fingerprint}")
                except Exception as e:
                    print(f"⚠️ [ADVANCED-BAN] DB error: {e}")
        
        if not ip_address and user_id in manager.user_ips:
            ip_address = manager.user_ips[user_id]
            print(f"📋 [ADVANCED-BAN] IP from user_ips: {ip_address}")
        
        # ===== 2. Бан по IP =====
        if ban_types.get("ip", False) and ip_address:
            try:
                if not hasattr(app.state, 'blocked_ips_collection') or app.state.blocked_ips_collection is None:
                    if app.state.db is not None:
                        app.state.blocked_ips_collection = app.state.db["blocked_ips"]
                
                if app.state.blocked_ips_collection is not None:
                    await app.state.blocked_ips_collection.update_one(
                        {"ip": ip_address},
                        {"$set": {
                            "ip": ip_address,
                            "user_id": user_id,
                            "reason": reason,
                            "banned_at": datetime.utcnow(),
                            "expires_at": expires_at,
                            "is_permanent": is_permanent,
                            "banned_by": moderator["username"],
                            "ban_type": "ip"
                        }},
                        upsert=True
                    )
                    results.append(f"✅ IP адрес {ip_address} заблокирован")
                    banned_items.append(f"ip:{ip_address}")
                    print(f"✅ [ADVANCED-BAN] IP {ip_address} banned (permanent={is_permanent})")
            except Exception as e:
                results.append(f"⚠️ Ошибка бана IP: {str(e)}")
                print(f"❌ [ADVANCED-BAN] IP ban error: {e}")
        
        # ===== 3. Бан по Browser =====
        if ban_types.get("browser", False) and browser_id:
            try:
                if not hasattr(app.state, 'blocked_browsers_collection') or app.state.blocked_browsers_collection is None:
                    if app.state.db is not None:
                        app.state.blocked_browsers_collection = app.state.db["blocked_browsers"]
                
                if app.state.blocked_browsers_collection is not None:
                    await app.state.blocked_browsers_collection.update_one(
                        {"browser_id": browser_id},
                        {"$set": {
                            "browser_id": browser_id,
                            "user_id": user_id,
                            "reason": reason,
                            "banned_at": datetime.utcnow(),
                            "expires_at": expires_at,
                            "is_permanent": is_permanent,
                            "banned_by": moderator["username"],
                            "ban_type": "browser"
                        }},
                        upsert=True
                    )
                    results.append(f"✅ Browser ID {browser_id[:8]}... заблокирован")
                    banned_items.append(f"browser:{browser_id}")
                    print(f"✅ [ADVANCED-BAN] Browser {browser_id[:8]}... banned (permanent={is_permanent})")
            except Exception as e:
                results.append(f"⚠️ Ошибка бана Browser: {str(e)}")
                print(f"❌ [ADVANCED-BAN] Browser ban error: {e}")
        
        # ===== 4. Бан по Fingerprint =====
        if ban_types.get("fingerprint", False) and fingerprint:
            try:
                if not hasattr(app.state, 'blocked_fingerprints_collection') or app.state.blocked_fingerprints_collection is None:
                    if app.state.db is not None:
                        app.state.blocked_fingerprints_collection = app.state.db["blocked_fingerprints"]
                
                if app.state.blocked_fingerprints_collection is not None:
                    await app.state.blocked_fingerprints_collection.update_one(
                        {"fingerprint": fingerprint},
                        {"$set": {
                            "fingerprint": fingerprint,
                            "user_id": user_id,
                            "reason": reason,
                            "banned_at": datetime.utcnow(),
                            "expires_at": expires_at,
                            "is_permanent": is_permanent,
                            "banned_by": moderator["username"],
                            "ban_type": "fingerprint"
                        }},
                        upsert=True
                    )
                    results.append(f"✅ Fingerprint {fingerprint[:8]}... заблокирован")
                    banned_items.append(f"fingerprint:{fingerprint}")
                    print(f"✅ [ADVANCED-BAN] Fingerprint {fingerprint[:8]}... banned (permanent={is_permanent})")
            except Exception as e:
                results.append(f"⚠️ Ошибка бана Fingerprint: {str(e)}")
                print(f"❌ [ADVANCED-BAN] Fingerprint ban error: {e}")
        
        # ===== 5. Бан по подсети =====
        if ban_types.get("subnet", False) and ip_address:
            try:
                ip_parts = ip_address.split('.')
                if len(ip_parts) >= 3:
                    subnet = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"
                    
                    if not hasattr(app.state, 'blocked_subnets_collection') or app.state.blocked_subnets_collection is None:
                        if app.state.db is not None:
                            app.state.blocked_subnets_collection = app.state.db["blocked_subnets"]
                    
                    if app.state.blocked_subnets_collection is not None:
                        await app.state.blocked_subnets_collection.update_one(
                            {"subnet": subnet},
                            {"$set": {
                                "subnet": subnet,
                                "user_id": user_id,
                                "reason": reason,
                                "banned_at": datetime.utcnow(),
                                "expires_at": expires_at,
                                "is_permanent": is_permanent,
                                "banned_by": moderator["username"],
                                "ban_type": "subnet"
                            }},
                            upsert=True
                        )
                        results.append(f"✅ Подсеть {subnet} заблокирована")
                        banned_items.append(f"subnet:{subnet}")
                        print(f"✅ [ADVANCED-BAN] Subnet {subnet} banned (permanent={is_permanent})")
            except Exception as e:
                results.append(f"⚠️ Ошибка бана подсети: {str(e)}")
                print(f"❌ [ADVANCED-BAN] Subnet ban error: {e}")
        
        # ===== Отключаем пользователя =====
        if user_id in manager.active_connections:
            try:
                await manager.force_disconnect_user(user_id)
                results.append("✅ Пользователь отключен")
                print(f"✅ [ADVANCED-BAN] User {user_id} disconnected")
            except Exception as e:
                results.append(f"⚠️ Ошибка отключения: {str(e)}")
                print(f"❌ [ADVANCED-BAN] Disconnect error: {e}")
        
        # ===== Логирование =====
        try:
            await log_moderation_action(
                moderator["username"],
                "advanced_ban",
                user_id,
                reason,
                duration_hours * 60 if duration_hours > 0 else None
            )
        except Exception as e:
            print(f"⚠️ [ADVANCED-BAN] Log error: {e}")
        
        try:
            ban_types_str = ', '.join([k for k, v in ban_types.items() if v])
            await notify_moderators(
                f"🔒 Расширенный бан пользователя {user_id} от {moderator['username']}\n"
                f"Причина: {reason}\n"
                f"Длительность: {'Навсегда' if is_permanent else f'{duration_hours} часов'}\n"
                f"Типы бана: {ban_types_str}"
            )
        except Exception as e:
            print(f"⚠️ [ADVANCED-BAN] Notification error: {e}")
        
        return {
            "status": "success",
            "message": f"Пользователь {user_id} забанен",
            "results": results,
            "banned_items": banned_items,
            "user_id": user_id,
            "duration_hours": duration_hours,
            "is_permanent": is_permanent
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [ADVANCED-BAN] Critical error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@admin_router.post("/ads/audio/upload")
async def upload_audio_ad(request: dict, moderator: dict = Depends(authenticate_admin)):
    print("=" * 60)
    print("📤 [AD-AUDIO-UPLOAD] ============ START ============")
    
    try:
        file_data = request.get("file_data", "")
        name = request.get("name", "").strip()
        duration = request.get("duration", 30)
        
        print(f"📤 [AD-AUDIO-UPLOAD] name={name}")
        print(f"📤 [AD-AUDIO-UPLOAD] duration={duration}")
        print(f"📤 [AD-AUDIO-UPLOAD] file_data length={len(file_data)} bytes ({len(file_data)/1024/1024:.2f} MB)")
        print(f"📤 [AD-AUDIO-UPLOAD] moderator={moderator.get('username')}")
        
        # ===== ПРОВЕРКА 1: file_data не пустой =====
        if not file_data:
            print("❌ [AD-AUDIO-UPLOAD] file_data ПУСТОЙ")
            raise HTTPException(status_code=400, detail="File data is required")
        
        # ===== ПРОВЕРКА 2: коллекция доступна =====
        print(f"📤 [AD-AUDIO-UPLOAD] app.state.db = {getattr(app.state, 'db', 'NOT SET')}")
        collection = getattr(app.state, 'audio_ads_collection', None)
        print(f"📤 [AD-AUDIO-UPLOAD] audio_ads_collection = {collection}")
        
        if collection is None:
            print("❌ [AD-AUDIO-UPLOAD] КОЛЛЕКЦИЯ = NONE! MongoDB не подключена?")
            raise HTTPException(status_code=503, detail="Database collection not available")
        
        # ===== ПРОВЕРКА 3: пробуем пинг =====
        try:
            await collection.database.command('ping')
            print("✅ [AD-AUDIO-UPLOAD] MongoDB ping OK")
        except Exception as ping_err:
            print(f"❌ [AD-AUDIO-UPLOAD] MongoDB ping FAILED: {ping_err}")
            raise HTTPException(status_code=503, detail=f"MongoDB not reachable: {ping_err}")
        
        # ===== ПРОВЕРКА 4: сколько уже документов =====
        try:
            before_count = await collection.count_documents({})
            print(f"📊 [AD-AUDIO-UPLOAD] Документов ДО вставки: {before_count}")
        except Exception as cnt_err:
            print(f"⚠️ [AD-AUDIO-UPLOAD] Не удалось посчитать документы: {cnt_err}")
            before_count = -1
        
        # ===== ПОДГОТОВКА ДАННЫХ =====
        ad_id = str(uuid.uuid4())
        ad_data = {
            "id": ad_id,
            "name": name if name else f"Аудио реклама {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "file_data": file_data,
            "duration": duration,
            "is_active": request.get("is_active", True),
            "created_at": datetime.utcnow(),
            "play_count": 0,
            "created_by": moderator["username"],
            # 🆕 Поля периода показа
            "show_from": None,        # ISO дата-время начала показа (UTC)
            "show_until": None,       # ISO дата-время окончания (UTC)
            "period_preset": "always" # 'day' | 'week' | 'month' | 'year' | 'always' | 'custom'
        }
        
        # Проверка размера документа
        doc_size = len(json.dumps(ad_data, default=str))
        print(f"📦 [AD-AUDIO-UPLOAD] Размер документа: {doc_size} bytes ({doc_size/1024/1024:.2f} MB)")
        if doc_size > 16 * 1024 * 1024:
            print(f"❌ [AD-AUDIO-UPLOAD] ДОКУМЕНТ > 16 MB! MongoDB отвергнет")
            raise HTTPException(status_code=413, detail="Document too large for MongoDB (max 16MB)")
        
        # ===== INSERT =====
        print(f"💾 [AD-AUDIO-UPLOAD] Вставляем в БД...")
        try:
            result = await collection.insert_one(ad_data)
            print(f"✅ [AD-AUDIO-UPLOAD] INSERT OK! inserted_id={result.inserted_id}")
        except Exception as ins_err:
            print(f"❌ [AD-AUDIO-UPLOAD] INSERT FAILED: {ins_err}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Insert failed: {ins_err}")
        
        # ===== ПРОВЕРКА ПОСЛЕ ВСТАВКИ =====
        try:
            after_count = await collection.count_documents({})
            print(f"📊 [AD-AUDIO-UPLOAD] Документов ПОСЛЕ вставки: {after_count}")
            if after_count <= before_count:
                print(f"⚠️ [AD-AUDIO-UPLOAD] КОЛИЧЕСТВО НЕ УВЕЛИЧИЛОСЬ! Что-то не так")
        except Exception as cnt_err:
            print(f"⚠️ [AD-AUDIO-UPLOAD] Не удалось посчитать после: {cnt_err}")
        
        # Проверяем что запись реально есть
        try:
            check = await collection.find_one({"id": ad_id})
            if check:
                print(f"✅ [AD-AUDIO-UPLOAD] Проверка: ЗАПИСЬ НАЙДЕНА в БД")
            else:
                print(f"❌ [AD-AUDIO-UPLOAD] Проверка: ЗАПИСЬ НЕ НАЙДЕНА! Загадка")
        except Exception as find_err:
            print(f"⚠️ [AD-AUDIO-UPLOAD] Ошибка проверки: {find_err}")
        
        print(f"✅ [AD-AUDIO-UPLOAD] ============ END SUCCESS ============")
        print("=" * 60)
        return {"status": "success", "ad_id": ad_id, "message": "Аудио реклама загружена"}
    
    except HTTPException:
        print(f"❌ [AD-AUDIO-UPLOAD] HTTPException")
        print("=" * 60)
        raise
    except Exception as e:
        print(f"❌ [AD-AUDIO-UPLOAD] НЕОЖИДАННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.delete("/ads/audio/{ad_id}")
async def delete_audio_ad(ad_id: str, moderator: dict = Depends(authenticate_admin)):
    print(f"🗑️ [AD-AUDIO-DELETE] ad_id={ad_id}, moderator={moderator.get('username')}")
    try:
        collection = getattr(app.state, 'audio_ads_collection', None)
        if collection is None:
            print("❌ [AD-AUDIO-DELETE] Коллекция недоступна")
            raise HTTPException(status_code=503, detail="Collection not available")
        
        # Сначала проверим, что запись существует
        existing = await collection.find_one({"id": ad_id})
        print(f"🗑️ [AD-AUDIO-DELETE] Найдено перед удалением: {'ДА' if existing else 'НЕТ'}")
        
        if not existing:
            raise HTTPException(status_code=404, detail="Ad not found")
        
        # Удаляем
        result = await collection.delete_one({"id": ad_id})
        print(f"🗑️ [AD-AUDIO-DELETE] deleted_count={result.deleted_count}")
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Ad not deleted (0 rows)")
        
        return {"status": "success", "message": "Аудио реклама удалена"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AD-AUDIO-DELETE] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.post("/ads/video/upload")
async def upload_video_ad(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        file_data = request.get("file_data", "")
        name = request.get("name", "").strip()
        duration = request.get("duration", 30)
        if not file_data:
            raise HTTPException(status_code=400, detail="File data is required")
        if not hasattr(app.state, 'video_ads_collection') or app.state.video_ads_collection is None:
            raise HTTPException(status_code=503, detail="Database collection not available")
        ad_id = str(uuid.uuid4())
        ad_data = {
            "id": ad_id,
            "name": name if name else f"Видео реклама {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "file_data": file_data,
            "duration": duration,
            "is_active": request.get("is_active", True),
            "created_at": datetime.utcnow(),
            "play_count": 0,
            "created_by": moderator["username"],
            "show_from": None,        # ISO дата-время начала показа (UTC)
            "show_until": None,       # ISO дата-время окончания (UTC)
            "period_preset": "always" # 'day' | 'week' | 'month' | 'year' | 'always' | 'custom'
        }
        await app.state.video_ads_collection.insert_one(ad_data)
        return {"status": "success", "ad_id": ad_id, "message": "Видео реклама загружена"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.delete("/ads/video/{ad_id}")
async def delete_video_ad(ad_id: str, moderator: dict = Depends(authenticate_admin)):
    print(f"🗑️ [AD-VIDEO-DELETE] ad_id={ad_id}, moderator={moderator.get('username')}")
    try:
        collection = getattr(app.state, 'video_ads_collection', None)
        if collection is None:
            raise HTTPException(status_code=503, detail="Collection not available")
        
        existing = await collection.find_one({"id": ad_id})
        if not existing:
            raise HTTPException(status_code=404, detail="Ad not found")
        
        result = await collection.delete_one({"id": ad_id})
        print(f"🗑️ [AD-VIDEO-DELETE] deleted_count={result.deleted_count}")
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Ad not deleted")
        
        return {"status": "success", "message": "Видео реклама удалена"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AD-VIDEO-DELETE] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ПРЕМИУМ ЭНДПОИНТЫ
# ============================================================
@admin_router.get("/premium/stats")
async def get_premium_stats(moderator: dict = Depends(authenticate_admin)):
    try:
        active_count = 0
        total_revenue = 0
        if hasattr(app.state, 'premium_collection') and app.state.premium_collection:
            active_count = await app.state.premium_collection.count_documents({"is_active": True})
        if hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection:
            revenue_data = await app.state.video_orders_collection.aggregate([
                {"$match": {"type": "premium", "status": "completed"}},
                {"$group": {"_id": None, "total": {"$sum": "$price"}}}
            ]).to_list(length=1)
            total_revenue = revenue_data[0]['total'] if revenue_data else 0
        premium_settings = await get_premium_settings()
        return {
            "active_premium_users": active_count,
            "total_revenue": total_revenue,
            "premium_price": premium_settings.get("premium_price", 499)
        }
    except Exception as e:
        return {"active_premium_users": 0, "total_revenue": 0, "premium_price": 499}

@admin_router.get("/premium/users")
async def get_premium_users(moderator: dict = Depends(authenticate_admin), limit: int = Query(50, ge=1, le=200)):
    print(f"📋 [PREMIUM-USERS] Загрузка списка, limit={limit}")
    try:
        collection = getattr(app.state, 'premium_collection', None)
        if collection is None:
            print("⚠️ [PREMIUM-USERS] Коллекция недоступна")
            return {"users": [], "total": 0}

        # 🔧 FIX: НЕ тянем большие поля + обязательно лимит
        cursor = collection.find(
            {"is_active": True},
            {
                "_id": 0,
                # Исключаем потенциально большие поля:
                "file_data": 0,
                "avatar": 0,
                "photos": 0,
                "metadata": 0,
            }
        ).limit(limit)

        users = await cursor.to_list(length=limit)
        print(f"✅ [PREMIUM-USERS] Загружено {len(users)} пользователей")

        # Сортируем на стороне Python (лёгкий список)
        try:
            users.sort(key=lambda x: str(x.get("start_date", "")), reverse=True)
        except Exception:
            pass

        total = await collection.count_documents({"is_active": True})

        return {
            "users": serialize_mongo_doc(users),
            "total": total,
            "limit": limit
        }
    except Exception as e:
        print(f"❌ [PREMIUM-USERS] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return {"users": [], "total": 0, "error": str(e)}

@admin_router.get("/premium/settings")
async def get_premium_settings_endpoint(moderator: dict = Depends(authenticate_admin)):
    try:
        return await get_premium_settings()
    except Exception as e:
        return {"premium_price": 499.0, "premium_duration_days": 30, "disable_ads": True, "priority_support": True, "extended_session_time": 3600}

@admin_router.post("/premium/settings")
async def update_premium_settings(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        settings = {
            "premium_price": request.get("premium_price", 499.0),
            "premium_duration_days": request.get("premium_duration_days", 30),
            "disable_ads": request.get("disable_ads", True),
            "priority_support": request.get("priority_support", True),
            "extended_session_time": request.get("extended_session_time", 3600),
            "updated_at": datetime.utcnow(),
            "updated_by": moderator["username"]
        }
        if hasattr(app.state, 'premium_settings_collection') and app.state.premium_settings_collection:
            await app.state.premium_settings_collection.update_one({"key": "premium_settings"}, {"$set": settings}, upsert=True)
        return {"status": "success", "settings": settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.post("/premium/activate/{user_id}")
async def admin_activate_premium(user_id: str, request: dict, moderator: dict = Depends(authenticate_admin)):
    print(f"👑 [PREMIUM-ACTIVATE] ========== START ==========")
    print(f"👑 [PREMIUM-ACTIVATE] user_id={user_id}")
    print(f"👑 [PREMIUM-ACTIVATE] request={request}")
    print(f"👑 [PREMIUM-ACTIVATE] moderator={moderator.get('username')}")
    
    try:
        duration_days = request.get("duration_days", 30)
        try:
            duration_days = int(duration_days)
        except (ValueError, TypeError):
            print(f"❌ [PREMIUM-ACTIVATE] duration_days не число: {request.get('duration_days')}")
            raise HTTPException(status_code=400, detail="duration_days must be a number")
        
        if duration_days <= 0:
            print(f"❌ [PREMIUM-ACTIVATE] duration_days <= 0")
            raise HTTPException(status_code=400, detail="duration_days must be positive")
        
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Проверяем коллекцию
        collection = getattr(app.state, 'premium_collection', None)
        print(f"👑 [PREMIUM-ACTIVATE] premium_collection={collection}")
        if collection is None:
            print(f"❌ [PREMIUM-ACTIVATE] premium_collection = None")
            raise HTTPException(status_code=503, detail="Premium collection not available")
        
        order_id = f"admin_{uuid.uuid4().hex[:8]}"
        print(f"👑 [PREMIUM-ACTIVATE] order_id={order_id}, duration={duration_days}")
        
        try:
            await activate_premium(user_id, duration_days, order_id, "admin")
            print(f"✅ [PREMIUM-ACTIVATE] Активирован")
        except Exception as act_err:
            print(f"❌ [PREMIUM-ACTIVATE] Ошибка activate_premium: {act_err}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Activate error: {act_err}")
        
        # Проверяем что записалось
        try:
            count = await collection.count_documents({"user_id": user_id, "is_active": True})
            print(f"📊 [PREMIUM-ACTIVATE] Активных премиумов у {user_id}: {count}")
        except Exception as cnt_err:
            print(f"⚠️ [PREMIUM-ACTIVATE] Не удалось посчитать: {cnt_err}")
        
        print(f"✅ [PREMIUM-ACTIVATE] ========== SUCCESS ==========")
        return {
            "status": "success",
            "message": f"Премиум активирован для {user_id} на {duration_days} дней",
            "user_id": user_id,
            "duration_days": duration_days,
            "order_id": order_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [PREMIUM-ACTIVATE] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.delete("/premium/deactivate/{user_id}")
async def admin_deactivate_premium(user_id: str, moderator: dict = Depends(authenticate_admin)):
    print(f"🚫 [PREMIUM-DEACTIVATE] user_id={user_id}, moderator={moderator.get('username')}")
    try:
        collection = getattr(app.state, 'premium_collection', None)
        if collection is None:
            print("❌ [PREMIUM-DEACTIVATE] Коллекция недоступна")
            raise HTTPException(status_code=503, detail="Premium collection not available")
        
        # Проверяем, что есть активные
        active_before = await collection.count_documents({"user_id": user_id, "is_active": True})
        print(f"📊 [PREMIUM-DEACTIVATE] Активных до: {active_before}")
        
        if active_before == 0:
            print(f"⚠️ [PREMIUM-DEACTIVATE] Нет активных записей у {user_id}")
            return {"status": "success", "message": f"Премиум уже не активен для {user_id}"}
        
        # Деактивируем ВСЕ активные записи (может быть несколько)
        result = await collection.update_many(
            {"user_id": user_id, "is_active": True},
            {"$set": {
                "is_active": False,
                "deactivated_at": datetime.utcnow(),
                "deactivated_by": moderator["username"]
            }}
        )
        print(f"✅ [PREMIUM-DEACTIVATE] Обновлено записей: {result.modified_count}")
        
        # Проверяем результат
        active_after = await collection.count_documents({"user_id": user_id, "is_active": True})
        print(f"📊 [PREMIUM-DEACTIVATE] Активных после: {active_after}")
        
        if active_after > 0:
            print(f"❌ [PREMIUM-DEACTIVATE] Остались активные записи! Что-то не так")
            raise HTTPException(status_code=500, detail=f"Failed to deactivate all. Still {active_after} active")
        
        return {
            "status": "success",
            "message": f"Премиум деактивирован для {user_id}",
            "deactivated_count": result.modified_count
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [PREMIUM-DEACTIVATE] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ПЛАТЕЖНЫЕ ЭНДПОИНТЫ
# ============================================================
@admin_router.get("/payment/settings")
async def get_payment_settings_admin(moderator: dict = Depends(authenticate_admin)):
    try:
        payment_stats = {"total_orders": 0, "successful_orders": 0, "failed_orders": 0, "pending_orders": 0, "total_revenue": 0, "today_revenue": 0}
        if hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection is not None:
            try:
                total_orders = await app.state.video_orders_collection.count_documents({})
                successful_orders = await app.state.video_orders_collection.count_documents({"status": "completed"})
                failed_orders = await app.state.video_orders_collection.count_documents({"status": "failed"})
                pending_orders = await app.state.video_orders_collection.count_documents({"status": "waiting_payment"})
                revenue_data = await app.state.video_orders_collection.aggregate([
                    {"$match": {"status": "completed"}},
                    {"$group": {"_id": None, "total": {"$sum": "$price"}}}
                ]).to_list(length=1)
                total_revenue = revenue_data[0]['total'] if revenue_data else 0
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                today_revenue_data = await app.state.video_orders_collection.aggregate([
                    {"$match": {"status": "completed", "paid_at": {"$gte": today_start}}},
                    {"$group": {"_id": None, "total": {"$sum": "$price"}}}
                ]).to_list(length=1)
                today_revenue = today_revenue_data[0]['total'] if today_revenue_data else 0
                payment_stats = {
                    "total_orders": total_orders,
                    "successful_orders": successful_orders,
                    "failed_orders": failed_orders,
                    "pending_orders": pending_orders,
                    "total_revenue": total_revenue,
                    "today_revenue": today_revenue
                }
            except Exception as e:
                print(f"⚠️ Error calculating payment stats: {e}")
        db_settings = {}
        if hasattr(app.state, 'system_settings_collection') and app.state.system_settings_collection is not None:
            db_settings_doc = await app.state.system_settings_collection.find_one({"key": "video_payment_settings"})
            if db_settings_doc:
                db_settings = db_settings_doc
        settings = {
            "morune_shop_id": db_settings.get("morune_shop_id") or PAYMENT_CONFIG.MORUNE_SHOP_ID,
            "morune_api_key": db_settings.get("morune_api_key") or PAYMENT_CONFIG.MORUNE_API_KEY,
            "morune_secret_key": "[СКРЫТО]" if db_settings.get("morune_secret_key") or PAYMENT_CONFIG.MORUNE_SECRET_KEY else "",
            "morune_payment_url": db_settings.get("morune_payment_url") or PAYMENT_CONFIG.MORUNE_PAYMENT_URL,
            "backend_url": PAYMENT_CONFIG.BACKEND_URL,
            "test_mode": db_settings.get("test_mode", PAYMENT_CONFIG.TEST_MODE),
            "currency": PAYMENT_CONFIG.CURRENCY,
            "payment_timeout": PAYMENT_CONFIG.PAYMENT_TIMEOUT,
            "statistics": payment_stats
        }
        return settings
    except Exception as e:
        logger.error(f"Error getting payment settings: {e}")
        return {"error": str(e), "status": "error", "test_mode": True}

@admin_router.post("/payment/settings")
async def update_payment_settings_admin(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        print(f"💳 [PAYMENT] Updating settings by {moderator['username']}")
        PAYMENT_CONFIG.MORUNE_SHOP_ID = request.get("morune_shop_id", PAYMENT_CONFIG.MORUNE_SHOP_ID)
        PAYMENT_CONFIG.MORUNE_API_KEY = request.get("morune_api_key", PAYMENT_CONFIG.MORUNE_API_KEY)
        PAYMENT_CONFIG.MORUNE_SECRET_KEY = request.get("morune_secret_key", PAYMENT_CONFIG.MORUNE_SECRET_KEY)
        PAYMENT_CONFIG.MORUNE_PAYMENT_URL = request.get("morune_payment_url", PAYMENT_CONFIG.MORUNE_PAYMENT_URL)
        PAYMENT_CONFIG.TEST_MODE = request.get("test_mode", PAYMENT_CONFIG.TEST_MODE)
        if hasattr(app.state, 'system_settings_collection') and app.state.system_settings_collection:
            payment_settings = {
                "key": "video_payment_settings",
                "morune_shop_id": PAYMENT_CONFIG.MORUNE_SHOP_ID,
                "morune_api_key": PAYMENT_CONFIG.MORUNE_API_KEY,
                "morune_secret_key": PAYMENT_CONFIG.MORUNE_SECRET_KEY,
                "morune_payment_url": PAYMENT_CONFIG.MORUNE_PAYMENT_URL,
                "test_mode": PAYMENT_CONFIG.TEST_MODE,
                "updated_at": datetime.utcnow(),
                "updated_by": moderator["username"]
            }
            await app.state.system_settings_collection.update_one({"key": "video_payment_settings"}, {"$set": payment_settings}, upsert=True)
        return {"status": "success", "message": "✅ Настройки оплаты сохранены", "test_mode": PAYMENT_CONFIG.TEST_MODE}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения настроек: {str(e)}")

@admin_router.get("/payment/test")
async def test_payment_connection_admin(moderator: dict = Depends(authenticate_admin)):
    return await test_morune_connection()

@admin_router.post("/payment/create-test")
async def create_test_payment_admin(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        order_id = f"admin_test_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        user_id = request.get("user_id", f"admin_test_{moderator['username']}")
        tariff_minutes = request.get("minutes", 15)
        price = request.get("price", 100.0)
        await manager.activate_video_mode(user_id, tariff_minutes)
        return {"status": "success", "order_id": order_id, "user_id": user_id, "minutes": tariff_minutes, "price": price, "test_mode": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка создания тестового заказа: {str(e)}")

# ============================================================
# УПРАВЛЕНИЕ ПЕРСОНАЛОМ (АДМИНЫ И МОДЕРАТОРЫ)
# ============================================================
@admin_router.get("/staff/permissions-list")
async def get_permissions_list(moderator: dict = Depends(authenticate_admin)):
    """Список всех возможных прав"""
    return {
        "permissions": ALL_MODERATOR_PERMISSIONS,
        "roles": ["admin", "moderator"]
    }

@admin_router.get("/staff/list")
async def get_staff_list(moderator: dict = Depends(authenticate_admin)):
    """Список всех админов и модераторов"""
    try:
        if not hasattr(app.state, 'moderators_collection') or app.state.moderators_collection is None:
            return {"staff": [], "total": 0}
        cursor = app.state.moderators_collection.find(
            {},
            {"password_hash": 0}
        ).sort("created_at", 1)
        staff = await cursor.to_list(length=200)
        return {
            "staff": serialize_mongo_doc(staff),
            "total": len(staff),
            "permissions": ALL_MODERATOR_PERMISSIONS
        }
    except Exception as e:
        print(f"❌ [STAFF] list error: {e}")
        return {"staff": [], "total": 0, "error": str(e)}

@admin_router.post("/staff/create")
async def create_staff(request: StaffCreateRequest, moderator: dict = Depends(authenticate_admin)):
    """Создать нового админа/модератора"""
    try:
        if not hasattr(app.state, 'moderators_collection') or app.state.moderators_collection is None:
            raise HTTPException(status_code=503, detail="Moderators collection not available")

        username = request.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="Username is required")

        existing = await app.state.moderators_collection.find_one({"username": username})
        if existing:
            raise HTTPException(status_code=409, detail="Пользователь с таким логином уже существует")

        role = request.role if request.role in ("admin", "moderator") else "moderator"

        # Валидация прав
        valid_perms = [p for p in (request.permissions or []) if p in ALL_MODERATOR_PERMISSIONS]
        if role == "admin":
            # У админа — все права
            valid_perms = list(ALL_MODERATOR_PERMISSIONS)

        password_hash = bcrypt.hashpw(
            request.password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        staff_data = {
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "permissions": valid_perms,
            "is_active": bool(request.is_active),
            "created_at": datetime.utcnow(),
            "created_by": moderator["username"]
        }
        result = await app.state.moderators_collection.insert_one(staff_data)

        await log_moderation_action(
            moderator["username"], "create_staff", username,
            f"Создан {role} с правами: {', '.join(valid_perms) or 'нет'}"
        )

        print(f"✅ [STAFF] Created {role} '{username}' by {moderator['username']}")

        return {
            "status": "success",
            "message": f"✅ {role.capitalize()} '{username}' создан",
            "id": str(result.inserted_id),
            "username": username,
            "role": role
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [STAFF] create error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.put("/staff/{username}")
async def update_staff(username: str, request: StaffUpdateRequest, moderator: dict = Depends(authenticate_admin)):
    """Обновить данные админа/модератора (пароль, роль, права, активность)"""
    try:
        if not hasattr(app.state, 'moderators_collection') or app.state.moderators_collection is None:
            raise HTTPException(status_code=503, detail="Moderators collection not available")

        existing = await app.state.moderators_collection.find_one({"username": username})
        if not existing:
            raise HTTPException(status_code=404, detail="Пользователь не найден")

        # Защита: нельзя понизить последнего активного админа
        if existing.get("role") == "admin" and request.role == "moderator":
            admin_count = await app.state.moderators_collection.count_documents(
                {"role": "admin", "is_active": True}
            )
            if admin_count <= 1:
                raise HTTPException(status_code=400, detail="Нельзя понизить последнего администратора")

        if existing.get("role") == "admin" and request.is_active is False:
            admin_count = await app.state.moderators_collection.count_documents(
                {"role": "admin", "is_active": True}
            )
            if admin_count <= 1:
                raise HTTPException(status_code=400, detail="Нельзя деактивировать последнего администратора")

        update_data = {"updated_at": datetime.utcnow(), "updated_by": moderator["username"]}

        if request.password:
            update_data["password_hash"] = bcrypt.hashpw(
                request.password.encode('utf-8'),
                bcrypt.gensalt()
            ).decode('utf-8')
            update_data["password_changed_at"] = datetime.utcnow()

        if request.role is not None:
            if request.role not in ("admin", "moderator"):
                raise HTTPException(status_code=400, detail="Роль должна быть admin или moderator")
            update_data["role"] = request.role

        if request.is_active is not None:
            update_data["is_active"] = bool(request.is_active)

        # Права
        final_role = update_data.get("role", existing.get("role", "moderator"))
        if request.permissions is not None:
            valid_perms = [p for p in request.permissions if p in ALL_MODERATOR_PERMISSIONS]
            update_data["permissions"] = valid_perms
        elif request.role == "admin":
            update_data["permissions"] = list(ALL_MODERATOR_PERMISSIONS)
        elif final_role == "admin":
            update_data["permissions"] = list(ALL_MODERATOR_PERMISSIONS)

        await app.state.moderators_collection.update_one(
            {"username": username},
            {"$set": update_data}
        )

        await log_moderation_action(
            moderator["username"], "update_staff", username,
            f"Обновлён: {', '.join(update_data.keys())}"
        )

        print(f"✅ [STAFF] Updated '{username}' by {moderator['username']}")

        return {
            "status": "success",
            "message": f"✅ '{username}' обновлён",
            "username": username,
            "updated_fields": list(update_data.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [STAFF] update error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.post("/staff/{username}/password")
async def change_staff_password(username: str, request: ChangePasswordRequest, moderator: dict = Depends(authenticate_admin)):
    """Сменить пароль админу/модератору"""
    try:
        if not hasattr(app.state, 'moderators_collection') or app.state.moderators_collection is None:
            raise HTTPException(status_code=503, detail="Moderators collection not available")

        existing = await app.state.moderators_collection.find_one({"username": username})
        if not existing:
            raise HTTPException(status_code=404, detail="Пользователь не найден")

        password_hash = bcrypt.hashpw(
            request.new_password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        await app.state.moderators_collection.update_one(
            {"username": username},
            {"$set": {
                "password_hash": password_hash,
                "password_changed_at": datetime.utcnow(),
                "password_changed_by": moderator["username"]
            }}
        )

        await log_moderation_action(
            moderator["username"], "change_staff_password", username,
            "Пароль изменён"
        )

        print(f"✅ [STAFF] Password changed for '{username}' by {moderator['username']}")

        return {"status": "success", "message": f"✅ Пароль для '{username}' изменён"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [STAFF] password error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.delete("/staff/{username}")
async def delete_staff(username: str, moderator: dict = Depends(authenticate_admin)):
    """Удалить админа/модератора"""
    try:
        if not hasattr(app.state, 'moderators_collection') or app.state.moderators_collection is None:
            raise HTTPException(status_code=503, detail="Moderators collection not available")

        # Нельзя удалить себя
        if username == moderator["username"]:
            raise HTTPException(status_code=400, detail="Нельзя удалить самого себя")

        existing = await app.state.moderators_collection.find_one({"username": username})
        if not existing:
            raise HTTPException(status_code=404, detail="Пользователь не найден")

        # Защита: нельзя удалить последнего активного админа
        if existing.get("role") == "admin":
            admin_count = await app.state.moderators_collection.count_documents(
                {"role": "admin", "is_active": True}
            )
            if admin_count <= 1:
                raise HTTPException(status_code=400, detail="Нельзя удалить последнего администратора")

        result = await app.state.moderators_collection.delete_one({"username": username})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Не удалось удалить")

        await log_moderation_action(
            moderator["username"], "delete_staff", username,
            f"Удалён {existing.get('role', 'staff')}"
        )

        print(f"✅ [STAFF] Deleted '{username}' by {moderator['username']}")

        return {"status": "success", "message": f"✅ '{username}' удалён"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [STAFF] delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))        

# ============================================================
# СИНХРОНИЗАЦИЯ
# ============================================================
@admin_router.get("/sync/stats")
async def get_sync_stats(moderator: dict = Depends(authenticate_admin)):
    print("📊 [SYNC-STATS] Getting sync statistics...")
    async def operation():
        try:
            if not hasattr(app.state, 'db') or app.state.db is None:
                return {"db_connected": False, "users_in_db": 0, "connections_in_db": 0, "rooms_in_db": 0, "users_in_memory": len(manager.user_sessions), "rooms_in_memory": len(manager.active_rooms)}
            users_in_db = 0
            connections_in_db = 0
            rooms_in_db = 0
            if hasattr(app.state, 'users_collection') and app.state.users_collection is not None:
                try:
                    users_in_db = await app.state.users_collection.count_documents({})
                except Exception as e:
                    print(f"❌ [SYNC-STATS] Error counting users: {e}")
            if hasattr(app.state, 'connections_collection') and app.state.connections_collection is not None:
                try:
                    connections_in_db = await app.state.connections_collection.count_documents({})
                except Exception as e:
                    print(f"❌ [SYNC-STATS] Error counting connections: {e}")
            if hasattr(app.state, 'rooms_collection') and app.state.rooms_collection is not None:
                try:
                    rooms_in_db = await app.state.rooms_collection.count_documents({})
                except Exception as e:
                    print(f"❌ [SYNC-STATS] Error counting rooms: {e}")
            return {
                "db_connected": True,
                "users_in_db": users_in_db,
                "connections_in_db": connections_in_db,
                "rooms_in_db": rooms_in_db,
                "users_in_memory": len(manager.user_sessions),
                "rooms_in_memory": len(manager.active_rooms),
                "active_connections": len(manager.active_connections)
            }
        except Exception as e:
            print(f"❌ [SYNC-STATS] Error in sync stats operation: {e}")
            return {"db_connected": False, "users_in_db": 0, "connections_in_db": 0, "rooms_in_db": 0, "users_in_memory": len(manager.user_sessions), "rooms_in_memory": len(manager.active_rooms), "error": str(e)}
    return await safe_db_operation(operation, {"db_connected": False, "users_in_db": 0, "connections_in_db": 0, "rooms_in_db": 0, "users_in_memory": len(manager.user_sessions), "rooms_in_memory": len(manager.active_rooms)}, "get_sync_stats")

@admin_router.post("/disconnect")
async def admin_disconnect(request: DisconnectRequest, moderator: dict = Depends(authenticate_admin)):
    await manager.force_disconnect_user(request.user_id)
    return {"status": "success", "message": f"User {request.user_id} disconnected"}

@admin_router.post("/broadcast")
async def admin_broadcast(request: BroadcastRequest, moderator: dict = Depends(authenticate_admin)):
    await manager.broadcast_message({
        "type": "system_message",
        "message": request.message,
        "timestamp": datetime.utcnow().isoformat(),
        "from_admin": True
    })
    return {"status": "success", "message": "Broadcast sent"}

@admin_router.post("/emergency-stop")
async def emergency_stop(moderator: dict = Depends(authenticate_admin)):
    for user_id in list(manager.active_connections.keys()):
        await manager.force_disconnect_user(user_id)
    return {"status": "success", "message": "All users disconnected"}

@admin_router.post("/block")
async def block_user_endpoint(request: BlockUserRequest, moderator: dict = Depends(authenticate_admin)):
    await block_user(request.user_id, request.reason, request.duration_hours)
    if request.user_id in manager.active_connections:
        await manager.force_disconnect_user(request.user_id)
    return {"status": "success", "message": f"User {request.user_id} blocked successfully"}

@admin_router.post("/unblock/{user_id}")
async def unblock_user(user_id: str, moderator: dict = Depends(authenticate_admin)):
    async def operation():
        result = await app.state.blocked_users_collection.delete_one({"user_id": user_id})
        if result.deleted_count > 0:
            return {"status": "success", "message": f"User {user_id} unblocked"}
        else:
            raise HTTPException(status_code=404, detail="User not found or not blocked")
    return await safe_db_operation(operation, {"status": "success", "message": f"User {user_id} would be unblocked"}, "unblock_user")

@admin_router.get("/blocked")
async def get_blocked_users(moderator: dict = Depends(authenticate_admin)):
    async def operation():
        now = datetime.utcnow()
        
        # ===== ЕДИНЫЙ ФИЛЬТР: показываем только активные баны =====
        active_filter = {
            "$or": [
                {"is_permanent": True},
                {"expires_at": {"$gt": now}},
                {"expires_at": None}
            ]
        }
        
        result = {
            "blocked_users": [],
            "blocked_ips": [],
            "blocked_browsers": [],
            "blocked_fingerprints": [],
            "blocked_subnets": [],
            "total": 0
        }
        
        # ===== Баны по ID =====
        if hasattr(app.state, 'blocked_users_collection') and app.state.blocked_users_collection is not None:
            try:
                blocked_users = await app.state.blocked_users_collection.find(active_filter).to_list(200)
                result["blocked_users"] = serialize_mongo_doc(blocked_users)
            except Exception as e:
                print(f"⚠️ [BLOCKED] users error: {e}")
        
        # ===== Баны по IP =====
        if hasattr(app.state, 'blocked_ips_collection') and app.state.blocked_ips_collection is not None:
            try:
                blocked_ips = await app.state.blocked_ips_collection.find(active_filter).to_list(200)
                result["blocked_ips"] = serialize_mongo_doc(blocked_ips)
            except Exception as e:
                print(f"⚠️ [BLOCKED] ips error: {e}")
        
        # ===== Баны по Browser =====
        if hasattr(app.state, 'blocked_browsers_collection') and app.state.blocked_browsers_collection is not None:
            try:
                blocked_browsers = await app.state.blocked_browsers_collection.find(active_filter).to_list(200)
                result["blocked_browsers"] = serialize_mongo_doc(blocked_browsers)
            except Exception as e:
                print(f"⚠️ [BLOCKED] browsers error: {e}")
        
        # ===== Баны по Fingerprint =====
        if hasattr(app.state, 'blocked_fingerprints_collection') and app.state.blocked_fingerprints_collection is not None:
            try:
                blocked_fingerprints = await app.state.blocked_fingerprints_collection.find(active_filter).to_list(200)
                result["blocked_fingerprints"] = serialize_mongo_doc(blocked_fingerprints)
            except Exception as e:
                print(f"⚠️ [BLOCKED] fingerprints error: {e}")
        
        # ===== Баны по Subnet =====
        if hasattr(app.state, 'blocked_subnets_collection') and app.state.blocked_subnets_collection is not None:
            try:
                blocked_subnets = await app.state.blocked_subnets_collection.find(active_filter).to_list(200)
                result["blocked_subnets"] = serialize_mongo_doc(blocked_subnets)
            except Exception as e:
                print(f"⚠️ [BLOCKED] subnets error: {e}")
        
        # ===== Общее число =====
        result["total"] = (
            len(result["blocked_users"]) +
            len(result["blocked_ips"]) +
            len(result["blocked_browsers"]) +
            len(result["blocked_fingerprints"]) +
            len(result["blocked_subnets"])
        )
        
        print(f"✅ [BLOCKED] Loaded: users={len(result['blocked_users'])}, ips={len(result['blocked_ips'])}, browsers={len(result['blocked_browsers'])}, fps={len(result['blocked_fingerprints'])}, subnets={len(result['blocked_subnets'])}")
        
        return result
    
    return await safe_db_operation(operation, {
        "blocked_users": [], "blocked_ips": [], "blocked_browsers": [],
        "blocked_fingerprints": [], "blocked_subnets": [], "total": 0
    }, "get_blocked_users")

@admin_router.post("/unblock-ip/{ip:path}")
async def unblock_ip(ip: str, moderator: dict = Depends(authenticate_admin)):
    """Разблокировать IP адрес"""
    try:
        if not hasattr(app.state, 'blocked_ips_collection') or app.state.blocked_ips_collection is None:
            if app.state.db is not None:
                app.state.blocked_ips_collection = app.state.db["blocked_ips"]
        
        if app.state.blocked_ips_collection is not None:
            result = await app.state.blocked_ips_collection.delete_one({"ip": ip})
            if result.deleted_count > 0:
                return {"status": "success", "message": f"IP {ip} разблокирован"}
        raise HTTPException(status_code=404, detail="IP не найден в бане")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/unblock-browser/{browser_id}")
async def unblock_browser(browser_id: str, moderator: dict = Depends(authenticate_admin)):
    """Разблокировать Browser ID"""
    try:
        if not hasattr(app.state, 'blocked_browsers_collection') or app.state.blocked_browsers_collection is None:
            if app.state.db is not None:
                app.state.blocked_browsers_collection = app.state.db["blocked_browsers"]
        
        if app.state.blocked_browsers_collection is not None:
            result = await app.state.blocked_browsers_collection.delete_one({"browser_id": browser_id})
            if result.deleted_count > 0:
                return {"status": "success", "message": f"Browser {browser_id[:8]}... разблокирован"}
        raise HTTPException(status_code=404, detail="Browser не найден в бане")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/unblock-fingerprint/{fingerprint}")
async def unblock_fingerprint(fingerprint: str, moderator: dict = Depends(authenticate_admin)):
    """Разблокировать Fingerprint"""
    try:
        if not hasattr(app.state, 'blocked_fingerprints_collection') or app.state.blocked_fingerprints_collection is None:
            if app.state.db is not None:
                app.state.blocked_fingerprints_collection = app.state.db["blocked_fingerprints"]
        
        if app.state.blocked_fingerprints_collection is not None:
            result = await app.state.blocked_fingerprints_collection.delete_one({"fingerprint": fingerprint})
            if result.deleted_count > 0:
                return {"status": "success", "message": f"Fingerprint {fingerprint[:8]}... разблокирован"}
        raise HTTPException(status_code=404, detail="Fingerprint не найден в бане")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@admin_router.post("/unblock-subnet/{subnet:path}")
async def unblock_subnet(subnet: str, moderator: dict = Depends(authenticate_admin)):
    """Разблокировать подсеть"""
    try:
        if not hasattr(app.state, 'blocked_subnets_collection') or app.state.blocked_subnets_collection is None:
            if app.state.db is not None:
                app.state.blocked_subnets_collection = app.state.db["blocked_subnets"]
        
        if app.state.blocked_subnets_collection is not None:
            result = await app.state.blocked_subnets_collection.delete_one({"subnet": subnet})
            if result.deleted_count > 0:
                return {"status": "success", "message": f"Подсеть {subnet} разблокирована"}
        raise HTTPException(status_code=404, detail="Подсеть не найдена в бане")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    

@admin_router.get("/settings")
async def get_settings(moderator: dict = Depends(authenticate_admin)):
    if not hasattr(app.state, 'system_settings'):
        app.state.system_settings = {
            "maintenance_mode": False,
            "max_users": 1000,
            "allow_registration": True,
            "system_message": "Добро пожаловать в голосовой чат!",
            "muffle_enabled": False,
            "maintenance_message": "Ведутся технические работы.",
            "auto_moderation_enabled": True,
            "max_warnings_before_ban": 3,
            "mute_duration_minutes": 60,
            "ban_duration_hours": 24
        }
    return app.state.system_settings

@admin_router.post("/settings")
async def update_settings(request: SystemSettingsRequest, moderator: dict = Depends(authenticate_admin)):
    if not hasattr(app.state, 'system_settings'):
        app.state.system_settings = {
            "maintenance_mode": False,
            "max_users": 1000,
            "allow_registration": True,
            "system_message": "Добро пожаловать в голосовой чат!",
            "muffle_enabled": False,
            "maintenance_message": "Ведутся технические работы.",
            "auto_moderation_enabled": True,
            "max_warnings_before_ban": 3,
            "mute_duration_minutes": 60,
            "ban_duration_hours": 24
        }
    current_settings = app.state.system_settings
    if request.maintenance_mode is not None:
        current_settings["maintenance_mode"] = request.maintenance_mode
    if request.max_users is not None:
        current_settings["max_users"] = request.max_users
    if request.allow_registration is not None:
        current_settings["allow_registration"] = request.allow_registration
    if request.system_message is not None:
        current_settings["system_message"] = request.system_message
    if request.muffle_enabled is not None:
        current_settings["muffle_enabled"] = request.muffle_enabled
        await manager.broadcast_message({"type": "muffle_status", "enabled": request.muffle_enabled})
    app.state.system_settings = current_settings
    return {"status": "success", "settings": current_settings}

@admin_router.post("/muffle/toggle")
async def toggle_muffle(moderator: dict = Depends(authenticate_admin)):
    app.state.system_settings["muffle_enabled"] = not app.state.system_settings["muffle_enabled"]
    await manager.broadcast_message({
        "type": "muffle_status",
        "enabled": app.state.system_settings["muffle_enabled"],
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"status": "success", "muffle_enabled": app.state.system_settings["muffle_enabled"]}

@admin_router.post("/queues/clear")
async def clear_queues(request: ClearQueuesRequest, moderator: dict = Depends(authenticate_admin)):
    cleared_count = await manager.clear_search_queues(request.categories)
    return {"status": "success", "message": f"Cleared {cleared_count} users from queues", "cleared_count": cleared_count}

@admin_router.get("/system/status")
async def get_system_status(moderator: dict = Depends(authenticate_admin)):
    stats = manager.get_online_stats()
    system_stats = get_system_stats()
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "system_settings": app.state.system_settings,
        "server_uptime": system_stats["uptime"],
        "server_uptime_seconds": system_stats["uptime_seconds"],
        "connections_count": len(manager.active_connections),
        "memory_usage": system_stats["memory_usage"],
        "cpu_usage": system_stats["cpu_usage"],
        "system_memory": {"used": system_stats["system_memory_used"], "total": system_stats["system_memory_total"], "percent": system_stats["system_memory_percent"]},
        "disk_usage": {"used": system_stats["disk_used"], "total": system_stats["disk_total"], "percent": system_stats["disk_percent"]},
        "process_stats": system_stats
    }

@admin_router.post("/maintenance")
async def set_maintenance_mode(request: MaintenanceRequest, moderator: dict = Depends(authenticate_admin)):
    app.state.system_settings["maintenance_mode"] = request.enabled
    if request.message:
        app.state.system_settings["maintenance_message"] = request.message
    if request.enabled:
        await manager.broadcast_message({
            "type": "maintenance_mode",
            "enabled": True,
            "message": request.message or app.state.system_settings["maintenance_message"]
        })
    return {"status": "success", "enabled": request.enabled, "message": request.message or app.state.system_settings["maintenance_message"]}

@admin_router.get("/maintenance")
async def get_maintenance_mode(moderator: dict = Depends(authenticate_admin)):
    return {"enabled": app.state.system_settings.get("maintenance_mode", False), "message": app.state.system_settings.get("maintenance_message", "")}

@admin_router.get("/video/stats")
async def get_video_stats_admin(moderator: dict = Depends(authenticate_admin)):
    try:
        stats = manager.get_video_stats()
        total_orders = 0
        total_revenue = 0
        if hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection is not None:
            total_orders = await app.state.video_orders_collection.count_documents({})
            revenue_data = await app.state.video_orders_collection.aggregate([
                {"$match": {"status": "completed"}},
                {"$group": {"_id": None, "total": {"$sum": "$price"}}}
            ]).to_list(length=1)
            if revenue_data:
                total_revenue = revenue_data[0].get('total', 0)
        stats.update({"total_orders": total_orders, "total_revenue": total_revenue})
        return stats
    except Exception as e:
        logger.error(f"Error getting video stats: {e}")
        return {"video_mode_enabled": manager.video_mode_enabled, "active_video_users": 0, "total_orders": 0, "total_revenue": 0}

@admin_router.get("/video/tariffs")
async def get_video_tariffs_admin(moderator: dict = Depends(authenticate_admin)):
    try:
        if not hasattr(manager, 'video_tariffs') or not manager.video_tariffs:
            manager.video_tariffs = [
                {"id": "tariff_1", "name": "15 минут", "minutes": 15, "price": 100.0, "is_active": True},
                {"id": "tariff_2", "name": "30 минут", "minutes": 30, "price": 180.0, "is_active": True},
                {"id": "tariff_3", "name": "60 минут", "minutes": 60, "price": 300.0, "is_active": True},
                {"id": "tariff_4", "name": "Навсегда", "minutes": 0, "price": 1000.0, "is_active": True}
            ]
        return {"status": "success", "tariffs": manager.video_tariffs, "total": len(manager.video_tariffs)}
    except Exception as e:
        return {"status": "success", "tariffs": [], "total": 0, "warning": str(e)}

@admin_router.post("/video/tariffs")
async def create_video_tariff(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        name = request.get("name", "").strip()
        price = request.get("price")
        minutes = request.get("minutes", 15)
        if not name:
            raise HTTPException(status_code=400, detail="Tariff name is required")
        if price is None or float(price) <= 0:
            raise HTTPException(status_code=400, detail="Valid price is required")
        tariff_id = f"tariff_{uuid.uuid4().hex[:8]}"
        tariff_data = {
            "id": tariff_id,
            "name": name,
            "minutes": int(minutes),
            "price": float(price),
            "is_active": bool(request.get("is_active", True)),
            "created_at": datetime.utcnow().isoformat(),
            "created_by": moderator["username"]
        }
        if hasattr(app.state, 'video_tariffs_collection') and app.state.video_tariffs_collection:
            await app.state.video_tariffs_collection.update_one({"id": tariff_id}, {"$set": tariff_data}, upsert=True)
        manager.video_tariffs.append(tariff_data)
        return {"status": "success", "message": "Tariff created", "tariff": tariff_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating tariff: {str(e)}")

@admin_router.put("/video/tariffs/{tariff_id}")
async def update_video_tariff(tariff_id: str, request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        name = request.get("name", "").strip()
        price = request.get("price")
        minutes = request.get("minutes", 15)
        if not name:
            raise HTTPException(status_code=400, detail="Tariff name is required")
        if price is None or float(price) <= 0:
            raise HTTPException(status_code=400, detail="Valid price is required")
        for i, tariff in enumerate(manager.video_tariffs):
            if tariff.get("id") == tariff_id:
                manager.video_tariffs[i].update({
                    "name": name,
                    "minutes": int(minutes),
                    "price": float(price),
                    "is_active": bool(request.get("is_active", True)),
                    "updated_at": datetime.utcnow().isoformat(),
                    "updated_by": moderator["username"]
                })
                updated_tariff = manager.video_tariffs[i]
                if hasattr(app.state, 'video_tariffs_collection') and app.state.video_tariffs_collection:
                    await app.state.video_tariffs_collection.update_one({"id": tariff_id}, {"$set": updated_tariff}, upsert=True)
                return {"status": "success", "message": "Tariff updated", "tariff": updated_tariff}
        raise HTTPException(status_code=404, detail="Tariff not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating tariff: {str(e)}")



@admin_router.delete("/video/tariffs/{tariff_id}")
async def delete_video_tariff(tariff_id: str, moderator: dict = Depends(authenticate_admin)):
    try:
        for i, tariff in enumerate(manager.video_tariffs):
            if tariff.get("id") == tariff_id:
                deleted_tariff = manager.video_tariffs.pop(i)
                if hasattr(app.state, 'video_tariffs_collection') and app.state.video_tariffs_collection:
                    await app.state.video_tariffs_collection.delete_one({"id": tariff_id})
                return {"status": "success", "message": "Tariff deleted", "deleted_tariff": deleted_tariff}
        raise HTTPException(status_code=404, detail="Tariff not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting tariff: {str(e)}")

@admin_router.get("/video/orders")
async def get_video_orders_admin(moderator: dict = Depends(authenticate_admin), limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0)):
    try:
        if not hasattr(app.state, 'video_orders_collection') or not app.state.video_orders_collection:
            return {"orders": []}
        cursor = app.state.video_orders_collection.find({}).sort("created_at", -1).skip(offset).limit(limit)
        orders = await cursor.to_list(length=limit)
        return {"orders": serialize_mongo_doc(orders)}
    except Exception as e:
        return {"orders": []}

@admin_router.post("/video/toggle-global")
async def toggle_global_video_mode(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        enabled = request.get("enabled", True)
        await manager.toggle_global_video_mode(enabled)
        return {"status": "success", "message": f"Global video mode {'enabled' if enabled else 'disabled'}", "video_mode_enabled": enabled}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error toggling video mode")

@admin_router.get("/video/data")
async def get_video_data(moderator: dict = Depends(authenticate_admin)):
    try:
        stats = manager.get_video_stats()
        video_users = []
        for user_id, user_session in manager.user_sessions.items():
            if manager.is_video_mode_active(user_id):
                video_users.append({
                    "user_id": user_id,
                    "room_id": user_session.get("room_id"),
                    "status": user_session.get("status"),
                    "video_mode_active": True,
                    "video_time_left": manager.get_video_time_left(user_id),
                    "last_activity": user_session.get("last_activity")
                })
        total_orders = 0
        recent_orders = []
        if hasattr(app.state, 'video_orders_collection') and app.state.video_orders_collection:
            total_orders = await app.state.video_orders_collection.count_documents({})
            cursor = app.state.video_orders_collection.find({}).sort("created_at", -1).limit(10)
            recent_orders = await cursor.to_list(length=10)
        return {
            "status": "success",
            "stats": stats,
            "video_users": video_users,
            "total_orders": total_orders,
            "recent_orders": serialize_mongo_doc(recent_orders),
            "tariffs": manager.video_tariffs
        }
    except Exception as e:
        return {"status": "error", "stats": manager.get_video_stats(), "video_users": [], "total_orders": 0, "recent_orders": [], "tariffs": manager.video_tariffs}

@admin_router.get("/messages/recent")
async def get_recent_messages(
    moderator: dict = Depends(authenticate_admin),
    limit: int = Query(10, ge=1, le=50)
):
    try:
        messages = []
        messages_collection = getattr(app.state, 'messages_log_collection', None)
        if messages_collection is not None:
            try:
                cursor = messages_collection.find().sort("sent_at", -1).limit(limit)
                messages = await cursor.to_list(length=limit)
                messages = serialize_mongo_doc(messages)
                print(f"📨 [MESSAGES] Loaded {len(messages)} real messages from DB")
            except Exception as e:
                print(f"⚠️ [MESSAGES] DB load error: {e}")
        
        if not messages:
            messages = [
                {
                    "type": "system",
                    "title": "📢 Нет сообщений",
                    "text": "Отправьте первое сообщение через форму слева",
                    "recipients_count": 0,
                    "sent_at": datetime.utcnow().isoformat(),
                    "message_type": "system"
                }
            ]
        
        return messages
        
    except Exception as e:
        print(f"❌ [MESSAGES] Error: {e}")
        import traceback
        traceback.print_exc()
        return [
            {
                "type": "error",
                "title": "❌ Ошибка загрузки",
                "text": str(e),
                "recipients_count": 0,
                "sent_at": datetime.utcnow().isoformat(),
                "message_type": "error"
            }
        ]

@admin_router.post("/messages/send")
async def send_admin_message(request: dict, moderator: dict = Depends(authenticate_admin)):
    try:
        print(f"📨 [MESSAGE] Received from {moderator['username']}")
        
        message_type = request.get("type", "normal")
        title = request.get("title", "")
        text = request.get("text", "")
        recipient = request.get("recipient", "all")
        specific_user_id = request.get("specific_user_id")
        
        message_data = {
            "type": "admin_message",
            "message_type": message_type,
            "title": title,
            "text": text,
            "from_admin": moderator["username"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        SERVICE_IDS = ['chat', 'moderation', 'admin', 'system']
        recipients_count = 0
        real_users = []
        
        for user_id in manager.active_connections.keys():
            if user_id in SERVICE_IDS:
                continue
            if user_id.startswith(('chat_', 'mod_', 'admin_')):
                continue
            real_users.append(user_id)
        
        if recipient == "all":
            for user_id in real_users:
                await manager.send_personal_message(message_data, user_id)
            recipients_count = len(real_users)
            print(f"📨 [MESSAGE] Broadcast to {recipients_count} real users")
            
        elif recipient == "specific" and specific_user_id:
            if specific_user_id in manager.active_connections and specific_user_id not in SERVICE_IDS:
                await manager.send_personal_message(message_data, specific_user_id)
                recipients_count = 1
                
        elif recipient == "searching":
            for user_id, session in manager.user_sessions.items():
                if user_id in SERVICE_IDS:
                    continue
                if session.get("status") == UserStatus.SEARCHING:
                    await manager.send_personal_message(message_data, user_id)
                    recipients_count += 1
                    
        elif recipient == "connected":
            for user_id, session in manager.user_sessions.items():
                if user_id in SERVICE_IDS:
                    continue
                if session.get("status") == UserStatus.CONNECTED:
                    await manager.send_personal_message(message_data, user_id)
                    recipients_count += 1
        
        try:
            messages_collection = getattr(app.state, 'messages_log_collection', None)
            if messages_collection is not None:
                log_entry = {
                    "type": message_type,
                    "title": title,
                    "text": text,
                    "recipient": recipient,
                    "specific_user_id": specific_user_id,
                    "recipients_count": recipients_count,
                    "from_admin": moderator["username"],
                    "sent_at": datetime.utcnow(),
                    "timestamp_str": datetime.utcnow().isoformat()
                }
                await messages_collection.insert_one(log_entry)
                print(f"✅ [MESSAGE] Saved to DB: {title}")
        except Exception as db_error:
            print(f"⚠️ [MESSAGE] DB save error: {db_error}")
        
        return {
            "status": "success",
            "message": "Message sent",
            "recipients_count": recipients_count
        }
        
    except Exception as e:
        print(f"❌ [MESSAGE] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@admin_router.get("/logs")
async def get_server_logs(moderator: dict = Depends(authenticate_admin), limit: int = Query(50, ge=1, le=1000)):
    try:
        logs = []
        if hasattr(app.state, 'moderation_logs_collection') and app.state.moderation_logs_collection:
            cursor = app.state.moderation_logs_collection.find().sort("timestamp", -1).limit(limit)
            db_logs = await cursor.to_list(length=limit)
            for log in db_logs:
                logs.append({
                    "timestamp": log.get("timestamp", datetime.utcnow()),
                    "message": f"{log.get('moderator', 'system')} {log.get('action', 'action')} {log.get('target_user', 'user')} - {log.get('reason', '')}",
                    "type": "moderation"
                })
        system_logs = [
            {"timestamp": datetime.utcnow() - timedelta(minutes=5), "message": f"Система запущена. Онлайн: {len(manager.active_connections)}", "type": "system"},
            {"timestamp": datetime.utcnow() - timedelta(minutes=10), "message": f"Активные комнаты: {len(manager.active_rooms)}", "type": "system"}
        ]
        logs.extend(system_logs)
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"logs": serialize_mongo_doc(logs[:limit])}
    except Exception as e:
        return {"logs": [{"timestamp": datetime.utcnow(), "message": f"Сервер работает. Онлайн: {len(manager.active_connections)}", "type": "system"}], "error": str(e)}

@admin_router.get("/reports")
async def get_reports(moderator: dict = Depends(authenticate_admin), limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
    async def operation():
        reports = await app.state.reports_collection.find().sort("timestamp", -1).skip(offset).limit(limit).to_list(limit)
        total = await app.state.reports_collection.count_documents({})
        return {"reports": serialize_mongo_doc(reports), "total": total, "limit": limit, "offset": offset}
    return await safe_db_operation(operation, {"reports": [], "total": 0, "limit": limit, "offset": offset}, "get_reports")

@admin_router.post("/reports/{report_id}/resolve")
async def resolve_report(report_id: str, resolve_request: dict, moderator: dict = Depends(authenticate_admin)):
    valid_actions = ['no_action', 'warn', 'mute', 'ban']
    action = resolve_request.get("action")
    if action not in valid_actions:
        raise HTTPException(status_code=422, detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}")
    if not ObjectId.is_valid(report_id):
        raise HTTPException(status_code=422, detail="Invalid report ID format")
    
    async def operation():
        report_object_id = ObjectId(report_id)
        report = await app.state.reports_collection.find_one({"_id": report_object_id})
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        result = await app.state.reports_collection.update_one(
            {"_id": report_object_id},
            {"$set": {
                "resolved": True, 
                "resolved_by": moderator["username"], 
                "resolved_at": datetime.utcnow(), 
                "moderator_action": action, 
                "moderator_notes": resolve_request.get("moderator_notes")
            }}
        )
        
        if result.modified_count > 0:
            reporter_id = report.get("reporter_id")
            reported_id = report.get("reported_user_id")
            report_reason = report.get("reason", "нарушение правил")
            reason_ru = translate_reason(report_reason)
            
            # Если действие не "без действия" — применяем наказание
            if action != "no_action":
                if reported_id:
                    reason = f"Автоматическое действие по жалобе: {report_reason}"
                    if action == "warn":
                        await warn_user(reported_id, reason, "medium", moderator["username"])
                    elif action == "mute":
                        await mute_user(reported_id, reason, 60, moderator["username"])
                    elif action == "ban":
                        await ban_user(reported_id, reason, 24, moderator["username"])
            
            # 🆕 УВЕДОМЛЯЕМ ТОГО, КТО ПОДАВАЛ ЖАЛОБУ
            action_text_for_reporter = {
                "no_action": "Нарушение не подтверждено. Пользователь остался без наказания.",
                "warn": "Пользователь получил предупреждение.",
                "mute": "Пользователь заглушен на 60 минут.",
                "ban": "Пользователь забанен на 24 часа."
            }.get(action, "Жалоба обработана.")
            
            await send_or_queue_notification(reporter_id, {
                "type": "report_resolved",
                "icon": "✅",
                "title": "Ваша жалоба обработана",
                "message": f"Пользователь {reported_id[:12] if reported_id else '—'}...",
                "submessage": action_text_for_reporter,
                "timeout": 12000
            })
            
            # 🆕 УВЕДОМЛЯЕМ ТОГО, НА КОГО ЖАЛОВАЛИСЬ (только если действие применено)
            if action != "no_action" and reported_id:
                action_text_for_reported = {
                    "warn": "Вы получили предупреждение от модератора.",
                    "mute": "Вы заглушены на 60 минут.",
                    "ban": "Вы забанены на 24 часа."
                }.get(action, "Модератор применил к вам действие.")
                
                await send_or_queue_notification(reported_id, {
                    "type": "moderation_action",
                    "icon": "🛡️",
                    "title": "Действие модератора",
                    "message": action_text_for_reported,
                    "submessage": f"Причина: {reason_ru}",
                    "timeout": 12000
                })
            
            # Уведомляем модераторов (для их панели)
            await moderation_manager.broadcast({
                "type": "report_resolved", 
                "report_id": report_id, 
                "moderator": moderator["username"], 
                "action": action
            })
            return {"status": "success", "message": "Report resolved", "action": action}
        else:
            raise HTTPException(status_code=404, detail="Report not found or already resolved")
    
    return await safe_db_operation(operation, {"status": "error", "message": "Database not available"}, "resolve_report")

@admin_router.get("/health")
async def health_check_admin(moderator: dict = Depends(authenticate_admin)):
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "stats": manager.get_online_stats()}

async def get_moderation_stats_internal():
    async def operation():
        try:
            # ИСПРАВЛЕНО: проверяем на None, а не на булево значение
            if app.state.db is None:
                return {"db_connected": False, "active_reports": 0, "banned_users": 0, "muted_users": 0, "moderation_actions_today": 0, "active_warnings": 0}
            
            active_reports = 0
            if app.state.reports_collection is not None:
                active_reports = await app.state.reports_collection.count_documents({"$or": [{"resolved": {"$ne": True}}, {"resolved": {"$exists": False}}]})
            
            banned_users = 0
            if app.state.blocked_users_collection is not None:
                banned_users = await app.state.blocked_users_collection.count_documents({"$or": [{"expires_at": {"$gt": datetime.utcnow()}}, {"expires_at": None}]})
            
            muted_users = 0
            if app.state.mutes_collection is not None:
                muted_users = await app.state.mutes_collection.count_documents({"expires_at": {"$gt": datetime.utcnow()}})
            
            actions_today = 0
            if app.state.moderation_logs_collection is not None:
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                actions_today = await app.state.moderation_logs_collection.count_documents({"timestamp": {"$gte": today_start}})
            
            active_warnings = 0
            if app.state.warnings_collection is not None:
                active_warnings = await app.state.warnings_collection.count_documents({"expires_at": {"$gt": datetime.utcnow()}})
            
            return {
                "db_connected": True,
                "active_reports": active_reports,
                "banned_users": banned_users,
                "muted_users": muted_users,
                "moderation_actions_today": actions_today,
                "active_warnings": active_warnings
            }
        except Exception as e:
            print(f"❌ [ADMIN] Error in get_moderation_stats_internal: {e}")
            return {"db_connected": False, "active_reports": 0, "banned_users": 0, "muted_users": 0, "moderation_actions_today": 0, "active_warnings": 0}
    return await safe_db_operation(operation, {"db_connected": False, "active_reports": 0, "banned_users": 0, "muted_users": 0, "moderation_actions_today": 0, "active_warnings": 0}, "get_moderation_stats_internal")

# ============================================================
# МОДЕРАЦИЯ ЭНДПОИНТЫ
# ============================================================
@moderation_router.get("/reports/recent")
async def get_recent_reports(moderator: dict = Depends(authenticate_moderator), limit: int = Query(5, ge=1, le=20)):
    async def operation():
        if app.state.reports_collection is None:
            return []
        reports = await app.state.reports_collection.find({"$or": [{"resolved": {"$ne": True}}, {"resolved": {"$exists": False}}]}).sort("timestamp", -1).limit(limit).to_list(limit)
        return serialize_mongo_doc(reports)
    return await safe_db_operation(operation, [], "get_recent_reports")

@moderation_router.get("/reports")
async def get_reports_moderation(moderator: dict = Depends(authenticate_moderator), resolved: Optional[bool] = None, limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
    async def operation():
        query = {}
        if resolved is not None:
            query["resolved"] = resolved
        if app.state.reports_collection is None:
            return {"reports": [], "total": 0, "limit": limit, "offset": offset}
        cursor = app.state.reports_collection.find(query).sort("timestamp", -1).skip(offset).limit(limit)
        reports = await cursor.to_list(length=limit)
        total = await app.state.reports_collection.count_documents(query)
        return {"reports": serialize_mongo_doc(reports), "total": total, "limit": limit, "offset": offset}
    return await safe_db_operation(operation, {"reports": [], "total": 0, "limit": limit, "offset": offset}, "get_reports")

@moderation_router.post("/reports/{report_id}/resolve")
async def resolve_report_moderation(report_id: str, resolve_request: dict, moderator: dict = Depends(authenticate_moderator)):
    valid_actions = ['no_action', 'warn', 'mute', 'ban']
    action = resolve_request.get("action")
    if action not in valid_actions:
        raise HTTPException(status_code=422, detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}")
    if not ObjectId.is_valid(report_id):
        raise HTTPException(status_code=422, detail="Invalid report ID format")
    
    async def operation():
        report_object_id = ObjectId(report_id)
        report = await app.state.reports_collection.find_one({"_id": report_object_id})
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        result = await app.state.reports_collection.update_one(
            {"_id": report_object_id},
            {"$set": {
                "resolved": True, 
                "resolved_by": moderator["username"], 
                "resolved_at": datetime.utcnow(), 
                "moderator_action": action, 
                "moderator_notes": resolve_request.get("moderator_notes")
            }}
        )
        
        if result.modified_count > 0:
            reporter_id = report.get("reporter_id")
            reported_id = report.get("reported_user_id")
            report_reason = report.get("reason", "нарушение правил")
            reason_ru = translate_reason(report_reason)
            
            if action != "no_action":
                if reported_id:
                    reason = f"Автоматическое действие по жалобе: {report_reason}"
                    if action == "warn":
                        await warn_user(reported_id, reason, "medium", moderator["username"])
                    elif action == "mute":
                        await mute_user(reported_id, reason, 60, moderator["username"])
                    elif action == "ban":
                        await ban_user(reported_id, reason, 24, moderator["username"])
            
            # 🆕 Уведомляем того, кто подавал
            action_text_for_reporter = {
                "no_action": "Нарушение не подтверждено. Пользователь остался без наказания.",
                "warn": "Пользователь получил предупреждение.",
                "mute": "Пользователь заглушен на 60 минут.",
                "ban": "Пользователь забанен на 24 часа."
            }.get(action, "Жалоба обработана.")
            
            await send_or_queue_notification(reporter_id, {
                "type": "report_resolved",
                "icon": "✅",
                "title": "Ваша жалоба обработана",
                "message": f"Пользователь {reported_id[:12] if reported_id else '—'}...",
                "submessage": action_text_for_reporter,
                "timeout": 12000
            })
            
            # 🆕 Уведомляем того, на кого жаловались
            if action != "no_action" and reported_id:
                action_text_for_reported = {
                    "warn": "Вы получили предупреждение от модератора.",
                    "mute": "Вы заглушены на 60 минут.",
                    "ban": "Вы забанены на 24 часа."
                }.get(action, "Модератор применил к вам действие.")
                
                await send_or_queue_notification(reported_id, {
                    "type": "moderation_action",
                    "icon": "🛡️",
                    "title": "Действие модератора",
                    "message": action_text_for_reported,
                    "submessage": f"Причина: {reason_ru}",
                    "timeout": 12000
                })
            
            await moderation_manager.broadcast({
                "type": "report_resolved", 
                "report_id": report_id, 
                "moderator": moderator["username"], 
                "action": action
            })
            return {"status": "success", "message": "Report resolved", "action": action}
        
        raise HTTPException(status_code=404, detail="Report not found or already resolved")
    
    return await safe_db_operation(operation, {"status": "error", "message": "Database not available"}, "resolve_report")

@moderation_router.post("/warn")
async def warn_user_endpoint(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        data = await parse_moderation_request(request)
        user_id = data.get("user_id")
        reason = data.get("reason", "Нарушение правил")
        severity = data.get("severity", "medium")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        await warn_user(user_id, reason, severity, moderator["username"])
        await moderation_manager.broadcast({"type": "user_warned", "user_id": user_id, "moderator": moderator["username"], "reason": reason})
        return {"status": "success", "message": f"User {user_id} warned"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@moderation_router.post("/mute")
async def mute_user_endpoint(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        data = await parse_moderation_request(request)
        user_id = data.get("user_id")
        reason = data.get("reason", "Нарушение правил")
        duration_minutes = data.get("duration_minutes", 60)
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        await mute_user(user_id, reason, duration_minutes, moderator["username"])
        if user_id in manager.active_connections:
            await manager.send_personal_message({
                "type": "force_mute",
                "reason": f"Модератор применил заглушку: {reason}",
                "duration_minutes": duration_minutes
            }, user_id)
        await moderation_manager.broadcast({"type": "user_muted", "user_id": user_id, "moderator": moderator["username"], "reason": reason, "duration": duration_minutes})
        return {"status": "success", "message": f"User {user_id} muted for {duration_minutes} minutes", "user_id": user_id, "duration": duration_minutes}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================
# РАСШИРЕННЫЙ БАН ДЛЯ МОДЕРАТОРОВ
# ============================================================
@moderation_router.post("/advanced-ban")
async def moderation_advanced_ban(request: dict, moderator: dict = Depends(authenticate_moderator)):
    """
    Расширенный бан для модераторов.
    Требует права 'advanced_ban'.
    """
    # ===== Проверка права =====
    perms = moderator.get("permissions", [])
    if "advanced_ban" not in perms and moderator.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Нет права advanced_ban")

    try:
        user_id = request.get("user_id")
        reason = request.get("reason", "Нарушение правил")
        reason = translate_reason(reason)
        duration_hours = request.get("duration_hours", 24)
        ban_types = request.get("ban_types", {})

        print(f"🔒 [MOD-ADVANCED-BAN] user={user_id}, types={ban_types}, mod={moderator['username']}")

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        is_permanent = (duration_hours == 0)
        expires_at = None if is_permanent else (datetime.utcnow() + timedelta(hours=duration_hours))

        results = []
        banned_items = []

        # ===== 1. Бан по ID =====
        if ban_types.get("user", False):
            try:
                if not hasattr(app.state, 'blocked_users_collection') or app.state.blocked_users_collection is None:
                    if app.state.db is not None:
                        app.state.blocked_users_collection = app.state.db["blocked_users"]
                if app.state.blocked_users_collection is not None:
                    await app.state.blocked_users_collection.update_one(
                        {"user_id": user_id},
                        {"$set": {
                            "user_id": user_id, "reason": reason,
                            "banned_at": datetime.utcnow(), "expires_at": expires_at,
                            "is_permanent": is_permanent,
                            "banned_by": moderator["username"], "ban_type": "user"
                        }},
                        upsert=True
                    )
                    results.append("✅ ID заблокирован")
                    banned_items.append(f"user:{user_id}")
            except Exception as e:
                results.append(f"⚠️ ID: {e}")

        # ===== Получаем данные пользователя =====
        ip_address = None
        browser_id = None
        fingerprint = None

        if user_id in manager.user_sessions:
            s = manager.user_sessions[user_id]
            ip_address = s.get("ip_address")
            browser_id = s.get("browser_id")
            fingerprint = s.get("fingerprint")

        if not ip_address or not browser_id or not fingerprint:
            if hasattr(app.state, 'users_collection') and app.state.users_collection:
                try:
                    ud = await app.state.users_collection.find_one({"user_id": user_id})
                    if ud:
                        ip_address = ip_address or ud.get("ip_address")
                        browser_id = browser_id or ud.get("browser_id")
                        fingerprint = fingerprint or ud.get("fingerprint")
                except Exception:
                    pass

        if not ip_address and user_id in manager.user_ips:
            ip_address = manager.user_ips[user_id]

        # ===== 2. IP =====
        if ban_types.get("ip", False) and ip_address:
            try:
                if not hasattr(app.state, 'blocked_ips_collection') or app.state.blocked_ips_collection is None:
                    if app.state.db is not None:
                        app.state.blocked_ips_collection = app.state.db["blocked_ips"]
                if app.state.blocked_ips_collection is not None:
                    await app.state.blocked_ips_collection.update_one(
                        {"ip": ip_address},
                        {"$set": {
                            "ip": ip_address, "user_id": user_id, "reason": reason,
                            "banned_at": datetime.utcnow(), "expires_at": expires_at,
                            "is_permanent": is_permanent,
                            "banned_by": moderator["username"], "ban_type": "ip"
                        }},
                        upsert=True
                    )
                    results.append(f"✅ IP {ip_address} заблокирован")
                    banned_items.append(f"ip:{ip_address}")
            except Exception as e:
                results.append(f"⚠️ IP: {e}")

        # ===== 3. Browser =====
        if ban_types.get("browser", False) and browser_id:
            try:
                if not hasattr(app.state, 'blocked_browsers_collection') or app.state.blocked_browsers_collection is None:
                    if app.state.db is not None:
                        app.state.blocked_browsers_collection = app.state.db["blocked_browsers"]
                if app.state.blocked_browsers_collection is not None:
                    await app.state.blocked_browsers_collection.update_one(
                        {"browser_id": browser_id},
                        {"$set": {
                            "browser_id": browser_id, "user_id": user_id, "reason": reason,
                            "banned_at": datetime.utcnow(), "expires_at": expires_at,
                            "is_permanent": is_permanent,
                            "banned_by": moderator["username"], "ban_type": "browser"
                        }},
                        upsert=True
                    )
                    results.append(f"✅ Browser {browser_id[:8]}... заблокирован")
                    banned_items.append(f"browser:{browser_id}")
            except Exception as e:
                results.append(f"⚠️ Browser: {e}")

        # ===== 4. Fingerprint =====
        if ban_types.get("fingerprint", False) and fingerprint:
            try:
                if not hasattr(app.state, 'blocked_fingerprints_collection') or app.state.blocked_fingerprints_collection is None:
                    if app.state.db is not None:
                        app.state.blocked_fingerprints_collection = app.state.db["blocked_fingerprints"]
                if app.state.blocked_fingerprints_collection is not None:
                    await app.state.blocked_fingerprints_collection.update_one(
                        {"fingerprint": fingerprint},
                        {"$set": {
                            "fingerprint": fingerprint, "user_id": user_id, "reason": reason,
                            "banned_at": datetime.utcnow(), "expires_at": expires_at,
                            "is_permanent": is_permanent,
                            "banned_by": moderator["username"], "ban_type": "fingerprint"
                        }},
                        upsert=True
                    )
                    results.append(f"✅ Fingerprint {fingerprint[:8]}... заблокирован")
                    banned_items.append(f"fingerprint:{fingerprint}")
            except Exception as e:
                results.append(f"⚠️ Fingerprint: {e}")

        # ===== 5. Subnet =====
        if ban_types.get("subnet", False) and ip_address:
            try:
                parts = ip_address.split('.')
                if len(parts) >= 3:
                    subnet = f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
                    if not hasattr(app.state, 'blocked_subnets_collection') or app.state.blocked_subnets_collection is None:
                        if app.state.db is not None:
                            app.state.blocked_subnets_collection = app.state.db["blocked_subnets"]
                    if app.state.blocked_subnets_collection is not None:
                        await app.state.blocked_subnets_collection.update_one(
                            {"subnet": subnet},
                            {"$set": {
                                "subnet": subnet, "user_id": user_id, "reason": reason,
                                "banned_at": datetime.utcnow(), "expires_at": expires_at,
                                "is_permanent": is_permanent,
                                "banned_by": moderator["username"], "ban_type": "subnet"
                            }},
                            upsert=True
                        )
                        results.append(f"✅ Подсеть {subnet} заблокирована")
                        banned_items.append(f"subnet:{subnet}")
            except Exception as e:
                results.append(f"⚠️ Subnet: {e}")

        # ===== Отключаем пользователя =====
        if user_id in manager.active_connections:
            try:
                await manager.force_disconnect_user(user_id)
                results.append("✅ Пользователь отключён")
            except Exception as e:
                results.append(f"⚠️ Disconnect: {e}")

        # ===== Логирование =====
        try:
            await log_moderation_action(
                moderator["username"], "advanced_ban", user_id, reason,
                duration_hours * 60 if duration_hours > 0 else None
            )
        except Exception:
            pass

        return {
            "status": "success",
            "message": f"Пользователь {user_id} забанен (расширенно)",
            "results": results,
            "banned_items": banned_items,
            "user_id": user_id,
            "is_permanent": is_permanent,
            "banned_by": moderator["username"]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [MOD-ADVANCED-BAN] {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))        

@moderation_router.post("/unmute")
async def unmute_user_endpoint(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        data = await parse_moderation_request(request)
        user_id = data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        await unmute_user(user_id, moderator["username"])
        await moderation_manager.broadcast({"type": "user_unmuted", "user_id": user_id, "moderator": moderator["username"]})
        if user_id in manager.active_connections:
            await manager.send_personal_message({"type": "unmuted_by_moderator", "message": "Модератор снял с вас заглушку"}, user_id)
        return {"status": "success", "message": f"User {user_id} unmuted", "user_id": user_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@moderation_router.post("/kick")
async def kick_user_endpoint(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        data = await parse_moderation_request(request)
        user_id = data.get("user_id")
        reason = data.get("reason", "Нарушение правил")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        await kick_user(user_id, reason, moderator["username"])
        if user_id in manager.active_connections:
            await manager.force_disconnect_user(user_id)
        await moderation_manager.broadcast({"type": "user_kicked", "user_id": user_id, "moderator": moderator["username"], "reason": reason})
        return {"status": "success", "message": f"User {user_id} kicked"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@moderation_router.post("/ban")
async def ban_user_endpoint(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        data = await parse_moderation_request(request)
        user_id = data.get("user_id")
        reason = data.get("reason", "Нарушение правил")
        duration_hours = data.get("duration_hours", 24)
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        await ban_user(user_id, reason, duration_hours, moderator["username"])
        if user_id in manager.active_connections:
            await manager.force_disconnect_user(user_id)
        await moderation_manager.broadcast({"type": "user_banned", "user_id": user_id, "moderator": moderator["username"], "reason": reason, "duration": duration_hours})
        return {"status": "success", "message": f"User {user_id} banned for {duration_hours} hours", "user_id": user_id, "duration_hours": duration_hours}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@moderation_router.post("/unban")
async def unban_user_endpoint(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        data = await parse_moderation_request(request)
        user_id = data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        success = await unban_user(user_id, moderator["username"])
        if success:
            await moderation_manager.broadcast({"type": "user_unbanned", "user_id": user_id, "moderator": moderator["username"]})
            return {"status": "success", "message": f"User {user_id} unbanned", "user_id": user_id}
        else:
            raise HTTPException(status_code=404, detail="User not found or not banned")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@moderation_router.get("/user/{user_id}")
async def get_user_status_moderation(user_id: str, moderator: dict = Depends(authenticate_moderator)):
    try:
        status = await get_user_moderation_status(user_id)
        return serialize_mongo_doc(status)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@moderation_router.get("/stats")
async def get_moderation_stats_endpoint(moderator: dict = Depends(authenticate_moderator)):
    async def operation():
        if app.state.reports_collection is None:
            return {"total_reports": 0, "active_reports": 0, "banned_users": 0, "muted_users": 0, "active_warnings": 0, "moderation_actions_today": 0, "status": "success"}
        
        active_reports = await app.state.reports_collection.count_documents({"$or": [{"resolved": {"$ne": True}}, {"resolved": {"$exists": False}}]})
        total_reports = await app.state.reports_collection.count_documents({})
        
        banned_users = 0
        if app.state.blocked_users_collection is not None:
            banned_users = await app.state.blocked_users_collection.count_documents({"$or": [{"expires_at": {"$gt": datetime.utcnow()}}, {"expires_at": None}]})
        
        muted_users = 0
        if app.state.mutes_collection is not None:
            muted_users = await app.state.mutes_collection.count_documents({"expires_at": {"$gt": datetime.utcnow()}})
        
        active_warnings = 0
        if app.state.warnings_collection is not None:
            active_warnings = await app.state.warnings_collection.count_documents({"expires_at": {"$gt": datetime.utcnow()}})
        
        actions_today = 0
        if app.state.moderation_logs_collection is not None:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            actions_today = await app.state.moderation_logs_collection.count_documents({"timestamp": {"$gte": today_start}})
        
        return {
            "total_reports": total_reports,
            "active_reports": active_reports,
            "banned_users": banned_users,
            "muted_users": muted_users,
            "active_warnings": active_warnings,
            "moderation_actions_today": actions_today,
            "status": "success"
        }
    return await safe_db_operation(operation, {"total_reports": 0, "active_reports": 0, "banned_users": 0, "muted_users": 0, "active_warnings": 0, "moderation_actions_today": 0, "status": "error"}, "get_moderation_stats")

@moderation_router.get("/logs")
async def get_moderation_logs(moderator: dict = Depends(authenticate_moderator), limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    async def operation():
        if app.state.moderation_logs_collection is None:
            return {"logs": [], "total": 0, "limit": limit, "offset": offset}
        cursor = app.state.moderation_logs_collection.find().sort("timestamp", -1).skip(offset).limit(limit)
        logs = await cursor.to_list(length=limit)
        total = await app.state.moderation_logs_collection.count_documents({})
        return {"logs": serialize_mongo_doc(logs), "total": total, "limit": limit, "offset": offset}
    return await safe_db_operation(operation, {"logs": [], "total": 0, "limit": limit, "offset": offset}, "get_moderation_logs")

@moderation_router.get("/blocked")
async def get_blocked_users_moderation(moderator: dict = Depends(authenticate_moderator)):
    """Заблокированные пользователи (для панели модерации)."""
    async def operation():
        now = datetime.utcnow()
        active_filter = {
            "$or": [
                {"is_permanent": True},
                {"expires_at": {"$gt": now}},
                {"expires_at": None}
            ]
        }
        
        result = {
            "blocked_users": [],
            "blocked_ips": [],
            "blocked_browsers": [],
            "blocked_fingerprints": [],
            "blocked_subnets": [],
            "total": 0
        }
        
        # ===== Баны по ID =====
        if hasattr(app.state, 'blocked_users_collection') and app.state.blocked_users_collection is not None:
            try:
                blocked_users = await app.state.blocked_users_collection.find(active_filter).to_list(200)
                result["blocked_users"] = serialize_mongo_doc(blocked_users)
            except Exception as e:
                print(f"⚠️ [MOD-BLOCKED] users error: {e}")
        
        # ===== Баны по IP =====
        if hasattr(app.state, 'blocked_ips_collection') and app.state.blocked_ips_collection is not None:
            try:
                blocked_ips = await app.state.blocked_ips_collection.find(active_filter).to_list(200)
                result["blocked_ips"] = serialize_mongo_doc(blocked_ips)
            except Exception as e:
                print(f"⚠️ [MOD-BLOCKED] ips error: {e}")
        
        # ===== Баны по Browser =====
        if hasattr(app.state, 'blocked_browsers_collection') and app.state.blocked_browsers_collection is not None:
            try:
                blocked_browsers = await app.state.blocked_browsers_collection.find(active_filter).to_list(200)
                result["blocked_browsers"] = serialize_mongo_doc(blocked_browsers)
            except Exception as e:
                print(f"⚠️ [MOD-BLOCKED] browsers error: {e}")
        
        # ===== Баны по Fingerprint =====
        if hasattr(app.state, 'blocked_fingerprints_collection') and app.state.blocked_fingerprints_collection is not None:
            try:
                blocked_fps = await app.state.blocked_fingerprints_collection.find(active_filter).to_list(200)
                result["blocked_fingerprints"] = serialize_mongo_doc(blocked_fps)
            except Exception as e:
                print(f"⚠️ [MOD-BLOCKED] fingerprints error: {e}")
        
        # ===== Баны по Subnet =====
        if hasattr(app.state, 'blocked_subnets_collection') and app.state.blocked_subnets_collection is not None:
            try:
                blocked_subnets = await app.state.blocked_subnets_collection.find(active_filter).to_list(200)
                result["blocked_subnets"] = serialize_mongo_doc(blocked_subnets)
            except Exception as e:
                print(f"⚠️ [MOD-BLOCKED] subnets error: {e}")
        
        result["total"] = (
            len(result["blocked_users"]) +
            len(result["blocked_ips"]) +
            len(result["blocked_browsers"]) +
            len(result["blocked_fingerprints"]) +
            len(result["blocked_subnets"])
        )
        
        print(f"✅ [MOD-BLOCKED] users={len(result['blocked_users'])}, ips={len(result['blocked_ips'])}, browsers={len(result['blocked_browsers'])}, fps={len(result['blocked_fingerprints'])}, subnets={len(result['blocked_subnets'])}")
        
        return result
    
    return await safe_db_operation(operation, {
        "blocked_users": [], "blocked_ips": [], "blocked_browsers": [],
        "blocked_fingerprints": [], "blocked_subnets": [], "total": 0
    }, "get_blocked_users_moderation")    

@moderation_router.get("/system/stats")
async def get_unified_stats_endpoint(moderator: dict = Depends(authenticate_moderator)):
    try:
        basic_stats = manager.get_online_stats()
        moderation_stats = await get_moderation_stats_endpoint(moderator)
        return {**basic_stats, **moderation_stats, "server_status": "online", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"online_users": 0, "unique_users": 0, "active_rooms": 0, "searching_users": 0, "active_reports": 0, "banned_users": 0, "muted_users": 0, "server_status": "error", "error": str(e)}


# ============================================================
# ВЕБСОКЕТ ДЛЯ АДМИНСКОГО ЧАТА
# ============================================================

@app.websocket("/ws/chat")
async def admin_chat_websocket(websocket: WebSocket):
    """WebSocket чата для администраторов и модераторов"""
    print("🔴 [CHAT] New WebSocket connection")
    
    try:
        await websocket.accept()
        print("✅ [CHAT] WebSocket accepted")
    except Exception as e:
        print(f"❌ [CHAT] Failed to accept: {e}")
        return
    
    # Получаем параметры
    query_params = dict(websocket.query_params)
    source = query_params.get("source", "guest")
    username = query_params.get("username", "Гость")
    
    print(f"🔌 [CHAT] Source: {source}, Username: {username}")
    
    # ===== ЗАЩИТА ОТ ДУБЛИКАТОВ =====
    to_remove = []
    for conn in list(admin_chat_manager.active_connections):
        try:
            conn_username = getattr(conn, '_chat_username', None)
            if conn_username == username:
                print(f"⚠️ [CHAT] Closing duplicate for {username}")
                try:
                    await conn.close(code=1000, reason="Replaced")
                except:
                    pass
                to_remove.append(conn)
        except Exception as e:
            print(f"⚠️ [CHAT] Duplicate check error: {e}")
            to_remove.append(conn)
    
    for conn in to_remove:
        admin_chat_manager.active_connections.discard(conn)
    
    try:
        websocket._chat_username = username
    except:
        pass
    
    admin_chat_manager.active_connections.add(websocket)
    print(f"✅ [CHAT] Added. Total: {len(admin_chat_manager.active_connections)}")
    
    # ===== ОТПРАВЛЯЕМ ИСТОРИЮ =====
    try:
        history = admin_chat_messages[-100:] if admin_chat_messages else []
        await websocket.send_text(json.dumps({
            "type": "history",
            "messages": history
        }, default=str))
        print(f"📨 [CHAT] Sent history ({len(history)} messages)")
    except Exception as e:
        print(f"⚠️ [CHAT] History error: {e}")
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue
            
            msg_type = message.get("type")
            
            # ===== PING =====
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue
            
            # ===== HISTORY REQUEST =====
            if msg_type == "history_request":
                history = admin_chat_messages[-100:] if admin_chat_messages else []
                await websocket.send_text(json.dumps({
                    "type": "history",
                    "messages": history
                }, default=str))
                continue
            
            # ===== ОБЫЧНОЕ СООБЩЕНИЕ =====
            if msg_type == "message":
                chat_msg = {
                    "id": len(admin_chat_messages),
                    "sender": message.get("sender", "guest"),
                    "sender_name": message.get("sender_name", username),
                    "text": message.get("text", ""),
                    "timestamp": message.get("timestamp") or datetime.utcnow().isoformat(),
                    "type": "message"
                }
                
                print(f"📝 [CHAT] {chat_msg['sender_name']}: {chat_msg['text'][:60]}")
                
                admin_chat_messages.append(chat_msg)
                if len(admin_chat_messages) > ADMIN_CHAT_MAX_MESSAGES:
                    admin_chat_messages.pop(0)
                
                disconnected = []
                for conn in admin_chat_manager.active_connections.copy():
                    try:
                        await conn.send_text(json.dumps({
                            "type": "new_message",
                            "message": chat_msg
                        }, default=str))
                    except Exception as e:
                        print(f"⚠️ [CHAT] Broadcast error: {e}")
                        disconnected.append(conn)
                
                for conn in disconnected:
                    admin_chat_manager.active_connections.discard(conn)
                
                print(f"✅ [CHAT] Broadcast to {len(admin_chat_manager.active_connections)} clients")
                continue
            
            # ===== ГОЛОС =====
            if msg_type == "voice":
                chat_msg = {
                    "id": len(admin_chat_messages),
                    "sender": message.get("sender", "guest"),
                    "sender_name": message.get("sender_name", username),
                    "type": "voice",
                    "data": message.get("data", ""),
                    "duration": message.get("duration", ""),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                print(f"🎤 [CHAT] Voice from {chat_msg['sender_name']}")
                admin_chat_messages.append(chat_msg)
                if len(admin_chat_messages) > ADMIN_CHAT_MAX_MESSAGES:
                    admin_chat_messages.pop(0)
                
                for conn in admin_chat_manager.active_connections.copy():
                    try:
                        await conn.send_text(json.dumps({
                            "type": "new_message",
                            "message": chat_msg
                        }, default=str))
                    except:
                        pass
                continue
            
            # ===== ФАЙЛ =====
            if msg_type == "file":
                chat_msg = {
                    "id": len(admin_chat_messages),
                    "sender": message.get("sender", "guest"),
                    "sender_name": message.get("sender_name", username),
                    "type": "file",
                    "name": message.get("name", "file"),
                    "data": message.get("data", ""),
                    "mime": message.get("mime", "application/octet-stream"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                print(f"📎 [CHAT] File from {chat_msg['sender_name']}: {chat_msg['name']}")
                admin_chat_messages.append(chat_msg)
                if len(admin_chat_messages) > ADMIN_CHAT_MAX_MESSAGES:
                    admin_chat_messages.pop(0)
                
                for conn in admin_chat_manager.active_connections.copy():
                    try:
                        await conn.send_text(json.dumps({
                            "type": "new_message",
                            "message": chat_msg
                        }, default=str))
                    except:
                        pass
                continue
                
    except WebSocketDisconnect:
        print(f"🔌 [CHAT] {username} disconnected")
    except Exception as e:
        print(f"❌ [CHAT] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        admin_chat_manager.active_connections.discard(websocket)
        print(f"🔌 [CHAT] Cleaned. Total: {len(admin_chat_manager.active_connections)}")

@app.websocket("/ws/moderation")
async def moderation_websocket(websocket: WebSocket):
    await moderation_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        await moderation_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await moderation_manager.disconnect(websocket)


# ============================================================
# WEBSOCKET ЭНДПОИНТЫ
# ============================================================
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    try:
        browser_id = None
        ip_address = None
        fingerprint = None
        try:
            query_params = dict(websocket.query_params)
            browser_id = query_params.get('browser_id')
            fingerprint = query_params.get('fingerprint')
            ip_address = websocket.client.host if websocket.client else "unknown"
            if ip_address or fingerprint:
                await save_user_fingerprint(user_id, ip_address, fingerprint)
        except Exception as e:
            print(f"⚠️ [DEBUG] Error parsing WebSocket params: {e}")
            browser_id = None
            fingerprint = None
            ip_address = "unknown"
        
        await manager.connect(websocket, user_id, browser_id, ip_address)
        
        await manager.send_personal_message({
            "type": "global_video_mode_changed",
            "enabled": manager.video_mode_enabled,
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Глобальный видео-режим {'включен' if manager.video_mode_enabled else 'отключен'}"
        }, user_id)
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await manager.update_user_activity(user_id)
                
                # ============================================================
                # ===== ОСНОВНЫЕ ТИПЫ СООБЩЕНИЙ =====
                # ============================================================
                
                if message["type"] == "start_search":
                    search_params = message.get("search_params", {})
                    # 🆕 Логируем что пришло
                    print(f"🔍 [SEARCH] User {user_id} params: country={search_params.get('country')}, user_age={search_params.get('user_age')}, search_age={search_params.get('search_age')}")
                    await manager.start_search(user_id, search_params)
                
                elif message["type"] == "stop_search":
                    await manager.stop_search(user_id)
                
                elif message["type"] == "next_partner":
                    await manager.next_partner(user_id)
                
                elif message["type"] == "webrtc_signal":
                    await manager.handle_webrtc_signal(user_id, message["signal"])
                
                elif message["type"] == "disconnect":
                    await manager.handle_partner_disconnect(user_id, message.get("partner_id"))

                elif message["type"] == "client_cant_connect":
                    # 🆕 Клиент не смог подключиться (микрофон/камера) — освобождаем партнёра
                    partner_id = message.get("partner_id")
                    reason = message.get("reason", "unknown")
                    print(f"⚠️ [AUDIO] User {user_id} can't connect ({reason}), freeing partner {partner_id}")
                    
                    if partner_id:
                        # Освобождаем партнёра
                        await manager.handle_partner_disconnect(user_id, partner_id)
                        
                        # Убираем обоих из очередей поиска
                        for queue in manager.search_queues.values():
                            if user_id in queue:
                                queue.remove(user_id)
                            if partner_id in queue:
                                queue.remove(partner_id)
                        
                        # Уведомляем партнёра
                        if partner_id in manager.active_connections:
                            await manager.send_personal_message({
                                "type": "partner_disconnected",
                                "message": "Собеседник не смог подключиться (проблемы с микрофоном)"
                            }, partner_id)    
                
                elif message["type"] == "ping":
                    await manager.send_personal_message({"type": "pong", "online_stats": manager.get_online_stats()}, user_id)
                
                elif message["type"] == "get_online_stats":
                    await manager.send_personal_message({"type": "online_stats", "online_stats": manager.get_online_stats()}, user_id)
                
                elif message["type"] == "audio_data":
                    session = manager.user_sessions.get(user_id, {})
                    room_id = session.get("room_id")
                    if room_id and message.get("audio_chunk"):
                        await manager.handle_audio_data(user_id, room_id, message["audio_chunk"])
                
                # ============================================================
                # ===== ВИДЕО-РЕЖИМ (общие команды) =====
                # ============================================================
                
                elif message["type"] == "video_mode_request":
                    print(f"🎥 [VIDEO] Video mode request from {user_id}: {message}")
                    tariff_minutes = message.get("tariff_minutes")
                    if tariff_minutes is not None:
                        if not manager.video_mode_enabled:
                            await manager.send_personal_message({"type": "video_error", "message": "Видео-режим временно отключен администратором"}, user_id)
                            continue
                        if not manager.is_video_mode_active(user_id):
                            await manager.activate_video_mode(user_id, tariff_minutes)
                            print(f"✅ [VIDEO] Video mode activated for {user_id}, {tariff_minutes} minutes")
                        else:
                            time_left = manager.get_video_time_left(user_id)
                            await manager.send_personal_message({"type": "video_status", "is_active": True, "time_left": time_left}, user_id)
                    else:
                        await manager.send_personal_message({"type": "video_error", "message": "Не указана длительность тарифа"}, user_id)
                
                elif message["type"] == "get_video_status":
                    is_active = manager.is_video_mode_active(user_id)
                    time_left = manager.get_video_time_left(user_id)
                    await manager.send_personal_message({"type": "video_status", "is_active": is_active, "time_left": time_left}, user_id)
                
                elif message["type"] == "deactivate_video":
                    if manager.is_video_mode_active(user_id):
                        await manager.deactivate_video_mode(user_id)
                        await manager.send_personal_message({"type": "video_deactivated", "message": "Видео-режим деактивирован"}, user_id)
                
                # ============================================================
                # ===== ВИДЕО 1 НА 1 =====
                # ============================================================
                
                elif message["type"] == "join_video_room":
                    room_id = message.get("room_id")
                    mode = message.get("mode", "single")
                    print(f"🎥 [VIDEO-ROOM] User {user_id} joining room {room_id}, mode: {mode}")
                    
                    if room_id in manager.video_rooms:
                        room = manager.video_rooms[room_id]
                        if user_id not in room["participants"]:
                            room["participants"].append(user_id)
                            print(f"✅ [VIDEO-ROOM] User {user_id} added to room {room_id}")
                            
                            if user_id in manager.user_sessions:
                                manager.user_sessions[user_id]["status"] = UserStatus.CONNECTED
                                manager.user_sessions[user_id]["room_id"] = room_id
                            
                            await manager.broadcast_to_video_room(room_id, {
                                "type": "video_participant_joined",
                                "participant_id": user_id,
                                "participant_name": manager.user_sessions.get(user_id, {}).get("user_id", "User"),
                                "room_id": room_id,
                                "participants_count": len(room["participants"])
                            })
                            
                            await manager.send_personal_message({
                                "type": "video_room_participants",
                                "participants": room["participants"],
                                "room_id": room_id,
                                "mode": room.get("mode", "single")
                            }, user_id)
                    else:
                        print(f"🎥 [VIDEO-ROOM] Room {room_id} not found, creating new")
                        new_room_id = await manager.create_video_room(user_id, mode, [])
                        await manager.add_video_search_user(user_id, mode, new_room_id)
                        
                        await manager.send_personal_message({
                            "type": "video_room_created",
                            "room_id": new_room_id,
                            "mode": mode,
                            "message": "Комната создана, поиск участников..."
                        }, user_id)
                        
                        asyncio.create_task(manager.find_match_for_video_user(user_id, mode, new_room_id))
                
                elif message["type"] == "leave_video_room":
                    room_id = message.get("room_id")
                    if room_id:
                        await manager.leave_video_room(user_id, room_id)
                        await manager.broadcast_to_video_room(room_id, {
                            "type": "video_participant_left",
                            "participant_id": user_id,
                            "room_id": room_id
                        }, exclude_user=user_id)
                
                elif message["type"] == "video_signal":
                    signal_type = message.get("signal_type")
                    signal = message.get("signal")
                    room_id = message.get("room_id")
                    target_id = message.get("target_id")
                    
                    if room_id and target_id:
                        await manager.send_personal_message({
                            "type": "video_signal",
                            "signal_type": signal_type,
                            "signal": signal,
                            "from_user": user_id,
                            "room_id": room_id
                        }, target_id)
                        print(f"🎥 [VIDEO] Forwarded {signal_type} from {user_id} to {target_id} in room {room_id}")
                    elif room_id:
                        await manager.broadcast_to_video_room(room_id, {
                            "type": "video_signal",
                            "signal_type": signal_type,
                            "signal": signal,
                            "from_user": user_id,
                            "room_id": room_id
                        }, exclude_user=user_id)
                        print(f"🎥 [VIDEO] Broadcasted {signal_type} from {user_id} in room {room_id}")
                
                elif message["type"] == "video_message":
                    room_id = message.get("room_id")
                    text = message.get("message")
                    nickname = message.get("nickname", "User")

                    if room_id and text:
        # Ищем участников LiveKit-комнаты
                        room_info = livekit_room_manager.rooms.get(room_id)
                    if room_info:
                        for pid in list(room_info["users"]):
                            if pid != user_id and pid in manager.active_connections:
                             await manager.send_personal_message({
                                 "type": "video_message",
                                 "message": text,
                                 "user_id": user_id,
                                 "user_name": nickname,
                                 "nickname": nickname,
                                 "room_id": room_id,
                                 "timestamp": datetime.utcnow().isoformat()
                             }, pid)
                        print(f"💬 [VIDEO] Message from {user_id} in {room_id}: {text[:30]}... → {len(room_info['users'])-1} users")
                    else:
                        print(f"⚠️ [VIDEO] Room {room_id} not found in livekit_room_manager")                                         
                
                elif message["type"] == "video_search":
                    mode = message.get("mode", "single")
                    room_id = message.get("room_id")
                    if not room_id:
                        room_id = str(uuid.uuid4())
                    
                    await manager.create_video_room(user_id, mode, [])
                    await manager.add_video_search_user(user_id, mode, room_id)
                    
                    await manager.send_personal_message({
                        "type": "video_search_started",
                        "room_id": room_id,
                        "mode": mode,
                        "message": f"Поиск участников для видео-режима {'1 на 1' if mode == 'single' else '6 участников'}"
                    }, user_id)
                    
                    asyncio.create_task(manager.find_match_for_video_user(user_id, mode, room_id))
                
                elif message["type"] == "video_room_ended":
                    room_id = message.get("room_id")
                    if room_id:
                        await manager.leave_video_room(user_id, room_id)
                        await manager.broadcast_to_video_room(room_id, {
                            "type": "video_room_ended",
                            "ended_by": user_id,
                            "room_id": room_id
                        }, exclude_user=user_id)
                        print(f"🎥 [VIDEO] Room {room_id} ended by {user_id}")
                
                # ============================================================
                # ===== ГРУППОВОЕ ВИДЕО НА 6 УЧАСТНИКОВ =====
                # ============================================================
                
                elif message["type"] == "join_group_video_room":
                    room_id = message.get("room_id")
                    if not room_id:
                        room_id = str(uuid.uuid4())
                    
                    # Создаём или находим комнату на 6
                    if room_id not in manager.video_rooms:
                        manager.video_rooms[room_id] = {
                            "room_id": room_id,
                            "mode": "group6",
                            "participants": [user_id],
                            "created_at": datetime.utcnow().isoformat(),
                            "is_active": True
                        }
                        print(f"🎥 [GROUP-6] Created room {room_id}, first participant: {user_id}")
                    else:
                        if user_id not in manager.video_rooms[room_id]["participants"]:
                            manager.video_rooms[room_id]["participants"].append(user_id)
                            print(f"🎥 [GROUP-6] User {user_id} joined room {room_id}, total: {len(manager.video_rooms[room_id]['participants'])}")
                    
                    # Обновляем статус пользователя
                    if user_id in manager.user_sessions:
                        manager.user_sessions[user_id]["status"] = UserStatus.CONNECTED
                        manager.user_sessions[user_id]["room_id"] = room_id
                    
                    # Уведомляем всех в комнате, что присоединился новый участник
                    await manager.broadcast_to_video_room(room_id, {
                        "type": "group_video_6_participant_joined",
                        "participant_id": user_id,
                        "participant_name": manager.user_sessions.get(user_id, {}).get("nickname", user_id[:6]),
                        "room_id": room_id,
                        "participants_count": len(manager.video_rooms[room_id]["participants"])
                    }, exclude_user=user_id)
                    
                    # Отправляем пользователю список участников
                    await manager.send_personal_message({
                        "type": "video_room_participants",
                        "participants": manager.video_rooms[room_id]["participants"],
                        "room_id": room_id,
                        "mode": "group6"
                    }, user_id)
                
                elif message["type"] == "group_video_6_ended":
                    room_id = message.get("room_id")
                    if room_id and room_id in manager.video_rooms:
                        if user_id in manager.video_rooms[room_id]["participants"]:
                            manager.video_rooms[room_id]["participants"].remove(user_id)
                            print(f"🎥 [GROUP-6] User {user_id} left room {room_id}, remaining: {len(manager.video_rooms[room_id]['participants'])}")
                        
                        # Уведомляем оставшихся участников
                        await manager.broadcast_to_video_room(room_id, {
                            "type": "group_video_6_participant_left",
                            "participant_id": user_id,
                            "room_id": room_id,
                            "participants_left": len(manager.video_rooms[room_id]["participants"])
                        }, exclude_user=user_id)
                        
                        # Если комната пуста — удаляем
                        if len(manager.video_rooms[room_id]["participants"]) == 0:
                            del manager.video_rooms[room_id]
                            print(f"🎥 [GROUP-6] Room {room_id} deleted (empty)")
                    
                    # Убираем пользователя из очереди поиска
                    if user_id in manager.video_search_queues['group6']:
                        manager.video_search_queues['group6'].remove(user_id)
                
                elif message["type"] == "group_video_message":
                    room_id = message.get("room_id")
                    text = message.get("message")
                    nickname = message.get("nickname", "User")
                    if room_id and text:
                        room_info = livekit_room_manager.rooms.get(room_id)
                        if room_info:
                            for pid in list(room_info["users"]):
                                if pid != user_id and pid in manager.active_connections:
                                    await manager.send_personal_message({
                                        "type": "group_video_message",
                                        "message": text,
                                        "user_id": user_id,
                                        "user_name": nickname,
                                        "nickname": nickname,
                                        "room_id": room_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }, pid)
                            print(f"💬 [GROUP-6] Message from {user_id} in {room_id}: {text[:30]}... → {len(room_info['users'])-1} users")
                        else:
                            print(f"⚠️ [GROUP-6] Room {room_id} not found in livekit_room_manager")
                
                # ============================================================
                # ===== НИКНЕЙМ И ИНФО О КОМНАТЕ =====
                # ============================================================
                
                elif message["type"] == "set_nickname":
                    nickname = message.get("nickname")
                    if nickname:
                        if user_id in manager.user_sessions:
                            manager.user_sessions[user_id]["nickname"] = nickname
                        print(f"📝 [USER] Nickname set for {user_id}: {nickname}")
                        await manager.send_personal_message({
                            "type": "nickname_set",
                            "nickname": nickname,
                            "message": f"Ник установлен: {nickname}"
                        }, user_id)
                        
                        session = manager.user_sessions.get(user_id, {})
                        room_id = session.get("room_id")
                        if room_id and room_id in manager.video_rooms:
                            await manager.broadcast_to_video_room(room_id, {
                                "type": "video_participant_renamed",
                                "participant_id": user_id,
                                "new_name": nickname,
                                "room_id": room_id
                            }, exclude_user=user_id)
                
                elif message["type"] == "get_video_room_info":
                    room_id = message.get("room_id")
                    if room_id and room_id in manager.video_rooms:
                        room_info = manager.video_rooms[room_id]
                        await manager.send_personal_message({
                            "type": "video_room_info",
                            "room_id": room_id,
                            "mode": room_info.get("mode", "single"),
                            "participants": room_info.get("participants", []),
                            "participants_count": len(room_info.get("participants", [])),
                            "created_at": room_info.get("created_at")
                        }, user_id)
                    else:
                        await manager.send_personal_message({
                            "type": "video_room_info",
                            "room_id": room_id,
                            "exists": False,
                            "message": "Комната не найдена"
                        }, user_id)
                
                # ============================================================
                # ===== ПЛАТЕЖИ =====
                # ============================================================
                
                elif message["type"] == "check_payment_status":
                    pending_payment = await manager.get_pending_payment(user_id)
                    if pending_payment:
                        await manager.send_personal_message({
                            "type": "payment_status",
                            "status": "pending",
                            "order_id": pending_payment.get('order_id'),
                            "payment_url": pending_payment.get('payment_url'),
                            "time_left": 3600 - (time.time() - pending_payment.get('created_at', time.time()))
                        }, user_id)
                    else:
                        await manager.send_personal_message({
                            "type": "payment_status",
                            "status": "no_payment",
                            "message": "Нет ожидающих платежей"
                        }, user_id)
                
                elif message["type"] == "cancel_payment":
                    pending_payment = await manager.get_pending_payment(user_id)
                    if pending_payment:
                        await manager.remove_pending_payment(user_id)
                        await manager.send_personal_message({
                            "type": "payment_cancelled",
                            "order_id": pending_payment.get('order_id'),
                            "message": "Платеж отменен"
                        }, user_id)
                
        except WebSocketDisconnect:
            print(f"🔌 [DEBUG] WebSocket отключен: {user_id}")
            manager.disconnect(user_id)
        except Exception as e:
            print(f"❌ [DEBUG] WebSocket ошибка для {user_id}: {e}")
            import traceback
            traceback.print_exc()
            manager.disconnect(user_id)
    except Exception as e:
        print(f"❌ [DEBUG] Критическая ошибка в WebSocket endpoint: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass





# ============================================================
# МОНИТОРИНГ КОМНАТ
# ============================================================
@app.websocket("/ws/monitor/{room_id}")
async def monitor_websocket(websocket: WebSocket, room_id: str):
    await websocket.accept()
    print(f"🎙️ [MONITOR] New monitor connection for room: {room_id}")
    try:
        if room_id not in manager.active_rooms:
            await websocket.send_text(json.dumps({"type": "error", "message": f"Room {room_id} not found"}))
            await websocket.close()
            return
        await manager.add_room_monitor(room_id, websocket)
        room_info = manager.active_rooms.get(room_id, {})
        participants = room_info.get("participants", [])
        await websocket.send_text(json.dumps({
            "type": "monitor_started",
            "room_id": room_id,
            "message": "Мониторинг комнаты начат",
            "participants": participants,
            "participants_count": len(participants)
        }, default=str))
        
        participants_info = []
        for user_id in participants:
            session = manager.user_sessions.get(user_id, {})
            participants_info.append({
                "user_id": user_id,
                "status": session.get("status", "unknown"),
                "connected_at": session.get("connected_at"),
                "browser_id": session.get("browser_id", "unknown")
            })
        await websocket.send_text(json.dumps({
            "type": "room_status",
            "room_id": room_id,
            "participants": participants_info,
            "active": True,
            "timestamp": datetime.utcnow().isoformat()
        }, default=str))
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                if message["type"] == "stop_monitoring":
                    break
                elif message["type"] == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}))
                elif message["type"] == "get_room_info":
                    current_room_info = manager.active_rooms.get(room_id, {})
                    await websocket.send_text(json.dumps({
                        "type": "room_info",
                        "room_id": room_id,
                        "room_info": current_room_info,
                        "participants": manager.user_sessions.get(room_id, {}),
                        "timestamp": datetime.utcnow().isoformat()
                    }, default=str))
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                continue
    except WebSocketDisconnect:
        print(f"🎙️ [MONITOR] Monitor disconnected from room {room_id}")
    except Exception as e:
        print(f"❌ [MONITOR] Error: {e}")
    finally:
        await manager.remove_room_monitor(room_id, websocket)
        try:
            await websocket.close()
        except:
            pass

# ============================================================
# СТАТИЧЕСКИЕ ФАЙЛЫ И КОРНЕВЫЕ МАРШРУТЫ
# ============================================================
app.mount("/static", StaticFiles(directory="static"), name="static")
Path("static").mkdir(exist_ok=True)

@app.get("/")
async def root():
    return RedirectResponse(url="/admin")

@app.get("/admin")
async def admin_panel(request: Request):
    return HTMLResponse(ADMIN_HTML)

@app.get("/health")
async def health_check(request: Request):
    return {"status": "ok", "message": "Server is running", "timestamp": datetime.utcnow().isoformat()}

@app.get("/admin/chat/token")
async def get_chat_token_admin(moderator: dict = Depends(authenticate_admin)):
    import secrets
    token = secrets.token_urlsafe(32)
    if not hasattr(app.state, 'ws_tokens'):
        app.state.ws_tokens = {}
    app.state.ws_tokens[token] = {
        "username": moderator["username"],
        "expires": datetime.now() + timedelta(hours=1)
    }
    return {"token": token, "username": moderator["username"]}

@moderation_router.get("/chat/token")
async def get_chat_token_moderation(moderator: dict = Depends(authenticate_moderator)):
    import secrets
    token = secrets.token_urlsafe(32)
    if not hasattr(app.state, 'ws_tokens'):
        app.state.ws_tokens = {}
    app.state.ws_tokens[token] = {
        "username": moderator["username"],
        "expires": datetime.now() + timedelta(hours=1)
    }
    return {"token": token, "username": moderator["username"]}

@moderation_router.get("/check-auth")
async def moderation_check_auth(moderator: dict = Depends(authenticate_moderator)):
    return {
        "status": "success",
        "username": moderator["username"],
        "role": moderator.get("role", "moderator"),
        "permissions": moderator.get("permissions", [])
    }    

@app.get("/admin/chat/history")
async def get_chat_history_admin(
    moderator: dict = Depends(authenticate_admin),
    limit: int = Query(100, ge=1, le=500)
):
    try:
        messages = admin_chat_messages[-limit:] if admin_chat_messages else []
        return {
            "messages": messages,
            "total": len(admin_chat_messages)
        }
    except Exception as e:
        return {"messages": [], "total": 0, "error": str(e)}

# ============================================================
# HTML АДМИН-ПАНЕЛИ (полная версия из оригинального кода)
# ============================================================
ADMIN_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cosmic Voice - Админ панель</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        :root {
            --bg-primary: linear-gradient(135deg, #0a0a2a 0%, #1a1a3e 100%);
            --bg-card: rgba(0, 0, 0, 0.4);
            --bg-card-hover: rgba(0, 0, 0, 0.5);
            --bg-input: #1f2937;
            --border-color: rgba(0, 255, 255, 0.2);
            --border-hover: rgba(0, 255, 255, 0.4);
            --text-primary: #ffffff;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --glass-effect: rgba(0, 0, 0, 0.4);
            --table-header: rgba(0, 0, 0, 0.3);
            --table-hover: rgba(255, 255, 255, 0.03);
            --badge-online: rgba(34, 197, 94, 0.2);
            --badge-offline: rgba(107, 114, 128, 0.2);
            --badge-warning: rgba(245, 158, 11, 0.2);
            --chat-bg: rgba(0, 0, 0, 0.3);
            --chat-input: rgba(0, 0, 0, 0.4);
            --modal-overlay: rgba(0, 0, 0, 0.8);
            --tab-inactive: rgba(255, 255, 255, 0.05);
            --tab-hover: rgba(59, 130, 246, 0.2);
            --log-bg: rgba(0, 0, 0, 0.5);
        }

        body.light-theme {
            --bg-primary: linear-gradient(135deg, #e0e7ff 0%, #f0f4ff 100%);
            --bg-card: rgba(255, 255, 255, 0.85);
            --bg-card-hover: rgba(255, 255, 255, 0.95);
            --bg-input: #ffffff;
            --border-color: rgba(59, 130, 246, 0.3);
            --border-hover: rgba(59, 130, 246, 0.6);
            --text-primary: #1f2937;
            --text-secondary: #4b5563;
            --text-muted: #9ca3af;
            --glass-effect: rgba(255, 255, 255, 0.85);
            --table-header: rgba(59, 130, 246, 0.1);
            --table-hover: rgba(59, 130, 246, 0.05);
            --badge-online: rgba(34, 197, 94, 0.15);
            --badge-offline: rgba(107, 114, 128, 0.15);
            --badge-warning: rgba(245, 158, 11, 0.15);
            --chat-bg: rgba(255, 255, 255, 0.7);
            --chat-input: rgba(255, 255, 255, 0.9);
            --modal-overlay: rgba(0, 0, 0, 0.5);
            --tab-inactive: rgba(0, 0, 0, 0.05);
            --tab-hover: rgba(59, 130, 246, 0.1);
            --log-bg: rgba(0, 0, 0, 0.05);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            background: var(--bg-primary); 
            min-height: 100vh; 
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
            transition: background 0.3s ease, color 0.3s ease;
        }
        
        .glass-card { 
            background: var(--bg-card); 
            backdrop-filter: blur(12px); 
            border: 1px solid var(--border-color); 
            border-radius: 20px; 
            transition: all 0.3s ease; 
        }
        .glass-card:hover { 
            border-color: var(--border-hover); 
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
            background: var(--bg-card-hover);
        }
        
        .stat-card { cursor: pointer; transition: all 0.3s ease; }
        .stat-card:hover { transform: translateY(-5px); border-color: #3b82f6; }
        
        .tab-btn { 
            padding: 10px 20px; 
            border-radius: 12px; 
            transition: all 0.2s ease; 
            cursor: pointer; 
            font-size: 0.9rem; 
            background: var(--tab-inactive); 
            color: var(--text-secondary); 
        }
        .tab-btn:hover { background: var(--tab-hover); color: var(--text-primary); }
        .tab-btn.active { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; box-shadow: 0 4px 15px rgba(59,130,246,0.3); }
        .tab-content { display: none; animation: fadeIn 0.3s ease-out; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        
        .chat-container { background: var(--chat-bg); border-radius: 20px; display: flex; flex-direction: column; height: 550px; overflow: hidden; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
        .chat-message { display: flex; margin-bottom: 8px; animation: slideIn 0.3s ease; }
        @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
        .chat-message-avatar { width: 36px; height: 36px; border-radius: 50%; background: rgba(59,130,246,0.2); display: flex; align-items: center; justify-content: center; margin-right: 12px; flex-shrink: 0; font-size: 1.1rem; }
        .chat-message-content { max-width: 70%; }
        .chat-message-sender { font-size: 0.7rem; color: var(--text-secondary); margin-bottom: 4px; }
        .chat-message-text { background: rgba(59,130,246,0.15); padding: 8px 16px; border-radius: 18px; border-top-left-radius: 4px; font-size: 0.85rem; line-height: 1.4; color: var(--text-primary); word-wrap: break-word; }
        .chat-message-admin { justify-content: flex-end; }
        .chat-message-admin .chat-message-content { align-items: flex-end; }
        .chat-message-admin .chat-message-text { background: rgba(139,92,246,0.2); border-top-left-radius: 18px; border-top-right-radius: 4px; }
        
        .chat-input-area { padding: 12px 16px; border-top: 1px solid var(--border-color); display: flex; gap: 10px; background: var(--chat-input); }
        .chat-input { flex: 1; background: var(--bg-input); border: 1px solid var(--border-color); border-radius: 25px; padding: 10px 18px; color: var(--text-primary); outline: none; font-size: 0.9rem; }
        .chat-input:focus { border-color: #3b82f6; box-shadow: 0 0 8px rgba(59,130,246,0.3); }
        .chat-btn { background: none; border: none; cursor: pointer; padding: 8px; border-radius: 50%; transition: all 0.2s; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; }
        .chat-btn:hover { background: var(--tab-hover); transform: scale(1.05); }
        
        .data-table { width: 100%; border-collapse: collapse; }
        .data-table th { text-align: left; padding: 12px 16px; background: var(--table-header); color: var(--text-secondary); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; }
        .data-table td { padding: 12px 16px; border-bottom: 1px solid var(--border-color); color: var(--text-primary); }
        .data-table tr:hover { background: var(--table-hover); }
        
        .status-badge { display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 500; }
        .status-online { background: var(--badge-online); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
        .status-offline { background: var(--badge-offline); color: var(--text-secondary); border: 1px solid rgba(107,114,128,0.3); }
        .status-warning { background: var(--badge-warning); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
        
        .custom-scroll::-webkit-scrollbar { width: 6px; }
        .custom-scroll::-webkit-scrollbar-track { background: var(--border-color); border-radius: 10px; }
        .custom-scroll::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.5); border-radius: 10px; }
        
        .modal-overlay { background: var(--modal-overlay); backdrop-filter: blur(4px); }
        
        .recording-indicator { display: flex; align-items: center; justify-content: center; gap: 12px; padding: 8px; background: rgba(239,68,68,0.15); border-radius: 30px; margin-top: 8px; }
        .recording-dot { width: 6px; background: #ef4444; border-radius: 3px; animation: recordingPulse 1s ease-in-out infinite; }
        .recording-dot:nth-child(1) { animation-delay: 0s; height: 12px; }
        .recording-dot:nth-child(2) { animation-delay: 0.15s; height: 18px; }
        .recording-dot:nth-child(3) { animation-delay: 0.3s; height: 14px; }
        .recording-dot:nth-child(4) { animation-delay: 0.45s; height: 20px; }
        @keyframes recordingPulse { 0%,100% { height: 8px; } 50% { height: 20px; } }
        
        .preview-container { background: var(--bg-card); border-radius: 12px; padding: 10px 15px; margin-top: 10px; border-left: 3px solid #3b82f6; }
        .file-preview, .voice-preview { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
        .voice-preview audio { height: 40px; border-radius: 20px; }
        
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
        
        .log-container { background: var(--log-bg); border-radius: 12px; padding: 16px; font-family: monospace; font-size: 0.75rem; color: #10b981; }
        
        .theme-toggle {
            background: rgba(255,255,255,0.1);
            border: 1px solid var(--border-color);
            border-radius: 30px;
            padding: 6px 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
        }
        .theme-toggle:hover {
            background: var(--tab-hover);
            transform: scale(1.02);
        }
        .theme-icon { font-size: 1.1rem; }
        .theme-text { font-size: 0.85rem; }
        
        body.light-theme input, body.light-theme select, body.light-theme textarea {
            color: #1f2937;
            border-color: #d1d5db;
        }
        body.light-theme input::placeholder, body.light-theme textarea::placeholder {
            color: #9ca3af;
        }
        
        .btn-primary { background: linear-gradient(135deg, #3b82f6, #8b5cf6); }
        .btn-primary:hover { transform: scale(1.02); }
    </style>
</head>
<body>
    <!-- Login Modal -->
    <div id="loginModal" class="fixed inset-0 flex items-center justify-center z-50 modal-overlay">
        <div class="glass-card p-8 w-96 transform transition-all">
            <div class="text-center mb-6">
                <div class="text-5xl mb-3">🎙️</div>
                <h2 class="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Cosmic Voice Admin</h2>
                <p class="text-gray-400 text-sm mt-2">Войдите в систему управления</p>
            </div>
            <form id="loginForm" onsubmit="handleLogin(event); return false;">
                <input type="text" id="username" placeholder="Логин" required class="w-full bg-gray-800/50 border border-gray-700 rounded-xl px-4 py-3 text-white mb-3 focus:outline-none focus:border-blue-500">
                <input type="password" id="password" placeholder="Пароль" required class="w-full bg-gray-800/50 border border-gray-700 rounded-xl px-4 py-3 text-white mb-4 focus:outline-none focus:border-blue-500">
                <button type="submit" id="loginBtn" class="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 py-3 rounded-xl font-semibold transition">Войти</button>
                <div id="loginError" class="hidden mt-3 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300 text-sm text-center">Неверный логин или пароль</div>
            </form>
        </div>
    </div>

    <!-- Main Panel -->
    <div id="adminPanel" class="hidden">
        <!-- Header -->
        <header class="glass-card rounded-none border-t-0 border-x-0 sticky top-0 z-40">
            <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
                <div class="flex items-center gap-4">
                    <div class="text-2xl">🎙️</div>
                    <h1 class="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Cosmic Voice Admin</h1>
                    <div id="connectionStatus" class="status-badge status-offline"><span class="w-2 h-2 rounded-full bg-red-500 mr-2"></span>Оффлайн</div>
                    <div id="dbStatus" class="status-badge status-offline"><span class="w-2 h-2 rounded-full bg-red-500 mr-2"></span>MongoDB</div>
                </div>
                <div class="flex items-center gap-4">
                    <div class="text-right"><div class="text-xs text-gray-400">Администратор</div><div id="currentUser" class="font-semibold">admin</div></div>
                    <button id="themeToggle" class="theme-toggle" onclick="toggleTheme()">
                        <span class="theme-icon">🌙</span>
                        <span class="theme-text hidden sm:inline">Тёмная</span>
                    </button>
                    <button onclick="logout()" class="bg-red-500/20 hover:bg-red-500/30 px-4 py-2 rounded-xl transition">🚪 Выйти</button>
                </div>
            </div>
        </header>

        <div class="max-w-7xl mx-auto px-6 py-6">
            <!-- Stats Cards -->
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
                <div class="stat-card glass-card p-4 text-center border-l-4 border-blue-500"><div class="text-2xl font-bold" id="onlineUsers">0</div><div class="text-xs text-gray-400 mt-1">👥 Онлайн</div></div>
                <div class="stat-card glass-card p-4 text-center border-l-4 border-green-500"><div class="text-2xl font-bold" id="activeRooms">0</div><div class="text-xs text-gray-400 mt-1">🚪 Комнат</div></div>
                <div class="stat-card glass-card p-4 text-center border-l-4 border-yellow-500"><div class="text-2xl font-bold" id="searchingUsers">0</div><div class="text-xs text-gray-400 mt-1">🔍 В поиске</div></div>
                <div class="stat-card glass-card p-4 text-center border-l-4 border-purple-500"><div class="text-2xl font-bold" id="activeReports">0</div><div class="text-xs text-gray-400 mt-1">🚩 Жалоб</div></div>
                <div class="stat-card glass-card p-4 text-center border-l-4 border-red-500"><div class="text-2xl font-bold" id="bannedUsers">0</div><div class="text-xs text-gray-400 mt-1">🚫 Забанено</div></div>
                <div class="stat-card glass-card p-4 text-center border-l-4 border-orange-500"><div class="text-2xl font-bold" id="premiumCount">0</div><div class="text-xs text-gray-400 mt-1">👑 Премиум</div></div>
            </div>

            <!-- Tabs Navigation -->
            <div class="flex flex-wrap gap-2 mb-6 pb-2 overflow-x-auto custom-scroll">
                <button class="tab-btn active" data-tab="stats">📊 Стат</button>
                <button class="tab-btn" data-tab="connections">🔌 Кон</button>
                <button class="tab-btn" data-tab="moderation">⚖️ Модер</button>
                <button class="tab-btn" data-tab="reports">🚩 Жал</button>
                <button class="tab-btn" data-tab="messages">✉️ СМС</button>
                <button class="tab-btn" data-tab="video">🎥 Вид</button>
                <button class="tab-btn" data-tab="ads">📢 Рекл</button>
                <button class="tab-btn" data-tab="premium">👑 Прем</button>
                <button class="tab-btn" data-tab="payment">💳 Мани</button>
                <button class="tab-btn" data-tab="chat">💬 Чат</button>
                <button class="tab-btn" data-tab="system">⚙️ Сис</button>
                <button class="tab-btn" data-tab="frontend">🚀</button>
            </div>

            <!-- ==================== STATS TAB ==================== -->
            <div id="statsTab" class="tab-content active">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="glass-card p-6"><h3 class="font-semibold mb-4">📋 Очереди по категориям</h3><div id="queuesList" class="space-y-2"></div></div>
                    <div class="glass-card p-6"><h3 class="font-semibold mb-4">🚪 Активные комнаты</h3><div id="roomsList" class="space-y-2 max-h-96 overflow-y-auto custom-scroll"></div></div>
                </div>
                <div class="glass-card p-6 mt-6"><h3 class="font-semibold mb-4">📊 Статистика модерации</h3><div class="grid grid-cols-2 md:grid-cols-4 gap-4"><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-red-400" id="statBanned">0</div><div class="text-xs text-gray-400">Забанено</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-orange-400" id="statMuted">0</div><div class="text-xs text-gray-400">Заглушено</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-green-400" id="statActions">0</div><div class="text-xs text-gray-400">Действий сегодня</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-yellow-400" id="statWarnings">0</div><div class="text-xs text-gray-400">Предупреждений</div></div></div></div>
            </div>

            <!-- ==================== CONNECTIONS TAB ==================== -->
            <div id="connectionsTab" class="tab-content">
                <div class="glass-card overflow-hidden">
                    <div class="px-6 py-4 border-b border-gray-700/50 flex justify-between items-center"><h3 class="font-semibold">🔌 Активные подключения <span id="connCount" class="text-xs bg-blue-500/20 px-2 py-1 rounded-full ml-2">0</span></h3><button onclick="loadConnections()" class="bg-blue-500/20 hover:bg-blue-500/30 px-3 py-1 rounded-lg text-sm">🔄 Обновить</button></div>
                    <div class="overflow-x-auto"><table class="data-table"><thead><tr><th>Пользователь</th><th>Статус</th><th>IP адрес</th><th>Комната</th><th>Активность</th><th>Действия</th></tr></thead><tbody id="connTable"></tbody></table></div>
                </div>
            </div>

            <!-- ==================== MODERATION TAB ==================== -->
            <div id="moderationTab" class="tab-content">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="glass-card p-6"><h3 class="font-semibold mb-4">⚡ Ручная модерация</h3><div class="space-y-4"><input type="text" id="modUserId" placeholder="ID пользователя" class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2"><select id="modAction" class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2"><option value="warn">⚠️ Предупреждение</option><option value="mute">🔇 Заглушить</option><option value="kick">👢 Выгнать</option><option value="ban">🚫 Забанить</option><option value="unmute">🔊 Разглушить</option><option value="unban">🔓 Разбанить</option></select><div id="durationField" class="hidden"><div class="flex gap-3"><input type="number" id="modDuration" value="60" class="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2"><select id="durationUnit" class="bg-gray-800 border border-gray-700 rounded-xl px-4 py-2"><option value="minutes">Минут</option><option value="hours">Часов</option><option value="days">Дней</option></select></div></div><select id="modReason" class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2"><option value="spam">Спам</option><option value="harassment">Харассмент</option><option value="inappropriate_content">Неуместный контент</option><option value="voice_abuse">Злоупотребление голосом</option><option value="other">Другое</option></select><textarea id="modNotes" rows="2" placeholder="Примечания..." class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 resize-none"></textarea><button onclick="performModeration()" class="w-full bg-red-500 hover:bg-red-600 py-2 rounded-xl font-semibold transition">Применить</button></div></div>
                    <div class="glass-card p-6"><h3 class="font-semibold mb-4">🔍 Проверка статуса</h3><div class="flex gap-3 mb-4"><input type="text" id="checkUserId" placeholder="ID пользователя" class="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2"><button onclick="checkUserStatus()" class="bg-blue-500 px-6 rounded-xl hover:bg-blue-600 transition">🔍</button></div><div id="userStatusResult" class="hidden space-y-3"></div><div id="userStatusPlaceholder" class="text-center py-8 text-gray-400">Введите ID пользователя</div><div class="mt-6"><h4 class="font-medium mb-3">⚡ Быстрые действия</h4><div class="grid grid-cols-2 gap-2"><button onclick="quickModAction('warn')" class="bg-yellow-500/20 hover:bg-yellow-500/30 py-2 rounded-lg">⚠️ Предупредить</button><button onclick="quickModAction('mute')" class="bg-orange-500/20 hover:bg-orange-500/30 py-2 rounded-lg">🔇 Заглушить</button><button onclick="quickModAction('kick')" class="bg-red-500/20 hover:bg-red-500/30 py-2 rounded-lg">👢 Выгнать</button><button onclick="quickModAction('ban')" class="bg-red-500/20 hover:bg-red-500/30 py-2 rounded-lg">🚫 Забанить</button></div></div></div>
                </div>

                <!-- ===== НОВЫЙ БЛОК: РАСШИРЕННАЯ МОДЕРАЦИЯ ===== -->
    <div class="glass-card p-6 mt-6">
    <h3 class="font-semibold mb-4 flex justify-between items-center">
        <span>🔒 Расширенная модерация бана</span>
        <span class="text-xs text-gray-400">Полная блокировка пользователя</span>
    </h3>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <!-- Левая колонка - ID пользователя -->
        <div>
            <label class="text-sm text-gray-400 block mb-2">ID пользователя для бана</label>
            <div class="flex gap-2">
                <input type="text" id="advancedBanUserId" placeholder="Введите ID пользователя" class="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2">
                <button onclick="loadAdvancedBanUser()" class="bg-blue-500/20 hover:bg-blue-500/30 px-4 py-2 rounded-xl transition">🔍 Загрузить</button>
            </div>
            <div id="advancedBanUserInfo" class="mt-3 hidden">
                <div class="bg-gray-800/30 p-3 rounded-xl text-sm">
                    <div class="flex justify-between"><span class="text-gray-400">ID:</span><span id="advBanUserId" class="font-mono"></span></div>
                    <div class="flex justify-between"><span class="text-gray-400">Статус:</span><span id="advBanUserStatus"></span></div>
                    <div class="flex justify-between"><span class="text-gray-400">IP:</span><span id="advBanUserIp" class="font-mono text-xs"></span></div>
                    <div class="flex justify-between"><span class="text-gray-400">Browser ID:</span><span id="advBanUserBrowser" class="font-mono text-xs"></span></div>
                    <div class="flex justify-between"><span class="text-gray-400">Fingerprint:</span><span id="advBanUserFingerprint" class="font-mono text-xs"></span></div>
                </div>
            </div>
        </div>
        <!-- Правая колонка - Типы бана (ВСЕ ГАЛКИ СНЯТЫ ПО УМОЛЧАНИЮ) -->
        <div>
    <label class="text-sm text-gray-400 block mb-2">Тип блокировки</label>
    <div class="space-y-2">
        <div class="flex items-center gap-3 p-2 bg-gray-800/20 rounded-lg hover:bg-gray-800/40 transition cursor-pointer" onclick="toggleCheckbox('banTypeUser')">
            <input type="checkbox" id="banTypeUser" class="w-4 h-4 accent-red-500 pointer-events-none">
            <label class="flex-1 cursor-pointer">👤 Бан по ID пользователя</label>
        </div>
        
        <div class="flex items-center gap-3 p-2 bg-gray-800/20 rounded-lg hover:bg-gray-800/40 transition cursor-pointer" onclick="toggleCheckbox('banTypeIp')">
            <input type="checkbox" id="banTypeIp" class="w-4 h-4 accent-red-500 pointer-events-none">
            <label class="flex-1 cursor-pointer">🌐 Бан по IP адресу</label>
        </div>
        
        <div class="flex items-center gap-3 p-2 bg-gray-800/20 rounded-lg hover:bg-gray-800/40 transition cursor-pointer" onclick="toggleCheckbox('banTypeBrowser')">
            <input type="checkbox" id="banTypeBrowser" class="w-4 h-4 accent-red-500 pointer-events-none">
            <label class="flex-1 cursor-pointer">🖥️ Бан по Browser ID</label>
        </div>
        
        <div class="flex items-center gap-3 p-2 bg-gray-800/20 rounded-lg hover:bg-gray-800/40 transition cursor-pointer" onclick="toggleCheckbox('banTypeFingerprint')">
            <input type="checkbox" id="banTypeFingerprint" class="w-4 h-4 accent-red-500 pointer-events-none">
            <label class="flex-1 cursor-pointer">🆔 Бан по Fingerprint</label>
        </div>
        
        <div class="flex items-center gap-3 p-2 bg-gray-800/20 rounded-lg hover:bg-gray-800/40 transition cursor-pointer" onclick="toggleCheckbox('banTypeSubnet')">
            <input type="checkbox" id="banTypeSubnet" class="w-4 h-4 accent-red-500 pointer-events-none">
            <label class="flex-1 cursor-pointer">🔗 Бан по подсети (/24)</label>
        </div>
        
        <div class="flex items-center gap-3 p-2 bg-red-900/20 rounded-lg hover:bg-red-900/40 transition cursor-pointer" onclick="toggleCheckbox('banTypeAll')">
            <input type="checkbox" id="banTypeAll" class="w-4 h-4 accent-red-600 pointer-events-none">
            <label class="flex-1 cursor-pointer font-bold text-red-400">💀 МАКСИМАЛЬНЫЙ БАН (всё сразу)</label>
        </div>
    </div>
</div>
    </div>
    <!-- Длительность и причина -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
        <div>
            <label class="text-sm text-gray-400 block mb-2">Длительность</label>
            <div class="flex gap-2">
                <input type="number" id="advancedBanDuration" value="24" class="w-20 bg-gray-800 border border-gray-700 rounded-xl px-3 py-2">
                <select id="advancedBanUnit" class="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-3 py-2">
                    <option value="hours">Часов</option>
                    <option value="days">Дней</option>
                    <option value="weeks">Недель</option>
                    <option value="months">Месяцев</option>
                    <option value="permanent">Навсегда</option>
                </select>
            </div>
        </div>
        <div class="lg:col-span-2">
            <label class="text-sm text-gray-400 block mb-2">Причина бана</label>
            <div class="flex gap-2">
                <select id="advancedBanReason" class="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-3 py-2">
                    <option value="spam">Спам</option>
                    <option value="harassment">Харассмент</option>
                    <option value="inappropriate_content">Неуместный контент</option>
                    <option value="voice_abuse">Злоупотребление голосом</option>
                    <option value="ddos">DDOS атака</option>
                    <option value="multiple_accounts">Мультиаккаунт</option>
                    <option value="other">Другое</option>
                </select>
                <input type="text" id="advancedBanCustomReason" placeholder="Своя причина..." class="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-3 py-2">
            </div>
        </div>
    </div>
    <div class="flex gap-3 mt-4">
        <button onclick="executeAdvancedBan()" class="flex-1 bg-gradient-to-r from-red-600 to-red-800 hover:from-red-700 hover:to-red-900 py-3 rounded-xl font-bold transition text-white">🔒 Выполнить бан</button>
        <button onclick="clearAdvancedBan()" class="bg-gray-700 hover:bg-gray-600 px-6 py-3 rounded-xl transition">🗑️ Очистить</button>
    </div>
    <div id="advancedBanResult" class="mt-3 hidden"></div>
</div>
    <!-- ===== КОНЕЦ БЛОКА РАСШИРЕННОЙ МОДЕРАЦИИ ===== -->

                <div class="glass-card p-6 mt-6"><h3 class="font-semibold mb-4 flex justify-between"><span>🚫 Заблокированные пользователи</span><button onclick="loadBlockedUsers()" class="text-sm bg-blue-500/20 px-3 py-1 rounded">🔄 Обновить</button></h3><div id="blockedList" class="space-y-2"></div></div>
            </div>

            <!-- ==================== REPORTS TAB ==================== -->
            <div id="reportsTab" class="tab-content">
                <div class="glass-card p-6"><h3 class="font-semibold mb-4">🚩 Управление жалобами</h3><div class="flex gap-3 mb-4"><button onclick="loadReports('active')" class="px-4 py-2 rounded-lg bg-yellow-500/20 hover:bg-yellow-500/30">Активные</button><button onclick="loadReports('all')" class="px-4 py-2 rounded-lg bg-blue-500/20 hover:bg-blue-500/30">Все</button><button onclick="loadReports('resolved')" class="px-4 py-2 rounded-lg bg-green-500/20 hover:bg-green-500/30">Решённые</button></div><div class="overflow-x-auto"><table class="data-table"><thead><tr><th>Жалобщик</th><th>Нарушитель</th><th>Причина</th><th>Время</th><th>Статус</th><th>Действия</th></tr></thead><tbody id="reportsTable"></tbody></table></div></div>
            </div>

            <!-- ==================== MESSAGES TAB ==================== -->
            <div id="messagesTab" class="tab-content">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="glass-card p-6"><h3 class="font-semibold mb-4">✉️ Отправка сообщения</h3><select id="msgRecipient" class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 mb-3"><option value="all">Всем пользователям</option><option value="specific">Конкретному пользователю</option><option value="searching">В поиске</option><option value="connected">В разговоре</option></select><div id="specificUserDiv" class="hidden"><input type="text" id="specificUserId" placeholder="ID пользователя" class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 mb-3"></div><select id="msgType" class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 mb-3"><option value="system">🛠️ Системное</option><option value="normal">💬 Обычное</option><option value="notification">🔔 Оповещение</option></select><input type="text" id="msgTitle" placeholder="Заголовок" class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 mb-3"><textarea id="msgText" rows="4" placeholder="Текст сообщения..." class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 resize-none"></textarea><button onclick="sendMessage()" class="w-full bg-green-500 hover:bg-green-600 py-2 rounded-xl font-semibold transition">📤 Отправить</button></div>
                    <div class="glass-card p-6"><h3 class="font-semibold mb-4">📋 Последние сообщения</h3><div id="recentMessagesList" class="space-y-2 max-h-96 overflow-y-auto custom-scroll"></div></div>
                </div>
            </div>

            <!-- ==================== VIDEO TAB ==================== -->
            <div id="videoTab" class="tab-content">
                <div class="glass-card p-6 mb-6"><div class="flex justify-between items-center mb-4"><h3 class="font-semibold">🎥 Управление видео-режимом</h3><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="globalVideoToggle" class="sr-only peer" onchange="toggleGlobalVideo()"><div class="w-11 h-6 bg-gray-700 rounded-full peer peer-checked:bg-purple-600 after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-full"></div></label></div><div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6"><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-green-400" id="videoActiveUsers">0</div><div class="text-xs text-gray-400">Активных видео</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-blue-400" id="videoTotalOrders">0</div><div class="text-xs text-gray-400">Всего заказов</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-purple-400" id="videoRevenue">0 ₽</div><div class="text-xs text-gray-400">Выручка</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-yellow-400" id="activeTariffs">0</div><div class="text-xs text-gray-400">Тарифов</div></div></div><h4 class="font-semibold mb-3">📋 Тарифы видео-режима</h4><div id="tariffsList" class="space-y-2 mb-4"></div><button onclick="showTariffModal()" class="w-full bg-blue-500/20 hover:bg-blue-500/30 py-2 rounded-xl">➕ Добавить тариф</button></div>
            </div>

            <!-- ==================== ADS TAB ==================== -->
            <div id="adsTab" class="tab-content">
                <div class="glass-card p-6"><h3 class="font-semibold mb-4">📢 Управление рекламой</h3><div class="grid grid-cols-1 lg:grid-cols-2 gap-6"><div><h4 class="font-medium mb-3">🎵 Аудио реклама</h4><input type="file" id="audioFile" accept="audio/*" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-2"><input type="text" id="audioName" placeholder="Название" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-2"><button onclick="uploadAudioAd()" class="w-full bg-green-500 py-2 rounded-xl hover:bg-green-600 transition">⬆️ Загрузить</button><div id="audioAdsList" class="mt-4 space-y-2 max-h-60 overflow-y-auto custom-scroll"></div></div><div><h4 class="font-medium mb-3">📹 Видео реклама</h4><input type="file" id="videoFile" accept="video/*" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-2"><input type="text" id="videoName" placeholder="Название" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-2"><button onclick="uploadVideoAd()" class="w-full bg-green-500 py-2 rounded-xl hover:bg-green-600 transition">⬆️ Загрузить</button><div id="videoAdsList" class="mt-4 space-y-2 max-h-60 overflow-y-auto custom-scroll"></div></div></div><div class="mt-6 p-4 bg-gray-800/30 rounded-xl"><h4 class="font-medium mb-3">⚙️ Настройки рекламы</h4><div class="grid grid-cols-2 gap-4 mb-4"><div><label class="flex items-center gap-2"><input type="checkbox" id="audioEnabled"> Включить аудио рекламу</label></div><div><label class="flex items-center gap-2"><input type="checkbox" id="videoEnabled"> Включить видео рекламу</label></div></div><div class="grid grid-cols-2 gap-4"><div><label class="text-sm text-gray-400">Интервал аудио (сек):</label><input type="number" id="audioInterval" value="300" class="w-full bg-gray-800 border rounded-lg px-3 py-2 mt-1"></div><div><label class="text-sm text-gray-400">Интервал видео (сек):</label><input type="number" id="videoInterval" value="600" class="w-full bg-gray-800 border rounded-lg px-3 py-2 mt-1"></div></div><button onclick="saveAdSettings()" class="w-full mt-4 bg-blue-500 hover:bg-blue-600 py-2 rounded-xl transition">💾 Сохранить настройки</button></div></div>
            </div>

            <!-- ==================== PREMIUM TAB ==================== -->
            <div id="premiumTab" class="tab-content">
                <div class="glass-card p-6"><h3 class="font-semibold mb-4">👑 Премиум подписка</h3><div class="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6"><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-yellow-400" id="premiumActive">0</div><div class="text-xs text-gray-400">Активных</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-green-400" id="premiumRevenue">0 ₽</div><div class="text-xs text-gray-400">Выручка</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-blue-400" id="premiumPrice">0 ₽</div><div class="text-xs text-gray-400">Цена</div></div></div><div class="grid grid-cols-1 lg:grid-cols-2 gap-6"><div><h4 class="font-medium mb-3">✨ Активация премиум</h4><input type="text" id="premiumUserId" placeholder="ID пользователя" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><div class="flex gap-3 mb-3"><input type="number" id="premiumDays" value="30" class="w-24 bg-gray-800 border rounded-xl px-4 py-2"><select id="premiumUnit" class="flex-1 bg-gray-800 border rounded-xl px-4 py-2"><option value="days">Дней</option><option value="months">Месяцев</option></select></div><button onclick="activatePremium()" class="w-full bg-gradient-to-r from-yellow-600 to-amber-600 hover:from-yellow-700 hover:to-amber-700 py-2 rounded-xl font-semibold transition">👑 Активировать</button></div><div><h4 class="font-medium mb-3">⚙️ Настройки премиум</h4><input type="number" id="premiumSettingsPrice" placeholder="Цена (₽)" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><input type="number" id="premiumSettingsDuration" placeholder="Длительность (дней)" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><div class="flex items-center gap-3 mb-3"><input type="checkbox" id="premiumDisableAds"> <label>Отключить рекламу</label></div><div class="flex items-center gap-3 mb-3"><input type="checkbox" id="premiumPrioritySupport"> <label>Приоритетная поддержка</label></div><button onclick="savePremiumSettings()" class="w-full bg-blue-500 hover:bg-blue-600 py-2 rounded-xl transition">💾 Сохранить</button></div></div><h4 class="font-semibold mt-6 mb-3">👥 Премиум пользователи</h4><div id="premiumUsersList" class="space-y-2 max-h-96 overflow-y-auto custom-scroll"></div></div>
            </div>

            <!-- ==================== PAYMENT TAB ==================== -->
            <div id="paymentTab" class="tab-content">
                <div class="glass-card p-6"><h3 class="font-semibold mb-4">💳 Настройки оплаты Morune</h3><div class="grid grid-cols-1 lg:grid-cols-2 gap-6"><div><input type="text" id="paymentShopId" placeholder="Shop ID" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><input type="text" id="paymentApiKey" placeholder="API Key" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><input type="password" id="paymentSecretKey" placeholder="Secret Key" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><input type="text" id="paymentApiUrl" placeholder="API URL" value="https://api.morune.com/v1" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><label class="flex items-center gap-2 mb-4"><input type="checkbox" id="paymentTestMode"> Тестовый режим</label><button onclick="savePaymentSettings()" class="w-full bg-green-500 py-2 rounded-xl mb-3 hover:bg-green-600 transition">💾 Сохранить</button><button onclick="testPaymentConnection()" class="w-full bg-blue-500 py-2 rounded-xl hover:bg-blue-600 transition">🔌 Тест подключения</button></div><div><div class="bg-gray-800/30 p-4 rounded-xl mb-4"><h4 class="font-medium mb-2">Статус платежной системы</h4><div id="paymentStatusInfo" class="text-sm space-y-2"></div></div><div class="bg-gray-800/30 p-4 rounded-xl"><h4 class="font-medium mb-2">Статистика платежей</h4><div class="flex justify-between py-1"><span>Всего заказов:</span><span id="payTotalOrders" class="font-bold">0</span></div><div class="flex justify-between py-1"><span>Успешных:</span><span id="paySuccessOrders" class="font-bold text-green-400">0</span></div><div class="flex justify-between py-1"><span>Неудачных:</span><span id="payFailedOrders" class="font-bold text-red-400">0</span></div><div class="flex justify-between py-1"><span>Выручка:</span><span id="payRevenue" class="font-bold text-purple-400">0 ₽</span></div></div><button onclick="showTestPaymentModal()" class="w-full mt-4 bg-yellow-500/20 hover:bg-yellow-500/30 py-2 rounded-xl">🧪 Тестовый платеж</button></div></div></div>
            </div>

            <!-- ==================== CHAT TAB ==================== -->
            <div id="chatTab" class="tab-content">
                <div class="glass-card p-4">
                    <div class="chat-container">
                        <div class="px-4 py-3 border-b border-gray-700/50 flex justify-between items-center"><div class="flex items-center gap-2"><span>💬</span><span class="font-semibold">Чат с модератором</span><span id="chatStatus" class="text-xs px-2 py-1 rounded-full bg-yellow-500/20 text-yellow-400">● Подключение...</span></div><div class="text-xs text-gray-400">🔒 Зашифровано</div></div>
                        <div id="chatMessages" class="chat-messages custom-scroll"><div class="text-center py-8 text-gray-400">💬 Загрузка сообщений...</div></div>
                        <div class="chat-input-area"><div class="relative"><button id="emojiBtn" class="chat-btn" title="Эмодзи">😊</button><div id="emojiPicker" class="absolute bottom-12 left-0 bg-gray-800 rounded-xl p-2 w-72 border border-gray-600 shadow-2xl hidden z-50"><div class="emoji-list"></div></div></div><button id="fileBtn" class="chat-btn" title="Прикрепить файл">📎</button><input type="text" id="chatInput" class="chat-input" placeholder="Введите сообщение..." autocomplete="off" onkeypress="if(event.key==='Enter') sendChatMessage()"><button id="voiceBtn" class="chat-btn" title="Голосовое сообщение">🎤</button><button onclick="sendChatMessage()" class="chat-btn" title="Отправить">📤</button></div>
                        <div id="recordingIndicator" class="recording-indicator hidden"><div class="recording-dot"></div><div class="recording-dot"></div><div class="recording-dot"></div><div class="recording-dot"></div><span class="text-sm text-red-400 ml-2">🔴 Запись...</span></div>
                        <div id="previewContainer" class="preview-container hidden"></div>
                    </div>
                    <input type="file" id="fileInput" class="hidden" accept="image/*,audio/*,.pdf,.doc,.txt,.zip" onchange="handleFileSelect(this)">
                </div>
            </div>

            <!-- ==================== FRONTEND TAB ==================== -->
<div id="frontendTab" class="tab-content">
    <div class="glass-card p-6">
        <div class="flex justify-between items-center mb-6">
            <h3 class="font-semibold text-xl">🚀 Обновление фронтенда</h3>
            <button onclick="loadFrontendInfo()" class="bg-blue-500/20 hover:bg-blue-500/30 px-4 py-2 rounded-xl text-sm">
                🔄 Обновить
            </button>
        </div>

        <!-- Инфо о текущем build -->
        <div id="frontendInfo" class="bg-gray-800/30 p-4 rounded-xl mb-6">
            <div class="text-gray-400">Загрузка...</div>
        </div>

        <!-- Поле ввода версии -->
        <div class="mb-4">
            <label class="block text-sm text-gray-400 mb-2">
                📝 Версия нового обновления
            </label>
            <input 
                type="text" 
                id="newVersionInput" 
                placeholder="Например: v1.2.0, 2026.09.25, hotfix-auth"
                maxlength="50"
                class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500"
            >
            <div class="flex gap-2 mt-2 flex-wrap">
                <button onclick="setVersionPreset('1.0.0')" class="text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded-lg">1.0.0</button>
                <button onclick="setVersionPreset('v' + new Date().toISOString().slice(0,10).replace(/-/g,'.'))" class="text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded-lg">📅 Дата</button>
                <button onclick="setVersionPreset('hotfix-' + Date.now().toString(36).slice(-4))" class="text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded-lg">🔥 Hotfix</button>
                <button onclick="setVersionPreset('')" class="text-xs bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded-lg">🗑️ Очистить</button>
            </div>
            <div class="text-xs text-gray-500 mt-2">
                💡 Если оставить пустым — будет автоматическая версия (дата и время)
            </div>
        </div>

        <!-- Зона загрузки ZIP -->
        <div class="border-2 border-dashed border-gray-600 rounded-xl p-6 mb-4 hover:border-blue-500 transition" id="dropZone">
            <div class="text-center">
                <div class="text-6xl mb-3">📦</div>
                <h4 class="font-semibold text-lg mb-2">Загрузить ZIP</h4>
                <p class="text-sm text-gray-400 mb-4">
                    Перетащите ZIP-архив сюда или нажмите для выбора
                </p>
                <input type="file" id="frontendZipInput" accept=".zip" class="hidden">
                <button onclick="document.getElementById('frontendZipInput').click()"
                        class="bg-blue-500 hover:bg-blue-600 px-6 py-3 rounded-xl font-semibold transition">
                    📁 Выбрать ZIP
                </button>
            </div>
        </div>

        <!-- Прогресс загрузки -->
        <div id="uploadProgress" class="hidden mb-4">
            <div class="flex justify-between mb-2">
                <span class="text-sm text-gray-400">Загрузка...</span>
                <span id="uploadPercent" class="text-sm font-bold text-blue-400">0%</span>
            </div>
            <div class="h-3 bg-gray-700 rounded-full overflow-hidden">
                <div id="uploadBar" class="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all" style="width: 0%"></div>
            </div>
        </div>

        <!-- Результат -->
        <div id="uploadResult" class="hidden mb-4"></div>

        <!-- Кнопки -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <button onclick="loadFrontendInfo()"
                    class="bg-gray-700 hover:bg-gray-600 py-3 rounded-xl transition">
                🔍 Проверить версию
            </button>
            <button onclick="rollbackFrontend()"
                    class="bg-red-500/20 hover:bg-red-500/30 py-3 rounded-xl transition">
                ⏪ Откатить к бэкапу
            </button>
        </div>

        <!-- Инструкция -->
        <div class="mt-6 bg-gray-800/30 p-4 rounded-xl text-sm text-gray-400">
            <h4 class="font-semibold text-white mb-2">📖 Как обновить:</h4>
            <ol class="list-decimal list-inside space-y-1">
                <li>Соберите фронт: <code class="bg-gray-700 px-2 py-1 rounded">npm run build</code></li>
                <li>Заархивируйте <b>содержимое</b> папки <code class="bg-gray-700 px-2 py-1 rounded">build/</code> в ZIP</li>
                <li>Введите версию (или оставьте пусто для авто)</li>
                <li>Загрузите ZIP</li>
                <li>Перезапустите раздачу фронта (если используется <code class="bg-gray-700 px-2 py-1 rounded">serve</code>)</li>
            </ol>
        </div>
    </div>
</div>

            <!-- ==================== SYSTEM TAB ==================== -->
            <div id="systemTab" class="tab-content">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="glass-card p-6"><h3 class="font-semibold mb-4">⚙️ Настройки системы</h3><label class="flex items-center gap-3 mb-3"><input type="checkbox" id="sysMaintenance"> Режим техобслуживания</label><label class="flex items-center gap-3 mb-3"><input type="checkbox" id="sysRegistration" checked> Разрешить регистрацию</label><input type="number" id="sysMaxUsers" placeholder="Макс. пользователей" value="1000" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><textarea id="sysMessage" rows="3" placeholder="Системное сообщение" class="w-full bg-gray-800 border rounded-xl px-4 py-2 resize-none">Добро пожаловать в голосовой чат!</textarea><button onclick="saveSystemSettings()" class="w-full bg-blue-500 hover:bg-blue-600 py-2 rounded-xl transition mt-3">💾 Сохранить</button></div>
                    <div class="glass-card p-6"><h3 class="font-semibold mb-4">🛠️ Экстренные действия</h3><button onclick="emergencyStop()" class="w-full bg-red-500 hover:bg-red-600 py-2 rounded-xl mb-3 transition">⏻ Экстренная остановка</button><button onclick="clearQueues()" class="w-full bg-yellow-500 hover:bg-yellow-600 py-2 rounded-xl mb-3 transition">🗑️ Очистить очереди</button><button onclick="forceSync()" class="w-full bg-green-500 hover:bg-green-600 py-2 rounded-xl mb-3 transition">🔄 Синхронизация с БД</button></div>
                </div>
                <div class="glass-card p-6 mt-6"><h3 class="font-semibold mb-4 flex justify-between items-center"><span>📋 Логи сервера</span><button onclick="refreshLogs()" class="bg-blue-500/20 px-3 py-1 rounded-lg text-sm">🔄 Обновить</button></h3><div id="serverLogs" class="log-container h-48 overflow-y-auto custom-scroll"></div></div>
                <!-- ========== УПРАВЛЕНИЕ ПЕРСОНАЛОМ ========== -->
<div class="glass-card p-6 mt-6">
    <h3 class="font-semibold mb-4 flex justify-between items-center">
        <span>👥 Управление персоналом</span>
        <button onclick="loadStaffList()" class="bg-blue-500/20 px-3 py-1 rounded-lg text-sm">🔄 Обновить</button>
    </h3>

    <!-- Форма создания -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-3 mb-4 p-4 bg-gray-800/30 rounded-xl">
        <div class="lg:col-span-3 text-sm font-bold text-blue-400 mb-1">➕ Создать нового сотрудника</div>
        <input type="text" id="staffNewUsername" placeholder="Логин" class="bg-gray-800 border border-gray-700 rounded-xl px-4 py-2">
        <input type="text" id="staffNewPassword" placeholder="Пароль" class="bg-gray-800 border border-gray-700 rounded-xl px-4 py-2">
        <select id="staffNewRole" class="bg-gray-800 border border-gray-700 rounded-xl px-4 py-2" onchange="onStaffRoleChange()">
            <option value="moderator">Модератор</option>
            <option value="admin">Администратор</option>
        </select>

        <div class="lg:col-span-3">
            <div class="text-xs text-gray-400 mb-2">Права доступа:</div>
            <div id="staffNewPermissions" class="grid grid-cols-2 md:grid-cols-3 gap-2"></div>
        </div>

        <button onclick="createStaff()" class="lg:col-span-3 bg-green-500 hover:bg-green-600 py-2 rounded-xl font-semibold transition">
            ✅ Создать
        </button>
    </div>

    <!-- Список персонала -->
    <div class="text-sm font-bold text-blue-400 mb-2">📋 Существующие сотрудники</div>
    <div id="staffList" class="space-y-2 max-h-96 overflow-y-auto custom-scroll">
        <div class="text-center py-4 text-gray-400">Загрузка...</div>
    </div>
</div>

<!-- ========== СМЕНА ПАРОЛЯ ========== -->
<div class="glass-card p-6 mt-6">
    <h3 class="font-semibold mb-4">🔑 Смена пароля администратора/модератора</h3>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
        <input type="text" id="pwdChangeUsername" placeholder="Логин" class="bg-gray-800 border border-gray-700 rounded-xl px-4 py-2">
        <input type="password" id="pwdChangeNew" placeholder="Новый пароль" class="bg-gray-800 border border-gray-700 rounded-xl px-4 py-2">
        <button onclick="changeStaffPassword()" class="bg-yellow-500 hover:bg-yellow-600 py-2 rounded-xl font-semibold transition">
            🔑 Сменить пароль
        </button>
    </div>
    <div class="text-xs text-gray-500 mt-2">
        💡 Используйте список выше — там у каждого сотрудника есть кнопка быстрой смены пароля.
    </div>
</div>
                <div class="glass-card p-6 mt-6"><h3 class="font-semibold mb-4">💾 База данных</h3><div class="grid grid-cols-3 gap-4 mb-4"><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-blue-400" id="dbUsers">0</div><div class="text-xs text-gray-400">Пользователей</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-green-400" id="dbConns">0</div><div class="text-xs text-gray-400">Подключений</div></div><div class="bg-gray-800/30 p-3 rounded-xl text-center"><div class="text-xl font-bold text-purple-400" id="dbRooms">0</div><div class="text-xs text-gray-400">Комнат</div></div></div><div class="flex gap-3"><button onclick="forceSync()" class="flex-1 bg-blue-500 py-2 rounded-xl">🔄 Синхронизировать</button><button onclick="showSyncStats()" class="flex-1 bg-green-500/20 py-2 rounded-xl">📊 Статистика синхронизации</button></div></div>
            </div>
        </div>
    </div>

    <!-- Modals -->
    <div id="tariffModal" class="fixed inset-0 flex items-center justify-center z-50 hidden modal-overlay"><div class="glass-card p-6 w-96"><h3 id="tariffModalTitle" class="font-semibold mb-4">Новый тариф</h3><input type="text" id="tariffName" placeholder="Название" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><input type="number" id="tariffMinutes" placeholder="Минуты (0 = бессрочно)" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><input type="number" id="tariffPrice" placeholder="Цена" step="0.01" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><label class="flex items-center gap-2 mb-4"><input type="checkbox" id="tariffActive" checked> Активный</label><div class="flex gap-3"><button onclick="closeTariffModal()" class="flex-1 bg-gray-700 py-2 rounded-xl">Отмена</button><button onclick="saveTariff()" class="flex-1 bg-green-500 py-2 rounded-xl">Сохранить</button></div></div></div>
    <div id="testPaymentModal" class="fixed inset-0 flex items-center justify-center z-50 hidden modal-overlay"><div class="glass-card p-6 w-96"><h3 class="font-semibold mb-4">🧪 Тестовый платеж</h3><input type="text" id="testUserId" placeholder="ID пользователя" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><select id="testTariff" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><option value="15">15 минут - 100 ₽</option><option value="30">30 минут - 180 ₽</option><option value="60">60 минут - 300 ₽</option><option value="0">Навсегда - 1000 ₽</option></select><input type="number" id="testPrice" placeholder="Цена" step="0.01" value="100" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-4"><div class="flex gap-3"><button onclick="closeTestPaymentModal()" class="flex-1 bg-gray-700 py-2 rounded-xl">Отмена</button><button onclick="processTestPayment()" class="flex-1 bg-yellow-500 py-2 rounded-xl">Создать</button></div></div></div>
    <div id="resolveReportModal" class="fixed inset-0 flex items-center justify-center z-50 hidden modal-overlay"><div class="glass-card p-6 w-96"><h3 class="font-semibold mb-4">Решение жалобы</h3><select id="resolveAction" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><option value="no_action">Без действия</option><option value="warn">Предупредить</option><option value="mute">Заглушить</option><option value="ban">Забанить</option></select><textarea id="resolveNotes" rows="3" placeholder="Примечания..." class="w-full bg-gray-800 border rounded-xl px-4 py-2 resize-none mb-4"></textarea><div class="flex gap-3"><button onclick="closeResolveModal()" class="flex-1 bg-gray-700 py-2 rounded-xl">Отмена</button><button onclick="confirmResolveReport()" class="flex-1 bg-green-500 py-2 rounded-xl">Решить</button></div></div></div>
    <!-- ===== МОДАЛКА НАСТРОЙКИ ПЕРИОДА ПОКАЗА ===== -->
<div id="adScheduleModal" class="fixed inset-0 flex items-center justify-center z-50 hidden modal-overlay">
    <div class="glass-card p-6 w-96 max-w-[95vw]">
        <div class="flex justify-between items-center mb-4">
            <h3 id="adScheduleTitle" class="font-semibold text-lg">⚙️ Период показа</h3>
            <button onclick="closeAdScheduleModal()" class="text-gray-400 hover:text-white text-2xl">×</button>
        </div>

        <input type="hidden" id="adScheduleId" value="">
        <input type="hidden" id="adScheduleType" value="">

        <!-- Пресеты -->
        <label class="text-xs text-gray-400 block mb-2">Быстрый выбор</label>
        <div class="grid grid-cols-3 gap-2 mb-4">
            <button type="button" onclick="setAdSchedulePreset('day')" class="preset-btn bg-gray-800 hover:bg-blue-500/30 border border-gray-700 rounded-lg py-2 text-xs transition" data-preset="day">📅 1 день</button>
            <button type="button" onclick="setAdSchedulePreset('week')" class="preset-btn bg-gray-800 hover:bg-blue-500/30 border border-gray-700 rounded-lg py-2 text-xs transition" data-preset="week">📅 Неделя</button>
            <button type="button" onclick="setAdSchedulePreset('month')" class="preset-btn bg-gray-800 hover:bg-blue-500/30 border border-gray-700 rounded-lg py-2 text-xs transition" data-preset="month">📅 Месяц</button>
            <button type="button" onclick="setAdSchedulePreset('year')" class="preset-btn bg-gray-800 hover:bg-blue-500/30 border border-gray-700 rounded-lg py-2 text-xs transition" data-preset="year">📅 Год</button>
            <button type="button" onclick="setAdSchedulePreset('always')" class="preset-btn bg-gray-800 hover:bg-blue-500/30 border border-gray-700 rounded-lg py-2 text-xs transition" data-preset="always">♾️ Всегда</button>
            <button type="button" onclick="setAdSchedulePreset('custom')" class="preset-btn bg-gray-800 hover:bg-blue-500/30 border border-gray-700 rounded-lg py-2 text-xs transition" data-preset="custom">✏️ Свой</button>
        </div>

        <!-- Период -->
        <div class="grid grid-cols-2 gap-3 mb-4">
            <div>
                <label class="text-xs text-gray-400 block mb-1">С какого</label>
                <input type="datetime-local" id="adScheduleFrom" class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm">
            </div>
            <div>
                <label class="text-xs text-gray-400 block mb-1">По какое</label>
                <input type="datetime-local" id="adScheduleUntil" class="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm">
            </div>
        </div>

        <!-- Активность -->
        <label class="flex items-center gap-2 mb-4 cursor-pointer">
            <input type="checkbox" id="adScheduleActive" checked class="w-4 h-4">
            <span class="text-sm">Реклама активна</span>
        </label>

        <!-- Кнопки -->
        <div class="flex gap-3">
            <button onclick="closeAdScheduleModal()" class="flex-1 bg-gray-700 hover:bg-gray-600 py-2 rounded-xl transition">Отмена</button>
            <button onclick="saveAdSchedule()" class="flex-1 bg-green-500 hover:bg-green-600 py-2 rounded-xl transition font-semibold">💾 Сохранить</button>
        </div>
    </div>
</div>
    <div id="quickActionModal" class="fixed inset-0 flex items-center justify-center z-50 hidden modal-overlay"><div class="glass-card p-6 w-96"><h3 id="quickActionTitle" class="font-semibold mb-4">Быстрое действие</h3><input type="text" id="quickUserId" placeholder="ID пользователя" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><select id="quickReason" class="w-full bg-gray-800 border rounded-xl px-4 py-2 mb-3"><option value="spam">Спам</option><option value="harassment">Харассмент</option><option value="inappropriate_content">Неуместный контент</option><option value="other">Другое</option></select><div id="quickDurationDiv" class="hidden mb-3"><input type="number" id="quickDuration" placeholder="Длительность" value="60" class="w-full bg-gray-800 border rounded-xl px-4 py-2"></div><div class="flex gap-3"><button onclick="closeQuickModal()" class="flex-1 bg-gray-700 py-2 rounded-xl">Отмена</button><button onclick="confirmQuickAction()" class="flex-1 bg-red-500 py-2 rounded-xl">Применить</button></div></div></div>
    
    <script>
        const BACKEND_URL = window.location.origin;
        let authToken = '';
        let adminChatWs = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let recordedBlob = null;
        let attachedFile = null;
        let currentEditTariffId = null;
        let currentReportFilter = 'active';
        let currentResolveReportId = null;
        let currentQuickAction = null;
        let unreadCount = 0;

        function toggleTheme() {
            const body = document.body;
            const themeBtn = document.getElementById('themeToggle');
            const isLight = body.classList.contains('light-theme');
            
            if (isLight) {
                body.classList.remove('light-theme');
                localStorage.setItem('adminTheme', 'dark');
                themeBtn.innerHTML = '<span class="theme-icon">🌙</span><span class="theme-text hidden sm:inline">Тёмная</span>';
            } else {
                body.classList.add('light-theme');
                localStorage.setItem('adminTheme', 'light');
                themeBtn.innerHTML = '<span class="theme-icon">☀️</span><span class="theme-text hidden sm:inline">Светлая</span>';
            }
        }

        function loadTheme() {
            const savedTheme = localStorage.getItem('adminTheme');
            const themeBtn = document.getElementById('themeToggle');
            if (savedTheme === 'light') {
                document.body.classList.add('light-theme');
                if (themeBtn) themeBtn.innerHTML = '<span class="theme-icon">☀️</span><span class="theme-text hidden sm:inline">Светлая</span>';
            } else {
                document.body.classList.remove('light-theme');
                if (themeBtn) themeBtn.innerHTML = '<span class="theme-icon">🌙</span><span class="theme-text hidden sm:inline">Тёмная</span>';
            }
        }

        function getAuthHeaders() { return { 'Authorization': `Basic ${authToken}`, 'Content-Type': 'application/json' }; }

        async function handleLogin(e) {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            authToken = btoa(`${username}:${password}`);
            try {
                const res = await fetch(`${BACKEND_URL}/admin/health`, { headers: getAuthHeaders() });
                if (res.ok) {
                    document.getElementById('loginModal').classList.add('hidden');
                    document.getElementById('adminPanel').classList.remove('hidden');
                    document.getElementById('currentUser').textContent = username;
                    localStorage.setItem('adminAuth', authToken);
                    loadTheme();
                    initAdmin();
                } else {
                    document.getElementById('loginError').classList.remove('hidden');
                }
            } catch(e) { document.getElementById('loginError').classList.remove('hidden'); }
        }

        function logout() { localStorage.removeItem('adminAuth'); location.reload(); }

                // ============================================================
        // ФРОНТЕНД
        // ============================================================
        async function loadFrontendInfo() {
            const infoDiv = document.getElementById('frontendInfo');
            if (!infoDiv) return;

            infoDiv.innerHTML = '<div class="text-gray-400">Загрузка...</div>';

            try {
                const res = await fetch(`${BACKEND_URL}/admin/frontend-info`, {
                    headers: getAuthHeaders()
                });

                if (res.ok) {
                    const d = await res.json();
                    if (d.installed) {
                        const modDate = d.last_modified
                            ? new Date(d.last_modified * 1000).toLocaleString('ru-RU')
                            : 'неизвестно';

                        // Форматируем uploaded_at
                        let uploadedText = '';
                        if (d.uploaded_at) {
                            try {
                                const uploadDate = new Date(d.uploaded_at);
                                uploadedText = uploadDate.toLocaleString('ru-RU');
                            } catch (e) {}
                        }

                        infoDiv.innerHTML = `
                            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <div class="bg-green-500/20 p-3 rounded-lg">
                                    <div class="text-xs text-gray-400">Статус</div>
                                    <div class="font-bold text-green-400">✅ Установлен</div>
                                </div>
                                <div class="bg-blue-500/20 p-3 rounded-lg">
                                    <div class="text-xs text-gray-400">Версия</div>
                                    <div class="font-bold text-blue-400 text-lg">${d.version || '—'}</div>
                                    <div class="text-xs text-gray-500">${d.version_source || ''}</div>
                                </div>
                                <div class="bg-purple-500/20 p-3 rounded-lg">
                                    <div class="text-xs text-gray-400">Файлов</div>
                                    <div class="font-bold text-purple-400">${d.files_count || 0}</div>
                                </div>
                                <div class="bg-yellow-500/20 p-3 rounded-lg">
                                    <div class="text-xs text-gray-400">Изменён</div>
                                    <div class="font-bold text-yellow-400 text-xs">${modDate}</div>
                                </div>
                            </div>
                            ${d.uploaded_by || uploadedText ? `
                                <div class="mt-3 flex gap-4 text-xs text-gray-400">
                                    ${d.uploaded_by ? `<div>👤 Загрузил: <b class="text-white">${d.uploaded_by}</b></div>` : ''}
                                    ${uploadedText ? `<div>🕒 Загружено: <b class="text-white">${uploadedText}</b></div>` : ''}
                                </div>
                            ` : ''}
                            <div class="mt-3 text-xs text-gray-500">
                                📁 ${d.path}
                            </div>
                            ${d.has_backup ? '<div class="mt-2 text-xs text-green-400">💾 Есть бэкап для отката</div>' : ''}
                        `;
                    } else {
                        infoDiv.innerHTML = `
                            <div class="text-center py-6">
                                <div class="text-5xl mb-2">📭</div>
                                <div class="text-yellow-400 font-semibold">Папка build не найдена</div>
                                <div class="text-xs text-gray-400 mt-2">Соберите фронт: npm run build</div>
                                <div class="text-xs text-gray-500 mt-1">${d.path || ''}</div>
                            </div>
                        `;
                    }
                } else {
                    infoDiv.innerHTML = '<div class="text-red-400">Ошибка загрузки информации</div>';
                }
            } catch (e) {
                infoDiv.innerHTML = `<div class="text-red-400">Ошибка: ${e.message}</div>`;
            }
        }

                function setVersionPreset(value) {
            const input = document.getElementById('newVersionInput');
            if (input) input.value = value;
        }

        function uploadFrontendZip(file) {
            if (!file) return;

            if (!file.name.endsWith('.zip')) {
                alert('❌ Выберите ZIP-архив');
                return;
            }

            if (file.size > 200 * 1024 * 1024) {
                alert('❌ Файл слишком большой (макс 200 МБ)');
                return;
            }

            // 🆕 Читаем введённую версию
            const versionInput = document.getElementById('newVersionInput');
            const manualVersion = versionInput ? versionInput.value.trim() : '';

            const sizeMB = (file.size / 1024 / 1024).toFixed(2);
            const versionText = manualVersion || '(автоматически)';

            if (!confirm(
                `📦 Загрузить обновление?\n\n` +
                `Файл: ${file.name}\n` +
                `Размер: ${sizeMB} МБ\n` +
                `Версия: ${versionText}\n\n` +
                `Текущий build будет сохранён в бэкап.`
            )) {
                return;
            }

            const progressDiv = document.getElementById('uploadProgress');
            const progressBar = document.getElementById('uploadBar');
            const progressPercent = document.getElementById('uploadPercent');
            const resultDiv = document.getElementById('uploadResult');

            progressDiv.classList.remove('hidden');
            resultDiv.classList.add('hidden');
            progressBar.style.width = '0%';
            progressPercent.textContent = '0%';

            const xhr = new XMLHttpRequest();
            const formData = new FormData();
            formData.append('file', file);
            // 🆕 Передаём версию как form-поле
            formData.append('version', manualVersion);

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = percent + '%';
                    progressPercent.textContent = percent + '%';
                }
            });

            xhr.addEventListener('load', () => {
                progressDiv.classList.add('hidden');
                resultDiv.classList.remove('hidden');

                if (xhr.status === 200) {
                    try {
                        const d = JSON.parse(xhr.responseText);
                        resultDiv.innerHTML = `
                            <div class="bg-green-500/20 border border-green-500/50 p-4 rounded-xl">
                                <div class="flex items-start gap-3">
                                    <span class="text-3xl">✅</span>
                                    <div>
                                        <div class="font-bold text-green-400">Обновление установлено!</div>
                                        <div class="text-sm text-gray-300 mt-1">Версия: <b class="text-white">${d.version || 'N/A'}</b></div>
                                        <div class="text-xs text-gray-400">Загрузил: ${d.uploaded_by || 'unknown'}</div>
                                        <div class="text-xs text-gray-400">Файлов: ${d.files_count || 0}</div>
                                        <div class="text-xs text-yellow-400 mt-2">
                                            ⚠️ Перезапустите раздачу: <code>npx serve -s build -l 3000</code>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                        // Очищаем поле версии
                        if (versionInput) versionInput.value = '';
                        loadFrontendInfo();
                    } catch (e) {
                        resultDiv.innerHTML = `<div class="bg-green-500/20 p-4 rounded-xl">✅ Обновление установлено</div>`;
                        loadFrontendInfo();
                    }
                } else {
                    let errMsg = 'Неизвестная ошибка';
                    try {
                        const d = JSON.parse(xhr.responseText);
                        errMsg = d.detail || d.message || errMsg;
                    } catch (e) {
                        errMsg = xhr.responseText || errMsg;
                    }
                    resultDiv.innerHTML = `
                        <div class="bg-red-500/20 border border-red-500/50 p-4 rounded-xl">
                            <div class="flex items-start gap-3">
                                <span class="text-3xl">❌</span>
                                <div>
                                    <div class="font-bold text-red-400">Ошибка загрузки</div>
                                    <div class="text-sm text-gray-300 mt-1">${errMsg}</div>
                                </div>
                            </div>
                        </div>
                    `;
                }
            });

            xhr.addEventListener('error', () => {
                progressDiv.classList.add('hidden');
                resultDiv.classList.remove('hidden');
                resultDiv.innerHTML = `
                    <div class="bg-red-500/20 p-4 rounded-xl">
                        ❌ Ошибка сети: не удалось загрузить файл
                    </div>
                `;
            });

            xhr.open('POST', `${BACKEND_URL}/admin/upload-frontend`);
            xhr.setRequestHeader('Authorization', `Basic ${authToken}`);
            xhr.send(formData);
        }

        async function rollbackFrontend() {
            if (!confirm('⏪ Откатить к предыдущей версии фронтенда?')) return;

            const resultDiv = document.getElementById('uploadResult');
            resultDiv.classList.remove('hidden');
            resultDiv.innerHTML = '<div class="bg-yellow-500/20 p-4 rounded-xl">⏳ Откат...</div>';

            try {
                const res = await fetch(`${BACKEND_URL}/admin/frontend-rollback`, {
                    method: 'POST',
                    headers: getAuthHeaders()
                });

                if (res.ok) {
                    resultDiv.innerHTML = `
                        <div class="bg-green-500/20 p-4 rounded-xl">
                            ✅ Откат выполнен
                        </div>
                    `;
                    loadFrontendInfo();
                } else {
                    const d = await res.json().catch(() => ({}));
                    resultDiv.innerHTML = `<div class="bg-red-500/20 p-4 rounded-xl">❌ Ошибка: ${d.detail || 'Не удалось откатить'}</div>`;
                }
            } catch (e) {
                resultDiv.innerHTML = `<div class="bg-red-500/20 p-4 rounded-xl">❌ Ошибка сети: ${e.message}</div>`;
            }
        }

        // Drag-and-drop
        document.addEventListener('DOMContentLoaded', () => {
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('frontendZipInput');
            if (!dropZone || !fileInput) return;

            dropZone.addEventListener('click', (e) => {
                if (e.target.tagName !== 'BUTTON') fileInput.click();
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files[0]) {
                    uploadFrontendZip(e.target.files[0]);
                    e.target.value = '';
                }
            });

            ['dragenter', 'dragover'].forEach(evt => {
                dropZone.addEventListener(evt, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.add('border-blue-500', 'bg-blue-500/10');
                });
            });

            ['dragleave', 'drop'].forEach(evt => {
                dropZone.addEventListener(evt, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropZone.classList.remove('border-blue-500', 'bg-blue-500/10');
                });
            });

            dropZone.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if (files.length > 0) uploadFrontendZip(files[0]);
            });
        });

        function checkAuth() {
            const saved = localStorage.getItem('adminAuth');
            if (saved) { authToken = saved; fetch(`${BACKEND_URL}/admin/health`, { headers: getAuthHeaders() }).then(r => { if(r.ok){ document.getElementById('loginModal').classList.add('hidden'); document.getElementById('adminPanel').classList.remove('hidden'); loadTheme(); initAdmin(); } }); }
        }

        async function initAdmin() {
    setupTabs();
    connectChatWebSocket();
    
    // ===== HEARTBEAT =====
    setInterval(function() {
        if (adminChatWs && adminChatWs.readyState === WebSocket.OPEN) {
            try {
                adminChatWs.send(JSON.stringify({ type: 'ping' }));
            } catch(e) {}
        }
    }, 25000);
    
    loadStats();
    loadConnections();
    loadReports();
    loadBlockedUsers();
    loadMessages();
    loadVideoData();
    loadTariffs();
    loadAudioAds();
    loadVideoAds();
    loadAdSettings();
    loadPremiumStats();
    loadPremiumUsers();
    loadPremiumSettings();
    loadPaymentSettings();
    loadSystemSettings();
    refreshLogs();
    showSyncStats();
    loadFrontendInfo();
    loadStaffList();
    
    setInterval(function() { loadStats(); }, 10000);
    setInterval(function() {
        if (document.getElementById('connectionsTab').classList.contains('active')) {
            loadConnections();
        }
    }, 5000);
}

        function setupTabs() {
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                    document.getElementById(`${btn.dataset.tab}Tab`).classList.add('active');
                    if (btn.dataset.tab === 'chat' && unreadCount > 0) { unreadCount = 0; }
                    if (btn.dataset.tab === 'connections') loadConnections();
                    if (btn.dataset.tab === 'reports') loadReports(currentReportFilter);
                    if (btn.dataset.tab === 'moderation') loadBlockedUsers();
                    if (btn.dataset.tab === 'messages') loadMessages();
                    if (btn.dataset.tab === 'video') { loadVideoData(); loadTariffs(); }
                    if (btn.dataset.tab === 'ads') { loadAudioAds(); loadVideoAds(); loadAdSettings(); }
                    if (btn.dataset.tab === 'premium') { loadPremiumStats(); loadPremiumUsers(); }
                    if (btn.dataset.tab === 'frontend') { loadFrontendInfo(); }
                    if (btn.dataset.tab === 'system') { loadStaffList(); }
                });
            });
            document.getElementById('msgRecipient').addEventListener('change', function() {
                document.getElementById('specificUserDiv').classList.toggle('hidden', this.value !== 'specific');
            });
            document.getElementById('modAction').addEventListener('change', function() {
                document.getElementById('durationField').classList.toggle('hidden', !['mute','ban'].includes(this.value));
            });
        }

        async function loadStats() {
            try {
                const res = await fetch(`${BACKEND_URL}/admin/stats`, { headers: getAuthHeaders() });
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('onlineUsers').textContent = data.online_users || 0;
                    document.getElementById('activeRooms').textContent = data.active_rooms || 0;
                    document.getElementById('searchingUsers').textContent = data.searching_users || 0;
                    document.getElementById('activeReports').textContent = data.active_reports || 0;
                    document.getElementById('bannedUsers').textContent = data.banned_users || 0;
                    document.getElementById('premiumCount').textContent = data.premium_users || 0;
                    document.getElementById('statBanned').textContent = data.banned_users || 0;
                    document.getElementById('statMuted').textContent = data.muted_users || 0;
                    document.getElementById('statActions').textContent = data.moderation_actions_today || 0;
                    document.getElementById('statWarnings').textContent = data.active_warnings || 0;
                    document.getElementById('connectionStatus').innerHTML = '<span class="w-2 h-2 rounded-full bg-green-500 mr-2 pulse"></span>Онлайн';
                    document.getElementById('dbStatus').innerHTML = data.db_connected ? '<span class="w-2 h-2 rounded-full bg-green-500 mr-2"></span>MongoDB' : '<span class="w-2 h-2 rounded-full bg-red-500 mr-2"></span>MongoDB';
                    
                    const queues = data.queues_by_category || {};
                    const queueNames = { all:'Все темы', dating:'Знакомства', friends:'Друзья', fun:'Развлечения', advice:'Советы', games:'Игры' };
                    document.getElementById('queuesList').innerHTML = Object.entries(queues).map(([k,v]) => `<div class="flex justify-between p-2 bg-card rounded"><span>${queueNames[k]||k}</span><span class="bg-blue-500/20 px-2 rounded">${v}</span></div>`).join('');
                    
                    const rooms = {};
                    (data.connections || []).forEach(c => { if(c.room_id) { if(!rooms[c.room_id]) rooms[c.room_id]=[]; rooms[c.room_id].push(c); } });
                    document.getElementById('roomsList').innerHTML = Object.entries(rooms).map(([id, parts]) => `<div class="p-2 bg-card rounded"><div class="text-sm font-medium">${id.substring(0,12)}...</div><div class="text-xs text-muted">${parts.length} участников</div><div class="text-xs text-muted mt-1">${parts.map(p=>p.user_id).join(', ')}</div></div>`).join('');
                }
            } catch(e) { console.error(e); }
        }

        async function loadConnections() {
            try {
                const res = await fetch(`${BACKEND_URL}/admin/connections`, { headers: getAuthHeaders() });
                if (res.ok) {
                    const conns = await res.json();
                    document.getElementById('connCount').textContent = conns.length;
                    document.getElementById('connTable').innerHTML = conns.map(c => {
                        const statusClass = c.status === 'connected' ? 'status-online' : (c.status === 'searching' ? 'status-warning' : 'status-offline');
                        const statusText = c.status === 'connected' ? 'В разговоре' : (c.status === 'searching' ? 'Поиск' : 'Ожидание');
                        return `<tr><td class="py-2"><div class="text-sm">${c.user_id?.substring(0,25)}</div><div class="text-xs text-muted">${c.browser_id?.substring(0,15)||''}</div></td><td><span class="status-badge ${statusClass}">${statusText}</span></td><td class="font-mono text-xs">${c.ip_address||'-'}</td><td class="font-mono text-xs">${c.room_id?.substring(0,8)||'-'}<td><td class="text-xs text-muted">${new Date(c.last_activity).toLocaleTimeString()}</td><td><div class="flex gap-1"><button onclick="quickMod('${c.user_id}','warn')" class="text-yellow-400 hover:bg-yellow-500/20 p-1 rounded" title="Предупредить">⚠️</button><button onclick="quickMod('${c.user_id}','mute')" class="text-orange-400 hover:bg-orange-500/20 p-1 rounded" title="Заглушить">🔇</button><button onclick="quickMod('${c.user_id}','kick')" class="text-red-400 hover:bg-red-500/20 p-1 rounded" title="Выгнать">👢</button><button onclick="quickMod('${c.user_id}','ban')" class="text-red-400 hover:bg-red-500/20 p-1 rounded" title="Забанить">🚫</button><button onclick="goToAdvancedBan('${c.user_id}')" class="text-purple-400 hover:bg-purple-500/20 p-1 rounded" title="Расширенный бан">🔒</button></div></td></tr>`;
                    }).join('');
                }
            } catch(e) { console.error(e); }
        }

        function quickMod(userId, action) {
            document.getElementById('modUserId').value = userId;
            document.getElementById('modAction').value = action;
            if (action === 'mute' || action === 'ban') document.getElementById('durationField').classList.remove('hidden');
            document.querySelector('.tab-btn[data-tab="moderation"]').click();
        }

        function quickModAction(action) {
            currentQuickAction = action;
            document.getElementById('quickActionTitle').textContent = action === 'warn' ? 'Предупреждение' : (action === 'mute' ? 'Заглушить' : (action === 'kick' ? 'Выгнать' : 'Забанить'));
            document.getElementById('quickDurationDiv').classList.toggle('hidden', action !== 'mute' && action !== 'ban');
            document.getElementById('quickActionModal').classList.remove('hidden');
        }

        function closeQuickModal() { document.getElementById('quickActionModal').classList.add('hidden'); currentQuickAction = null; }

        async function confirmQuickAction() {
            const userId = document.getElementById('quickUserId').value;
            const reason = document.getElementById('quickReason').value;
            if (!userId) { alert('Введите ID пользователя'); return; }
            const data = { user_id: userId, reason: reason };
            if (currentQuickAction === 'mute') data.duration_minutes = parseInt(document.getElementById('quickDuration').value) || 60;
            if (currentQuickAction === 'ban') data.duration_hours = parseInt(document.getElementById('quickDuration').value) || 24;
            const res = await fetch(`${BACKEND_URL}/moderation/${currentQuickAction}`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify(data) });
            if (res.ok) { alert(`✅ ${currentQuickAction} выполнено`); closeQuickModal(); loadConnections(); loadBlockedUsers(); loadStats(); }
            else alert('Ошибка');
        }

        async function performModeration() {
            const userId = document.getElementById('modUserId').value;
            const action = document.getElementById('modAction').value;
            const reason = document.getElementById('modReason').value;
            const notes = document.getElementById('modNotes').value;
            if (!userId) { alert('Введите ID пользователя'); return; }
            const data = { user_id: userId, reason: reason, notes: notes };
            if (action === 'mute') data.duration_minutes = parseInt(document.getElementById('modDuration').value) || 60;
            if (action === 'ban') data.duration_hours = parseInt(document.getElementById('modDuration').value) || 24;
            try {
                const res = await fetch(`${BACKEND_URL}/moderation/${action}`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify(data) });
                if (res.ok) { alert(`✅ ${action} выполнено`); document.getElementById('modUserId').value = ''; document.getElementById('modNotes').value = ''; loadConnections(); loadBlockedUsers(); loadStats(); }
                else alert('Ошибка');
            } catch(e) { alert('Ошибка'); }
        }

        async function checkUserStatus() {
            const userId = document.getElementById('checkUserId').value;
            if (!userId) return;
            try {
                const res = await fetch(`${BACKEND_URL}/moderation/user/${userId}`, { headers: getAuthHeaders() });
                if (res.ok) {
                    const s = await res.json();
                    document.getElementById('userStatusPlaceholder').classList.add('hidden');
                    document.getElementById('userStatusResult').classList.remove('hidden');
                    document.getElementById('userStatusResult').innerHTML = `
                        <div class="grid grid-cols-2 gap-3">
                            <div class="bg-card p-3 rounded text-center"><div class="text-xs text-muted">Бан</div><div class="font-bold ${s.is_banned?'text-red-400':'text-green-400'}">${s.is_banned?'Забанен':'Активен'}</div></div>
                            <div class="bg-card p-3 rounded text-center"><div class="text-xs text-muted">Мут</div><div class="font-bold ${s.is_muted?'text-orange-400':'text-green-400'}">${s.is_muted?'Заглушен':'Активен'}</div></div>
                        </div>
                        <div class="bg-card p-3 rounded"><div class="text-xs text-muted mb-2">Предупреждения (${s.warnings_count})</div>${s.active_warnings?.map(w=>`<div class="text-xs text-yellow-300">• ${w.reason}</div>`).join('')||'<div class="text-xs text-muted">Нет</div>'}</div>
                        <div class="bg-card p-3 rounded"><div class="text-xs text-muted">Жалобы получено: ${s.total_reports}</div></div>
                    `;
                } else alert('Пользователь не найден');
            } catch(e) { alert('Ошибка'); }
        }

        async function loadBlockedUsers() {
    try {
        const res = await fetch(`${BACKEND_URL}/admin/blocked`, { headers: getAuthHeaders() });
        if (!res.ok) return;
        
        const data = await res.json();
        const users = data.blocked_users || [];
        const ips = data.blocked_ips || [];
        const browsers = data.blocked_browsers || [];
        const fingerprints = data.blocked_fingerprints || [];
        const subnets = data.blocked_subnets || [];
        
        const container = document.getElementById('blockedList');
        if (!container) return;
        
        if (users.length === 0 && ips.length === 0 && browsers.length === 0 && fingerprints.length === 0 && subnets.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-400">Нет заблокированных</div>';
            return;
        }
        
        let html = '';
        
        // ===== Баны по ID =====
        if (users.length > 0) {
            html += '<div class="mb-3"><div class="text-xs text-gray-400 mb-2 font-bold">👤 ПО ID (' + users.length + ')</div>';
            for (let i = 0; i < users.length; i++) {
                const u = users[i];
                html += `<div class="flex justify-between items-center p-2 bg-gray-800/30 rounded mb-1">
                    <div class="flex-1 min-w-0">
                        <div class="text-sm font-mono truncate">${u.user_id || 'unknown'}</div>
                        <div class="text-xs text-gray-400">${u.reason || 'без причины'} • ${u.expires_at ? new Date(u.expires_at).toLocaleString() : 'Перманентно'}</div>
                    </div>
                    <button onclick="unblockUser('${u.user_id}')" class="text-green-400 hover:text-green-300 text-sm ml-2 flex-shrink-0">🔓 Разбанить</button>
                </div>`;
            }
            html += '</div>';
        }
        
        // ===== Баны по IP =====
        if (ips.length > 0) {
            html += '<div class="mb-3"><div class="text-xs text-gray-400 mb-2 font-bold">🌐 ПО IP (' + ips.length + ')</div>';
            for (let i = 0; i < ips.length; i++) {
                const u = ips[i];
                html += `<div class="flex justify-between items-center p-2 bg-gray-800/30 rounded mb-1">
                    <div class="flex-1 min-w-0">
                        <div class="text-sm font-mono truncate">${u.ip || 'unknown'}</div>
                        <div class="text-xs text-gray-400">${u.reason || 'без причины'} • ${u.expires_at ? new Date(u.expires_at).toLocaleString() : 'Перманентно'}</div>
                    </div>
                    <button onclick="unblockIp('${u.ip}')" class="text-green-400 hover:text-green-300 text-sm ml-2 flex-shrink-0">🔓 Разбанить</button>
                </div>`;
            }
            html += '</div>';
        }
        
        // ===== Баны по Browser =====
        if (browsers.length > 0) {
            html += '<div class="mb-3"><div class="text-xs text-gray-400 mb-2 font-bold">🖥️ ПО BROWSER (' + browsers.length + ')</div>';
            for (let i = 0; i < browsers.length; i++) {
                const u = browsers[i];
                html += `<div class="flex justify-between items-center p-2 bg-gray-800/30 rounded mb-1">
                    <div class="flex-1 min-w-0">
                        <div class="text-sm font-mono truncate">${(u.browser_id || 'unknown').substring(0, 20)}...</div>
                        <div class="text-xs text-gray-400">${u.reason || 'без причины'} • ${u.expires_at ? new Date(u.expires_at).toLocaleString() : 'Перманентно'}</div>
                    </div>
                    <button onclick="unblockBrowser('${u.browser_id}')" class="text-green-400 hover:text-green-300 text-sm ml-2 flex-shrink-0">🔓 Разбанить</button>
                </div>`;
            }
            html += '</div>';
        }
        
        // ===== Баны по Fingerprint =====
        if (fingerprints.length > 0) {
            html += '<div class="mb-3"><div class="text-xs text-gray-400 mb-2 font-bold">🆔 ПО FINGERPRINT (' + fingerprints.length + ')</div>';
            for (let i = 0; i < fingerprints.length; i++) {
                const u = fingerprints[i];
                html += `<div class="flex justify-between items-center p-2 bg-gray-800/30 rounded mb-1">
                    <div class="flex-1 min-w-0">
                        <div class="text-sm font-mono truncate">${(u.fingerprint || 'unknown').substring(0, 20)}...</div>
                        <div class="text-xs text-gray-400">${u.reason || 'без причины'} • ${u.expires_at ? new Date(u.expires_at).toLocaleString() : 'Перманентно'}</div>
                    </div>
                    <button onclick="unblockFingerprint('${u.fingerprint}')" class="text-green-400 hover:text-green-300 text-sm ml-2 flex-shrink-0">🔓 Разбанить</button>
                </div>`;
            }
            html += '</div>';
        }
        
        // ===== Баны по Subnet =====
        if (subnets.length > 0) {
            html += '<div class="mb-3"><div class="text-xs text-gray-400 mb-2 font-bold">🔗 ПО ПОДСЕТИ (' + subnets.length + ')</div>';
            for (let i = 0; i < subnets.length; i++) {
                const u = subnets[i];
                html += `<div class="flex justify-between items-center p-2 bg-gray-800/30 rounded mb-1">
                    <div class="flex-1 min-w-0">
                        <div class="text-sm font-mono truncate">${u.subnet || 'unknown'}</div>
                        <div class="text-xs text-gray-400">${u.reason || 'без причины'} • ${u.expires_at ? new Date(u.expires_at).toLocaleString() : 'Перманентно'}</div>
                    </div>
                    <button onclick="unblockSubnet('${u.subnet}')" class="text-green-400 hover:text-green-300 text-sm ml-2 flex-shrink-0">🔓 Разбанить</button>
                </div>`;
            }
            html += '</div>';
        }
        
        container.innerHTML = html;
    } catch(e) {
        console.error('loadBlockedUsers error:', e);
    }
}

        async function unblockUser(userId) {
            if (!confirm(`Разблокировать ${userId}?`)) return;
            await fetch(`${BACKEND_URL}/admin/unblock/${userId}`, { method: 'POST', headers: getAuthHeaders() });
            loadBlockedUsers(); loadStats();
        }

        async function unblockIp(ip) {
    if (!confirm(`Разблокировать IP ${ip}?`)) return;
    try {
        const res = await fetch(`${BACKEND_URL}/admin/unblock-ip/${encodeURIComponent(ip)}`, { 
            method: 'POST', 
            headers: getAuthHeaders() 
        });
        if (res.ok) {
            alert('IP разблокирован');
            loadBlockedUsers();
            loadStats();
        } else {
            const err = await res.json();
            alert('Ошибка: ' + (err.detail || 'Не удалось разблокировать'));
        }
    } catch(e) {
        alert('Ошибка соединения: ' + e.message);
    }
}

async function unblockBrowser(browserId) {
    if (!confirm(`Разблокировать браузер ${browserId.substring(0, 15)}...?`)) return;
    try {
        const res = await fetch(`${BACKEND_URL}/admin/unblock-browser/${encodeURIComponent(browserId)}`, { 
            method: 'POST', 
            headers: getAuthHeaders() 
        });
        if (res.ok) {
            alert('Browser разблокирован');
            loadBlockedUsers();
            loadStats();
        } else {
            const err = await res.json();
            alert('Ошибка: ' + (err.detail || 'Не удалось разблокировать'));
        }
    } catch(e) {
        alert('Ошибка соединения: ' + e.message);
    }
}

async function unblockFingerprint(fingerprint) {
    if (!confirm(`Разблокировать fingerprint ${fingerprint.substring(0, 15)}...?`)) return;
    try {
        const res = await fetch(`${BACKEND_URL}/admin/unblock-fingerprint/${encodeURIComponent(fingerprint)}`, { 
            method: 'POST', 
            headers: getAuthHeaders() 
        });
        if (res.ok) {
            alert('Fingerprint разблокирован');
            loadBlockedUsers();
            loadStats();
        } else {
            const err = await res.json();
            alert('Ошибка: ' + (err.detail || 'Не удалось разблокировать'));
        }
    } catch(e) {
        alert('Ошибка соединения: ' + e.message);
    }
}

async function unblockSubnet(subnet) {
    if (!confirm(`Разблокировать подсеть ${subnet}?`)) return;
    try {
        const res = await fetch(`${BACKEND_URL}/admin/unblock-subnet/${encodeURIComponent(subnet)}`, { 
            method: 'POST', 
            headers: getAuthHeaders() 
        });
        if (res.ok) {
            alert('Подсеть разблокирована');
            loadBlockedUsers();
            loadStats();
        } else {
            const err = await res.json();
            alert('Ошибка: ' + (err.detail || 'Не удалось разблокировать'));
        }
    } catch(e) {
        alert('Ошибка соединения: ' + e.message);
    }
}

        async function loadReports(filter) {
            if (filter) currentReportFilter = filter;
            const url = currentReportFilter === 'all' ? `${BACKEND_URL}/moderation/reports?limit=100` : currentReportFilter === 'active' ? `${BACKEND_URL}/moderation/reports?resolved=false&limit=100` : `${BACKEND_URL}/moderation/reports?resolved=true&limit=100`;
            try {
                const res = await fetch(url, { headers: getAuthHeaders() });
                if (res.ok) {
                    const data = await res.json();
                    const reports = data.reports || [];
                    document.getElementById('reportsTable').innerHTML = reports.map(r => `<tr><td class="py-2">${r.reporter_id}</td><td>${r.reported_user_id}</td><td class="capitalize">${r.reason}</td><td class="text-xs">${new Date(r.timestamp).toLocaleString()}</td><td><span class="status-badge ${r.resolved?'bg-green-500/20 text-green-400':'bg-yellow-500/20 text-yellow-400'}">${r.resolved?'Решена':'Активна'}</span></td><td>${!r.resolved ? `<button onclick="openResolveModal('${r._id}')" class="text-green-400 hover:text-green-300">✅ Решить</button>` : ''}</td></tr>`).join('');
                }
            } catch(e) { console.error(e); }
        }

        function openResolveModal(id) { currentResolveReportId = id; document.getElementById('resolveReportModal').classList.remove('hidden'); }
        function closeResolveModal() { document.getElementById('resolveReportModal').classList.add('hidden'); currentResolveReportId = null; }
        async function confirmResolveReport() {
            if (!currentResolveReportId) return;
            const action = document.getElementById('resolveAction').value;
            const notes = document.getElementById('resolveNotes').value;
            await fetch(`${BACKEND_URL}/moderation/reports/${currentResolveReportId}/resolve`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify({ action, moderator_notes: notes }) });
            closeResolveModal(); loadReports(currentReportFilter); loadStats();
        }

        async function loadMessages() {
            try {
                const res = await fetch(`${BACKEND_URL}/admin/messages/recent?limit=10`, { headers: getAuthHeaders() });
                if (res.ok) {
                    const msgs = await res.json();
                    document.getElementById('recentMessagesList').innerHTML = msgs.length ? msgs.map(m => `<div class="p-2 bg-card rounded"><div class="text-sm font-medium">${m.title}</div><div class="text-xs text-muted">${m.text?.substring(0,100)}</div><div class="text-xs text-muted mt-1">📨 Получателей: ${m.recipients_count||1}</div></div>`).join('') : '<div class="text-center text-muted">Нет сообщений</div>';
                }
            } catch(e) { console.error(e); }
        }

        async function sendMessage() {
            const recipient = document.getElementById('msgRecipient').value;
            const type = document.getElementById('msgType').value;
            const title = document.getElementById('msgTitle').value;
            const text = document.getElementById('msgText').value;
            if (!title || !text) { alert('Заполните заголовок и текст'); return; }
            const data = { type, title, text, recipient };
            if (recipient === 'specific') data.specific_user_id = document.getElementById('specificUserId').value;
            try {
                const res = await fetch(`${BACKEND_URL}/admin/messages/send`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify(data) });
                if (res.ok) { alert('Сообщение отправлено'); document.getElementById('msgTitle').value = ''; document.getElementById('msgText').value = ''; loadMessages(); }
                else alert('Ошибка');
            } catch(e) { alert('Ошибка'); }
        }

        async function loadVideoData() {
            try {
                const res = await fetch(`${BACKEND_URL}/admin/video/stats`, { headers: getAuthHeaders() });
                if (res.ok) {
                    const d = await res.json();
                    document.getElementById('videoActiveUsers').textContent = d.active_video_users || 0;
                    document.getElementById('videoTotalOrders').textContent = d.total_orders || 0;
                    document.getElementById('videoRevenue').textContent = `${d.total_revenue || 0} ₽`;
                    document.getElementById('activeTariffs').textContent = d.active_tariffs || 0;
                    document.getElementById('globalVideoToggle').checked = d.video_mode_enabled;
                }
            } catch(e) { console.error(e); }
        }

        async function toggleGlobalVideo() {
            const enabled = document.getElementById('globalVideoToggle').checked;
            await fetch(`${BACKEND_URL}/admin/video/toggle-global`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify({ enabled }) });
        }

        async function loadTariffs() {
            try {
                const res = await fetch(`${BACKEND_URL}/admin/video/tariffs`, { headers: getAuthHeaders() });
                if (res.ok) {
                    const data = await res.json();
                    const tariffs = data.tariffs || [];
                    document.getElementById('tariffsList').innerHTML = tariffs.map((t,i) => `<div class="flex justify-between items-center p-2 bg-card rounded"><div><span class="font-medium">${t.name}</span> <span class="text-green-400">${t.price} ₽</span> <span class="text-xs text-muted">${t.minutes===0?'∞':t.minutes+'мин'}</span></div><div><button onclick="editTariff(${i})" class="text-blue-400 hover:text-blue-300 mr-2">✏️</button><button onclick="deleteTariff('${t.id}')" class="text-red-400 hover:text-red-300">🗑️</button></div></div>`).join('');
                }
            } catch(e) { console.error(e); }
        }

        function showTariffModal() { currentEditTariffId = null; document.getElementById('tariffModalTitle').textContent = 'Новый тариф'; document.getElementById('tariffModal').classList.remove('hidden'); }
        function closeTariffModal() { document.getElementById('tariffModal').classList.add('hidden'); }
        async function saveTariff() {
            const name = document.getElementById('tariffName').value;
            const minutes = parseInt(document.getElementById('tariffMinutes').value);
            const price = parseFloat(document.getElementById('tariffPrice').value);
            const is_active = document.getElementById('tariffActive').checked;
            if (!name || !price) { alert('Заполните поля'); return; }
            const data = { name, minutes, price, is_active };
            let url = `${BACKEND_URL}/admin/video/tariffs`, method = 'POST';
            if (currentEditTariffId) { url = `${BACKEND_URL}/admin/video/tariffs/${currentEditTariffId}`; method = 'PUT'; }
            const res = await fetch(url, { method, headers: getAuthHeaders(), body: JSON.stringify(data) });
            if (res.ok) { closeTariffModal(); loadTariffs(); loadVideoData(); }
            else alert('Ошибка');
        }
        async function editTariff(index) {
            const res = await fetch(`${BACKEND_URL}/admin/video/tariffs`, { headers: getAuthHeaders() });
            if (res.ok) {
                const data = await res.json();
                const t = data.tariffs[index];
                if (t) {
                    currentEditTariffId = t.id;
                    document.getElementById('tariffName').value = t.name;
                    document.getElementById('tariffMinutes').value = t.minutes;
                    document.getElementById('tariffPrice').value = t.price;
                    document.getElementById('tariffActive').checked = t.is_active;
                    document.getElementById('tariffModalTitle').textContent = 'Редактировать тариф';
                    document.getElementById('tariffModal').classList.remove('hidden');
                }
            }
        }
        async function deleteTariff(id) { if (confirm('Удалить тариф?')) { await fetch(`${BACKEND_URL}/admin/video/tariffs/${id}`, { method: 'DELETE', headers: getAuthHeaders() }); loadTariffs(); loadVideoData(); } }

        async function loadAudioAds() {
    console.log('🔄 [LOAD-AUDIO] Загрузка...');
    const container = document.getElementById('audioAdsList');
    if (!container) return;

    try {
        const res = await fetch(`${BACKEND_URL}/admin/ads/audio/list`, {
            headers: getAuthHeaders()
        });

        if (!res.ok) {
            container.innerHTML = '<div class="text-center py-4 text-red-400">Ошибка загрузки</div>';
            return;
        }

        const data = await res.json();
        const ads = data.ads || [];
        console.log('✅ [LOAD-AUDIO] Найдено:', ads.length);

        if (ads.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-400">Пока нет аудио-рекламы</div>';
            return;
        }

        // Рендерим базовый список
        container.innerHTML = ads.map(ad => {
            // Определяем метку периода
            const periodLabels = {
                'day': '📅 1 день',
                'week': '📅 Неделя',
                'month': '📅 Месяц',
                'year': '📅 Год',
                'always': '♾️ Всегда',
                'custom': '📅 Свой'
            };
            const periodLabel = periodLabels[ad.period_preset || 'always'] || '♾️ Всегда';
            const isActive = ad.is_active !== false;
            const statusLabel = isActive ? '🟢 Вкл' : '🔴 Выкл';
            
            return `
            <div class="flex justify-between items-center p-2 bg-card rounded mb-2 ${isActive ? '' : 'opacity-60'}">
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 flex-wrap mb-1">
                        <div class="text-sm font-medium">${ad.name || 'Без названия'}</div>
                        <span class="text-xs px-2 py-0.5 rounded ${isActive ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'}">${statusLabel}</span>
                        <span class="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-300">${periodLabel}</span>
                    </div>
                    <div class="text-xs text-muted mb-1">Показов: ${ad.play_count || 0} · ${ad.duration || 30}с${ad.show_from ? ' · с ' + new Date(ad.show_from).toLocaleDateString() : ''}${ad.show_until ? ' · по ' + new Date(ad.show_until).toLocaleDateString() : ''}</div>
                    <audio controls class="h-8 mt-1 w-full" data-ad-id="${ad.id}" preload="none"></audio>
                </div>
                <div class="flex gap-1 ml-2 flex-shrink-0">
                    <button onclick="openAudioScheduleModal('${ad.id}')" class="text-blue-400 hover:text-blue-300 p-1 text-lg" title="Настроить период показа">⚙️</button>
                    <button onclick="deleteAudioAd('${ad.id}')" class="text-red-400 hover:text-red-300 p-1 text-lg" title="Удалить">🗑️</button>
                </div>
            </div>
            `;
        }).join('');

        // Загружаем file_data отдельно для каждого audio
        for (const ad of ads) {
            try {
                const dataRes = await fetch(`${BACKEND_URL}/admin/ads/audio/${ad.id}/data`, {
                    headers: getAuthHeaders()
                });
                if (dataRes.ok) {
                    const dataJson = await dataRes.json();
                    const audioEl = container.querySelector(`audio[data-ad-id="${ad.id}"]`);
                    if (audioEl && dataJson.file_data) {
                        audioEl.src = dataJson.file_data;
                    }
                }
            } catch (e) {
                console.warn(`⚠️ Не удалось загрузить файл для ${ad.id}:`, e);
            }
        }
    } catch (err) {
        console.error('❌ [LOAD-AUDIO]', err);
        container.innerHTML = '<div class="text-center py-4 text-red-400">Ошибка: ' + err.message + '</div>';
    }
}

        async function uploadAudioAd() {
            const file = document.getElementById('audioFile').files[0];
            const name = document.getElementById('audioName').value;
            if (!file) { alert('Выберите файл'); return; }
            const reader = new FileReader();
            reader.onload = async (e) => {
                await fetch(`${BACKEND_URL}/admin/ads/audio/upload`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify({ name: name || file.name, file_data: e.target.result, duration: 30 }) });
                loadAudioAds();
                document.getElementById('audioFile').value = '';
                document.getElementById('audioName').value = '';
            };
            reader.readAsDataURL(file);
        }
        async function deleteAudioAd(id) { if(confirm('Удалить?')){ await fetch(`${BACKEND_URL}/admin/ads/audio/${id}`, { method: 'DELETE', headers: getAuthHeaders() }); loadAudioAds(); } }

        async function loadVideoAds() {
    console.log('🔄 [LOAD-VIDEO] Загрузка...');
    const container = document.getElementById('videoAdsList');
    if (!container) return;

    try {
        const res = await fetch(`${BACKEND_URL}/admin/ads/video/list`, {
            headers: getAuthHeaders()
        });

        if (!res.ok) {
            container.innerHTML = '<div class="text-center py-4 text-red-400">Ошибка загрузки</div>';
            return;
        }

        const data = await res.json();
        const ads = data.ads || [];
        console.log('✅ [LOAD-VIDEO] Найдено:', ads.length);

        if (ads.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-400">Пока нет видео-рекламы</div>';
            return;
        }

        container.innerHTML = ads.map(ad => {
            const periodLabels = {
                'day': '📅 1 день',
                'week': '📅 Неделя',
                'month': '📅 Месяц',
                'year': '📅 Год',
                'always': '♾️ Всегда',
                'custom': '📅 Свой'
            };
            const periodLabel = periodLabels[ad.period_preset || 'always'] || '♾️ Всегда';
            const isActive = ad.is_active !== false;
            const statusLabel = isActive ? '🟢 Вкл' : '🔴 Выкл';
            
            return `
            <div class="flex justify-between items-center p-2 bg-card rounded mb-2 ${isActive ? '' : 'opacity-60'}">
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 flex-wrap mb-1">
                        <div class="text-sm font-medium">${ad.name || 'Без названия'}</div>
                        <span class="text-xs px-2 py-0.5 rounded ${isActive ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'}">${statusLabel}</span>
                        <span class="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-300">${periodLabel}</span>
                    </div>
                    <div class="text-xs text-muted mb-1">Показов: ${ad.play_count || 0} · ${ad.duration || 30}с${ad.show_from ? ' · с ' + new Date(ad.show_from).toLocaleDateString() : ''}${ad.show_until ? ' · по ' + new Date(ad.show_until).toLocaleDateString() : ''}</div>
                    <video controls class="h-16 mt-1 w-full" data-ad-id="${ad.id}" preload="none"></video>
                </div>
                <div class="flex gap-1 ml-2 flex-shrink-0">
                    <button onclick="openVideoScheduleModal('${ad.id}')" class="text-blue-400 hover:text-blue-300 p-1 text-lg" title="Настроить период показа">⚙️</button>
                    <button onclick="deleteVideoAd('${ad.id}')" class="text-red-400 hover:text-red-300 p-1 text-lg" title="Удалить">🗑️</button>
                </div>
            </div>
            `;
        }).join('');

        for (const ad of ads) {
            try {
                const dataRes = await fetch(`${BACKEND_URL}/admin/ads/video/${ad.id}/data`, {
                    headers: getAuthHeaders()
                });
                if (dataRes.ok) {
                    const dataJson = await dataRes.json();
                    const videoEl = container.querySelector(`video[data-ad-id="${ad.id}"]`);
                    if (videoEl && dataJson.file_data) {
                        videoEl.src = dataJson.file_data;
                    }
                }
            } catch (e) {
                console.warn(`⚠️ Не удалось загрузить видео для ${ad.id}:`, e);
            }
        }
    } catch (err) {
        console.error('❌ [LOAD-VIDEO]', err);
        container.innerHTML = '<div class="text-center py-4 text-red-400">Ошибка: ' + err.message + '</div>';
    }
}

        async function uploadVideoAd() {
            const file = document.getElementById('videoFile').files[0];
            const name = document.getElementById('videoName').value;
            if (!file) { alert('Выберите файл'); return; }
            const reader = new FileReader();
            reader.onload = async (e) => {
                await fetch(`${BACKEND_URL}/admin/ads/video/upload`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify({ name: name || file.name, file_data: e.target.result, duration: 30 }) });
                loadVideoAds();
                document.getElementById('videoFile').value = '';
                document.getElementById('videoName').value = '';
            };
            reader.readAsDataURL(file);
        }
        async function deleteVideoAd(id) { if(confirm('Удалить?')){ await fetch(`${BACKEND_URL}/admin/ads/video/${id}`, { method: 'DELETE', headers: getAuthHeaders() }); loadVideoAds(); } }

        async function loadAdSettings() {
            const res = await fetch(`${BACKEND_URL}/admin/ads/settings`, { headers: getAuthHeaders() });
            if (res.ok) {
                const s = await res.json();
                document.getElementById('audioEnabled').checked = s.audio_enabled;
                document.getElementById('videoEnabled').checked = s.video_enabled;
                document.getElementById('audioInterval').value = s.audio_interval_seconds || 300;
                document.getElementById('videoInterval').value = s.video_interval_seconds || 600;
            }
        }
        async function saveAdSettings() {
            const settings = { audio_enabled: document.getElementById('audioEnabled').checked, video_enabled: document.getElementById('videoEnabled').checked, audio_interval_seconds: parseInt(document.getElementById('audioInterval').value), video_interval_seconds: parseInt(document.getElementById('videoInterval').value) };
            await fetch(`${BACKEND_URL}/admin/ads/settings`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify(settings) });
            alert('Настройки сохранены');
        }

        async function loadPremiumStats() {
            const res = await fetch(`${BACKEND_URL}/admin/premium/stats`, { headers: getAuthHeaders() });
            if (res.ok) {
                const d = await res.json();
                document.getElementById('premiumActive').textContent = d.active_premium_users || 0;
                document.getElementById('premiumRevenue').textContent = `${d.total_revenue || 0} ₽`;
                document.getElementById('premiumPrice').textContent = `${d.premium_price || 499} ₽`;
                document.getElementById('premiumCount').textContent = d.active_premium_users || 0;
            }
        }
        
        async function loadPremiumUsers() {
    console.log('🔄 [PREMIUM] Загрузка пользователей...');
    const container = document.getElementById('premiumUsersList');
    if (!container) {
        console.error('❌ [PREMIUM] Элемент #premiumUsersList не найден');
        return;
    }

    container.innerHTML = '<div class="text-center py-4 text-gray-400">Загрузка...</div>';

    try {
        const res = await fetch(`${BACKEND_URL}/admin/premium/users?limit=50`, {
            headers: getAuthHeaders()
        });

        console.log('📥 [PREMIUM] HTTP статус:', res.status);

        if (!res.ok) {
            container.innerHTML = `<div class="text-center py-4 text-red-400">Ошибка сервера: ${res.status}</div>`;
            return;
        }

        let data;
        try {
            data = await res.json();
        } catch (parseErr) {
            console.error('❌ [PREMIUM] Ошибка парсинга JSON:', parseErr);
            container.innerHTML = '<div class="text-center py-4 text-red-400">Сервер вернул битый ответ. Попробуйте ещё раз.</div>';
            return;
        }

        console.log('📥 [PREMIUM] Данные:', data);

        const users = data.users || [];
        const total = data.total || users.length;

        if (users.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-400">Пока нет премиум-пользователей</div>';
            return;
        }

        container.innerHTML = users.map(u => `
            <div class="flex justify-between items-center p-2 bg-card rounded mb-2">
                <div class="flex-1 min-w-0">
                    <div class="text-sm font-medium truncate">${u.user_id || 'unknown'}</div>
                    <div class="text-xs text-muted">
                        До: ${u.end_date ? new Date(u.end_date).toLocaleDateString('ru-RU') : '—'}
                        · ${u.duration_days || 0} дней
                        ${u.source ? ' · ' + u.source : ''}
                    </div>
                </div>
                <button onclick="deactivatePremium('${u.user_id}')" class="text-red-400 hover:text-red-300 ml-2 flex-shrink-0">🚫</button>
            </div>
        `).join('');

        console.log(`✅ [PREMIUM] Показано ${users.length} из ${total}`);
    } catch (err) {
        console.error('❌ [PREMIUM] Ошибка fetch:', err);
        container.innerHTML = `<div class="text-center py-4 text-red-400">Ошибка соединения: ${err.message}</div>`;
    }
}

        // ============================================================
// НАСТРОЙКИ ПРЕМИУМ
// ============================================================
async function loadPremiumSettings() {
    console.log('⚙️ [PREMIUM-SETTINGS] Загрузка настроек...');
    try {
        const res = await fetch(`${BACKEND_URL}/admin/premium/settings`, {
            headers: getAuthHeaders()
        });

        if (!res.ok) {
            console.error('❌ [PREMIUM-SETTINGS] Ошибка HTTP:', res.status);
            return;
        }

        const s = await res.json();
        console.log('📥 [PREMIUM-SETTINGS] Настройки:', s);

        const priceEl = document.getElementById('premiumSettingsPrice');
        const durationEl = document.getElementById('premiumSettingsDuration');
        const adsEl = document.getElementById('premiumDisableAds');
        const supportEl = document.getElementById('premiumPrioritySupport');

        if (priceEl) priceEl.value = s.premium_price || 499;
        if (durationEl) durationEl.value = s.premium_duration_days || 30;
        if (adsEl) adsEl.checked = s.disable_ads !== false;
        if (supportEl) supportEl.checked = s.priority_support !== false;
    } catch (e) {
        console.error('❌ [PREMIUM-SETTINGS] Ошибка:', e);
    }
}

async function savePremiumSettings() {
    console.log('💾 [PREMIUM-SETTINGS] Сохранение...');

    const priceEl = document.getElementById('premiumSettingsPrice');
    const durationEl = document.getElementById('premiumSettingsDuration');
    const adsEl = document.getElementById('premiumDisableAds');
    const supportEl = document.getElementById('premiumPrioritySupport');

    const settings = {
        premium_price: parseFloat(priceEl ? priceEl.value : 499),
        premium_duration_days: parseInt(durationEl ? durationEl.value : 30),
        disable_ads: adsEl ? adsEl.checked : true,
        priority_support: supportEl ? supportEl.checked : true
    };

    try {
        const res = await fetch(`${BACKEND_URL}/admin/premium/settings`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify(settings)
        });

        if (res.ok) {
            alert('✅ Настройки премиума сохранены');
        } else {
            const err = await res.json().catch(() => ({}));
            alert('❌ Ошибка: ' + (err.detail || 'Unknown'));
        }
    } catch (e) {
        console.error('❌ [PREMIUM-SETTINGS] Ошибка:', e);
        alert('❌ Ошибка сети');
    }
}

        async function activatePremium() {
            const userId = document.getElementById('premiumUserId').value;
            let days = parseInt(document.getElementById('premiumDays').value);
            const unit = document.getElementById('premiumUnit').value;
            if (unit === 'months') days *= 30;
            if (!userId) { alert('Введите ID пользователя'); return; }
            const res = await fetch(`${BACKEND_URL}/admin/premium/activate/${userId}`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify({ duration_days: days }) });
            if (res.ok) { alert('Премиум активирован'); loadPremiumStats(); loadPremiumUsers(); }
            else alert('Ошибка');
        }
        async function deactivatePremium(userId) {
    if (!confirm(`Деактивировать премиум для ${userId}?`)) return;
    
    console.log('🚫 [PREMIUM-DEACTIVATE] Деактивирую:', userId);
    
    try {
        const res = await fetch(`${BACKEND_URL}/admin/premium/deactivate/${userId}`, {
            method: 'DELETE',
            headers: getAuthHeaders()
        });
        
        const result = await res.json();
        console.log('📥 [PREMIUM-DEACTIVATE] Ответ:', res.status, result);
        
        if (res.ok) {
            alert('✅ Премиум деактивирован');
            
            // Ждём 300мс чтобы MongoDB точно записала, и обновляем
            setTimeout(() => {
                loadPremiumStats();
                loadPremiumUsers();
            }, 300);
        } else {
            alert('❌ Ошибка: ' + (result.detail || 'Unknown error'));
        }
    } catch (e) {
        console.error('❌ [PREMIUM-DEACTIVATE] Ошибка:', e);
        alert('❌ Ошибка сети: ' + e.message);
    }
}
        async function savePremiumSettings() {
            const settings = { premium_price: parseFloat(document.getElementById('premiumSettingsPrice').value), premium_duration_days: parseInt(document.getElementById('premiumSettingsDuration').value), disable_ads: document.getElementById('premiumDisableAds').checked, priority_support: document.getElementById('premiumPrioritySupport').checked };
            await fetch(`${BACKEND_URL}/admin/premium/settings`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify(settings) });
            alert('Настройки сохранены');
        }

        async function loadPaymentSettings() {
            const res = await fetch(`${BACKEND_URL}/admin/payment/settings`, { headers: getAuthHeaders() });
            if (res.ok) {
                const d = await res.json();
                document.getElementById('paymentShopId').value = d.morune_shop_id || '';
                document.getElementById('paymentApiKey').value = d.morune_api_key || '';
                document.getElementById('paymentApiUrl').value = d.morune_payment_url || 'https://api.morune.com/v1';
                document.getElementById('paymentTestMode').checked = d.test_mode;
                document.getElementById('payTotalOrders').textContent = d.statistics?.total_orders || 0;
                document.getElementById('paySuccessOrders').textContent = d.statistics?.successful_orders || 0;
                document.getElementById('payFailedOrders').textContent = d.statistics?.failed_orders || 0;
                document.getElementById('payRevenue').textContent = `${d.statistics?.total_revenue || 0} ₽`;
                document.getElementById('paymentStatusInfo').innerHTML = `<div class="flex justify-between"><span>Статус:</span><span class="${d.test_mode?'text-yellow-400':'text-green-400'}">${d.test_mode?'Тестовый режим':'Рабочий режим'}</span></div><div class="flex justify-between"><span>Shop ID:</span><span class="font-mono text-xs">${d.morune_shop_id?.substring(0,15)}...</span></div>`;
            }
        }
        async function savePaymentSettings() {
            const settings = { morune_shop_id: document.getElementById('paymentShopId').value, morune_api_key: document.getElementById('paymentApiKey').value, morune_secret_key: document.getElementById('paymentSecretKey').value, morune_payment_url: document.getElementById('paymentApiUrl').value, test_mode: document.getElementById('paymentTestMode').checked };
            const res = await fetch(`${BACKEND_URL}/admin/payment/settings`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify(settings) });
            if (res.ok) alert('Настройки сохранены');
            else alert('Ошибка');
        }
        async function testPaymentConnection() {
            const res = await fetch(`${BACKEND_URL}/admin/payment/test`, { headers: getAuthHeaders() });
            const result = await res.json();
            alert(result.message || (result.status === 'success' ? '✅ Подключение успешно' : '❌ Ошибка подключения'));
        }
        function showTestPaymentModal() { document.getElementById('testPaymentModal').classList.remove('hidden'); }
        function closeTestPaymentModal() { document.getElementById('testPaymentModal').classList.add('hidden'); }
        async function processTestPayment() {
            const userId = document.getElementById('testUserId').value;
            const minutes = parseInt(document.getElementById('testTariff').value);
            const price = parseFloat(document.getElementById('testPrice').value);
            if (!userId) { alert('Введите ID пользователя'); return; }
            const res = await fetch(`${BACKEND_URL}/admin/payment/create-test`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify({ user_id: userId, minutes, price }) });
            if (res.ok) { alert('Тестовый платеж создан'); closeTestPaymentModal(); loadPaymentSettings(); }
            else alert('Ошибка');
        }

        async function loadSystemSettings() {
            const res = await fetch(`${BACKEND_URL}/admin/settings`, { headers: getAuthHeaders() });
            if (res.ok) {
                const s = await res.json();
                document.getElementById('sysMaintenance').checked = s.maintenance_mode;
                document.getElementById('sysRegistration').checked = s.allow_registration;
                document.getElementById('sysMaxUsers').value = s.max_users || 1000;
                document.getElementById('sysMessage').value = s.system_message || '';
            }
        }
        async function saveSystemSettings() {
            const settings = { maintenance_mode: document.getElementById('sysMaintenance').checked, allow_registration: document.getElementById('sysRegistration').checked, max_users: parseInt(document.getElementById('sysMaxUsers').value), system_message: document.getElementById('sysMessage').value };
            await fetch(`${BACKEND_URL}/admin/settings`, { method: 'POST', headers: getAuthHeaders(), body: JSON.stringify(settings) });
            alert('Настройки сохранены');
        }
        async function emergencyStop() { if (confirm('Экстренная остановка? Все пользователи будут отключены.')) { await fetch(`${BACKEND_URL}/admin/emergency-stop`, { method: 'POST', headers: getAuthHeaders() }); alert('Выполнено'); loadStats(); } }
        async function clearQueues() { if (confirm('Очистить очереди поиска?')) { await fetch(`${BACKEND_URL}/admin/queues/clear`, { method: 'POST', headers: getAuthHeaders() }); alert('Очереди очищены'); loadStats(); } }
        async function forceSync() { await fetch(`${BACKEND_URL}/admin/sync/now`, { method: 'POST', headers: getAuthHeaders() }); alert('Синхронизация выполнена'); showSyncStats(); }

        

        async function refreshLogs() {
            const res = await fetch(`${BACKEND_URL}/admin/logs?limit=50`, { headers: getAuthHeaders() });
            if (res.ok) {
                const data = await res.json();
                const logs = data.logs || [];
                document.getElementById('serverLogs').innerHTML = logs.map(l => `<div class="mb-1 border-l-2 border-blue-500 pl-2"><span class="text-gray-500">[${new Date(l.timestamp).toLocaleTimeString()}]</span> <span class="text-green-300">${l.message}</span></div>`).join('');
            }
        }

        // ============================================================
// УПРАВЛЕНИЕ ПЕРСОНАЛОМ
// ============================================================
let ALL_PERMISSIONS_CACHE = [];
let PERMISSION_LABELS = {
    "view": "👁️ Просмотр панели",
    "view_reports": "🚩 Просмотр жалоб",
    "warn": "⚠️ Предупреждения",
    "mute": "🔇 Заглушение",
    "kick": "👢 Кик",
    "ban": "🚫 Бан",
    "manage_users": "👥 Управление пользователями",
    "manage_settings": "⚙️ Настройки системы",
    "manage_ads": "📢 Управление рекламой",
    "manage_payment": "💳 Управление оплатой",
    "manage_premium": "👑 Управление премиумом",
    "manage_staff": "👔 Управление персоналом",
    "view_logs": "📋 Просмотр логов",
    "advanced_ban": "🔒 Расширенный бан"
};

async function loadStaffList() {
    console.log('👥 [STAFF] Загрузка списка...');
    const container = document.getElementById('staffList');
    if (!container) return;

    container.innerHTML = '<div class="text-center py-4 text-gray-400">Загрузка...</div>';

    try {
        const res = await fetch(`${BACKEND_URL}/admin/staff/list`, { headers: getAuthHeaders() });
        if (!res.ok) {
            container.innerHTML = '<div class="text-center py-4 text-red-400">Ошибка загрузки</div>';
            return;
        }

        const data = await res.json();
        const staff = data.staff || [];
        ALL_PERMISSIONS_CACHE = data.permissions || [];
        renderPermissionCheckboxes('staffNewPermissions', [], false);

        if (staff.length === 0) {
            container.innerHTML = '<div class="text-center py-4 text-gray-400">Нет сотрудников</div>';
            return;
        }

        container.innerHTML = staff.map(s => {
            const isAdmin = s.role === 'admin';
            const roleLabel = isAdmin ? '👑 Администратор' : '🛡️ Модератор';
            const roleColor = isAdmin ? 'bg-red-500/20 text-red-300' : 'bg-blue-500/20 text-blue-300';
            const activeLabel = s.is_active ? '🟢 Активен' : '🔴 Отключён';
            const permsCount = (s.permissions || []).length;
            const created = s.created_at ? new Date(s.created_at).toLocaleDateString('ru-RU') : '—';

            return `
            <div class="p-3 bg-gray-800/30 rounded-xl mb-2">
                <div class="flex justify-between items-start flex-wrap gap-2">
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center gap-2 flex-wrap mb-1">
                            <span class="font-bold font-mono">${s.username}</span>
                            <span class="text-xs px-2 py-0.5 rounded ${roleColor}">${roleLabel}</span>
                            <span class="text-xs px-2 py-0.5 rounded bg-gray-700">${activeLabel}</span>
                        </div>
                        <div class="text-xs text-gray-400">
                            Создан: ${created} · Прав: ${permsCount}
                            ${s.created_by ? ' · от ' + s.created_by : ''}
                        </div>
                        <div class="text-xs text-gray-500 mt-1">
                            ${(s.permissions || []).map(p => PERMISSION_LABELS[p] || p).join(' · ') || 'нет прав'}
                        </div>
                    </div>
                    <div class="flex gap-1 flex-wrap">
                        <button onclick="editStaff('${s.username}')" class="bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 px-2 py-1 rounded text-xs">✏️ Изменить</button>
                        <button onclick="quickChangePassword('${s.username}')" class="bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-300 px-2 py-1 rounded text-xs">🔑 Пароль</button>
                        <button onclick="deleteStaff('${s.username}')" class="bg-red-500/20 hover:bg-red-500/30 text-red-300 px-2 py-1 rounded text-xs">🗑️ Удалить</button>
                    </div>
                </div>
            </div>
            `;
        }).join('');

        console.log(`✅ [STAFF] Загружено ${staff.length}`);
    } catch (e) {
        console.error('❌ [STAFF] Ошибка:', e);
        container.innerHTML = `<div class="text-center py-4 text-red-400">Ошибка: ${e.message}</div>`;
    }
}

function renderPermissionCheckboxes(containerId, selectedPermissions, disabled) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const selected = selectedPermissions || [];
    const perms = ALL_PERMISSIONS_CACHE.length ? ALL_PERMISSIONS_CACHE : Object.keys(PERMISSION_LABELS);

    container.innerHTML = perms.map(p => `
        <label class="flex items-center gap-2 text-xs cursor-pointer p-1 rounded hover:bg-gray-700/40">
            <input type="checkbox" class="perm-checkbox" data-perm="${p}" ${selected.includes(p) ? 'checked' : ''} ${disabled ? 'disabled' : ''}>
            <span>${PERMISSION_LABELS[p] || p}</span>
        </label>
    `).join('');
}

function onStaffRoleChange() {
    const role = document.getElementById('staffNewRole').value;
    const container = document.getElementById('staffNewPermissions');
    if (!container) return;

    if (role === 'admin') {
        // Включаем все права и блокируем
        container.querySelectorAll('.perm-checkbox').forEach(cb => {
            cb.checked = true;
            cb.disabled = true;
        });
        container.insertAdjacentHTML('beforeend',
            '<div class="col-span-full text-xs text-yellow-400 mt-1">ℹ️ Администратор автоматически получает все права</div>');
    } else {
        container.querySelectorAll('.perm-checkbox').forEach(cb => {
            cb.disabled = false;
        });
        const note = container.querySelector('.text-yellow-400');
        if (note) note.remove();
    }
}

function getSelectedPermissions(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return [];
    return Array.from(container.querySelectorAll('.perm-checkbox:checked'))
        .map(cb => cb.dataset.perm);
}

async function createStaff() {
    const username = document.getElementById('staffNewUsername').value.trim();
    const password = document.getElementById('staffNewPassword').value;
    const role = document.getElementById('staffNewRole').value;
    const permissions = getSelectedPermissions('staffNewPermissions');

    if (!username || username.length < 3) {
        alert('Логин должен быть не короче 3 символов');
        return;
    }
    if (!password || password.length < 4) {
        alert('Пароль должен быть не короче 4 символов');
        return;
    }

    if (!confirm(`Создать ${role === 'admin' ? 'администратора' : 'модератора'} "${username}"?`)) return;

    try {
        const res = await fetch(`${BACKEND_URL}/admin/staff/create`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({
                username: username,
                password: password,
                role: role,
                permissions: permissions,
                is_active: true
            })
        });

        const result = await res.json();
        if (res.ok) {
            alert(result.message || '✅ Создано');
            document.getElementById('staffNewUsername').value = '';
            document.getElementById('staffNewPassword').value = '';
            document.getElementById('staffNewRole').value = 'moderator';
            onStaffRoleChange();
            loadStaffList();
        } else {
            alert('❌ Ошибка: ' + (result.detail || 'Unknown'));
        }
    } catch (e) {
        alert('❌ Ошибка сети: ' + e.message);
    }
}

async function deleteStaff(username) {
    if (!confirm(`⚠️ Удалить "${username}"?\n\nЭто действие необратимо.`)) return;

    try {
        const res = await fetch(`${BACKEND_URL}/admin/staff/${encodeURIComponent(username)}`, {
            method: 'DELETE',
            headers: getAuthHeaders()
        });
        const result = await res.json();
        if (res.ok) {
            alert('✅ Удалено');
            loadStaffList();
        } else {
            alert('❌ Ошибка: ' + (result.detail || 'Unknown'));
        }
    } catch (e) {
        alert('❌ Ошибка сети: ' + e.message);
    }
}

async function quickChangePassword(username) {
    const newPwd = prompt(`Введите новый пароль для "${username}" (мин. 4 символа):`);
    if (!newPwd) return;
    if (newPwd.length < 4) {
        alert('Пароль слишком короткий');
        return;
    }
    if (!confirm(`Сменить пароль для "${username}"?`)) return;

    try {
        const res = await fetch(`${BACKEND_URL}/admin/staff/${encodeURIComponent(username)}/password`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({ new_password: newPwd })
        });
        const result = await res.json();
        if (res.ok) {
            alert('✅ Пароль изменён');
        } else {
            alert('❌ Ошибка: ' + (result.detail || 'Unknown'));
        }
    } catch (e) {
        alert('❌ Ошибка сети: ' + e.message);
    }
}

async function changeStaffPassword() {
    const username = document.getElementById('pwdChangeUsername').value.trim();
    const newPwd = document.getElementById('pwdChangeNew').value;
    if (!username || !newPwd) {
        alert('Заполните все поля');
        return;
    }
    if (newPwd.length < 4) {
        alert('Пароль слишком короткий');
        return;
    }

    try {
        const res = await fetch(`${BACKEND_URL}/admin/staff/${encodeURIComponent(username)}/password`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({ new_password: newPwd })
        });
        const result = await res.json();
        if (res.ok) {
            alert('✅ Пароль изменён');
            document.getElementById('pwdChangeUsername').value = '';
            document.getElementById('pwdChangeNew').value = '';
        } else {
            alert('❌ Ошибка: ' + (result.detail || 'Unknown'));
        }
    } catch (e) {
        alert('❌ Ошибка сети: ' + e.message);
    }
}

// Модалка редактирования сотрудника
function editStaff(username) {
    // Находим данные в списке
    fetch(`${BACKEND_URL}/admin/staff/list`, { headers: getAuthHeaders() })
        .then(r => r.json())
        .then(data => {
            const s = (data.staff || []).find(x => x.username === username);
            if (!s) { alert('Не найден'); return; }

            const newRole = prompt(
                `Редактирование "${username}"\n\nВведите роль (admin/moderator):\nТекущая: ${s.role}`,
                s.role
            );
            if (newRole === null) return;
            if (newRole !== 'admin' && newRole !== 'moderator') {
                alert('Роль должна быть admin или moderator');
                return;
            }

            const newActive = confirm(
                `Пользователь активен?\n\nOK = да\nОтмена = нет\n\nТекущий статус: ${s.is_active ? 'активен' : 'отключён'}`
            );

            // Права через prompt (простой способ)
            let newPermissions = s.permissions || [];
            if (newRole === 'moderator') {
                const permsStr = prompt(
                    `Права через запятую.\n\nДоступные:\n${Object.keys(PERMISSION_LABELS).join(', ')}\n\nТекущие:`,
                    (s.permissions || []).join(', ')
                );
                if (permsStr === null) return;
                newPermissions = permsStr.split(',').map(p => p.trim()).filter(p => p);
            }

            fetch(`${BACKEND_URL}/admin/staff/${encodeURIComponent(username)}`, {
                method: 'PUT',
                headers: getAuthHeaders(),
                body: JSON.stringify({
                    role: newRole,
                    permissions: newPermissions,
                    is_active: newActive
                })
            })
            .then(r => r.json())
            .then(res => {
                if (res.status === 'success') {
                    alert('✅ Обновлено');
                    loadStaffList();
                } else {
                    alert('❌ Ошибка: ' + (res.detail || 'Unknown'));
                }
            })
            .catch(e => alert('❌ Ошибка сети: ' + e.message));
        })
        .catch(e => alert('❌ ' + e.message));
}

        async function showSyncStats() {
            const res = await fetch(`${BACKEND_URL}/admin/sync/stats`, { headers: getAuthHeaders() });
            if (res.ok) {
                const d = await res.json();
                document.getElementById('dbUsers').textContent = d.users_in_db || 0;
                document.getElementById('dbConns').textContent = d.connections_in_db || 0;
                document.getElementById('dbRooms').textContent = d.rooms_in_db || 0;
            }
        }

        function connectChatWebSocket() {
    // ===== ПОДКЛЮЧАЕМСЯ НАПРЯМУЮ К ОСНОВНОМУ СЕРВЕРУ =====
    var wsUrl = 'ws://192.168.0.103:8000/ws/chat?source=admin&username=' + encodeURIComponent('Администратор');
    console.log('🔌 [ADMIN-CHAT] Connecting to main server:', wsUrl);
    
    // ===== ЗАЩИТА ОТ ДУБЛИРУЮЩИХСЯ ПОДКЛЮЧЕНИЙ =====
    if (adminChatWs && adminChatWs.readyState === WebSocket.OPEN) {
        console.log('✅ [ADMIN-CHAT] Already connected, skipping');
        return;
    }
    if (adminChatWs && adminChatWs.readyState === WebSocket.CONNECTING) {
        console.log('⏳ [ADMIN-CHAT] Connection in progress, skipping');
        return;
    }
    if (adminChatWs && (adminChatWs.readyState === WebSocket.CLOSING || adminChatWs.readyState === WebSocket.CLOSED)) {
        adminChatWs = null;
    }
    
    try {
        adminChatWs = new WebSocket(wsUrl);
        
        adminChatWs.onopen = function() {
    console.log('✅ [ADMIN-CHAT] WebSocket connected');
    var statusEl = document.getElementById('chatStatus');
    if (statusEl) {
        statusEl.innerHTML = '● В сети';
        statusEl.className = 'text-xs px-2 py-1 rounded-full bg-green-500/20 text-green-400';
    }
    
    // ===== HISTORY REQUEST =====
    try {
        this.send(JSON.stringify({ type: 'history_request' }));
        console.log('📨 [ADMIN-CHAT] Requested history');
    } catch(e) {
        console.error('❌ [ADMIN-CHAT] History request error:', e);
    }
    
    // ===== WELCOME-СООБЩЕНИЕ (ТОЛЬКО 1 РАЗ ЗА СЕССИЮ) =====
    if (!sessionStorage.getItem('admin_welcome_sent')) {
        try {
            this.send(JSON.stringify({
                type: 'message',
                text: '👨‍💼 Администратор подключился к чату',
                sender: 'admin',
                sender_name: 'Администратор',
                timestamp: new Date().toISOString()
            }));
            sessionStorage.setItem('admin_welcome_sent', 'true');
            console.log('📨 [ADMIN-CHAT] Welcome sent (first time this session)');
        } catch(e) {
            console.error('❌ [ADMIN-CHAT] Welcome error:', e);
        }
    } else {
        console.log('ℹ️ [ADMIN-CHAT] Welcome already sent this session, skipping');
    }
};
        
        adminChatWs.onmessage = function(event) {
            // ===== ЗАЩИТА ОТ STALE =====
            if (adminChatWs !== this) {
                console.log('⚠️ [ADMIN-CHAT] Stale connection, ignoring');
                return;
            }
            
            try {
                var data = JSON.parse(event.data);
                console.log('📨 [ADMIN-CHAT] Received:', data.type);
                
                if (data.type === 'history') {
                    console.log('📨 [ADMIN-CHAT] History:', (data.messages || []).length, 'msgs');
                    renderChatMessages(data.messages || []);
                } else if (data.type === 'new_message') {
                    console.log('📨 [ADMIN-CHAT] New message');
                    addChatMessage(data.message);
                } else if (data.type === 'pong') {
                    console.log('💓 [ADMIN-CHAT] Pong');
                } else {
                    console.log('⚠️ [ADMIN-CHAT] Unknown type, ignoring:', data.type);
                }
            } catch(e) {
                console.error('❌ [ADMIN-CHAT] Parse error:', e);
            }
        };
        
        adminChatWs.onerror = function(error) {
            if (adminChatWs !== this) return;
            console.error('❌ [ADMIN-CHAT] Error:', error);
            var statusEl = document.getElementById('chatStatus');
            if (statusEl) {
                statusEl.innerHTML = '● Ошибка';
                statusEl.className = 'text-xs px-2 py-1 rounded-full bg-red-500/20 text-red-400';
            }
        };
        
        adminChatWs.onclose = function(event) {
            console.log('🔌 [ADMIN-CHAT] Closed, code:', event.code);
            if (adminChatWs !== this) {
                console.log('⚠️ [ADMIN-CHAT] Stale close, ignoring');
                return;
            }
            var statusEl = document.getElementById('chatStatus');
            if (statusEl) {
                statusEl.innerHTML = '● Подключение...';
                statusEl.className = 'text-xs px-2 py-1 rounded-full bg-yellow-500/20 text-yellow-400';
            }
            setTimeout(function() { connectChatWebSocket(); }, 5000);
        };
        
    } catch(e) {
        console.error('❌ [ADMIN-CHAT] Create error:', e);
    }
}


        function renderChatMessages(messages) {
    var container = document.getElementById('chatMessages');
    if (!container) return;
    
    container.innerHTML = '';
    if (!messages || messages.length === 0) {
        container.innerHTML = '<div class="text-center py-8 text-gray-400">💬 Сообщений пока нет</div>';
        return;
    }
    for (var i = 0; i < messages.length; i++) {
        addChatMessage(messages[i]);
    }
}

        function addChatMessage(msg) {
            const container = document.getElementById('chatMessages');
            const isAdmin = msg.sender === 'admin';
            const isModerator = msg.sender === 'moderator';
            const time = new Date(msg.timestamp).toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' });
            const senderName = msg.sender_name || (isAdmin ? 'Администратор' : (isModerator ? 'Модератор' : 'Неизвестный'));
            const avatar = isAdmin ? '👨‍💼' : (isModerator ? '🛡️' : '👤');
            let content = '';
            if (msg.type === 'voice') {
                content = `<audio controls src="${msg.data}" class="h-8 rounded"></audio>`;
            } else if (msg.type === 'file') {
                const isImage = msg.mime?.startsWith('image/');
                if (isImage) {
                    content = `<img src="${msg.data}" class="max-w-[200px] max-h-[100px] rounded cursor-pointer" onclick="window.open('${msg.data}','_blank')">`;
                } else {
                    content = `<div class="flex items-center gap-2"><span>📄 ${msg.name}</span><a href="${msg.data}" download class="text-blue-400 hover:text-blue-300">⬇️ Скачать</a></div>`;
                }
            } else {
                content = escapeHtml(msg.text || '');
            }
            const html = `<div class="chat-message ${isAdmin ? 'chat-message-admin' : ''}">
                <div class="chat-message-avatar">${avatar}</div>
                <div class="chat-message-content">
                    <div class="chat-message-sender">${senderName} • ${time}</div>
                    <div class="chat-message-text">${content}</div>
                </div>
            </div>`;
            container.insertAdjacentHTML('beforeend', html);
            container.scrollTop = container.scrollHeight;
            if (!isAdmin && !document.getElementById('chatTab').classList.contains('active')) { 
                unreadCount++; 
                const badge = document.querySelector('.tab-btn[data-tab="chat"]');
                if (badge && !badge.innerHTML.includes('🔴')) badge.innerHTML = '💬 Чат 🔴';
            }
        }

        function escapeHtml(text) { if (!text) return ''; return text.replace(/[&<>]/g, function(m) { if (m === '&') return '&amp;'; if (m === '<') return '&lt;'; if (m === '>') return '&gt;'; return m; }); }

        function sendChatMessage() {
    var input = document.getElementById('chatInput');
    if (!input) {
        console.error('❌ [ADMIN-CHAT] Input not found');
        return;
    }
    
    var text = input.value.trim();
    console.log('📤 [ADMIN-CHAT] sendChatMessage called, text:', text);
    
    // Проверяем, есть ли что отправлять
    if (!text && !recordedBlob && !attachedFile) {
        console.log('⚠️ [ADMIN-CHAT] Nothing to send');
        return;
    }
    
    // Проверяем WebSocket
    if (!adminChatWs || adminChatWs.readyState !== WebSocket.OPEN) {
        console.warn('⚠️ [ADMIN-CHAT] WebSocket not open, reconnecting...');
        alert('Чат не подключен. Переподключаем...');
        connectChatWebSocket();
        return;
    }
    
    // ===== ОТПРАВКА ТЕКСТА =====
    if (text) {
        var msg = {
            type: 'message',
            text: text,
            sender: 'admin',
            sender_name: 'Администратор',
            timestamp: new Date().toISOString()
        };
        
        console.log('📤 [ADMIN-CHAT] Sending message:', msg);
        
        try {
            adminChatWs.send(JSON.stringify(msg));
            console.log('✅ [ADMIN-CHAT] Message sent successfully');
            input.value = '';
            
            
            
        } catch(e) {
            console.error('❌ [ADMIN-CHAT] Send error:', e);
            alert('Ошибка отправки: ' + e.message);
        }
    }
    
    // ===== ОТПРАВКА ГОЛОСОВОГО =====
    if (recordedBlob) {
        console.log('📤 [ADMIN-CHAT] Sending voice');
        var reader = new FileReader();
        reader.onload = function() {
            try {
                adminChatWs.send(JSON.stringify({ 
                    type: 'voice', 
                    data: reader.result, 
                    duration: '00:05', 
                    sender: 'admin', 
                    sender_name: 'Администратор' 
                }));
                console.log('✅ [ADMIN-CHAT] Voice sent');
            } catch(e) {
                console.error('❌ [ADMIN-CHAT] Voice error:', e);
            }
        };
        reader.readAsDataURL(recordedBlob);
        recordedBlob = null;
        document.getElementById('previewContainer').innerHTML = '';
        document.getElementById('previewContainer').style.display = 'none';
        document.getElementById('recordingIndicator').style.display = 'none';
        document.getElementById('voiceBtn').innerHTML = '🎤';
    }
    
    // ===== ОТПРАВКА ФАЙЛА - ИСПРАВЛЕНО =====
    if (attachedFile) {
        // Проверяем, что файл существует
        if (!attachedFile || !attachedFile.name) {
            console.error('❌ [ADMIN-CHAT] attachedFile is null or invalid');
            attachedFile = null;
            document.getElementById('previewContainer').innerHTML = '';
            document.getElementById('previewContainer').style.display = 'none';
            return;
        }
        
        console.log('📤 [ADMIN-CHAT] Sending file:', attachedFile.name);
        
        // Сохраняем файл в локальную переменную
        var fileToSend = attachedFile;
        
        var reader = new FileReader();
        reader.onload = function() {
            try {
                adminChatWs.send(JSON.stringify({ 
                    type: 'file', 
                    name: fileToSend.name, 
                    data: reader.result, 
                    mime: fileToSend.type, 
                    sender: 'admin', 
                    sender_name: 'Администратор' 
                }));
                console.log('✅ [ADMIN-CHAT] File sent');
            } catch(e) {
                console.error('❌ [ADMIN-CHAT] File error:', e);
            }
        };
        reader.onerror = function(e) {
            console.error('❌ [ADMIN-CHAT] File read error:', e);
            alert('Ошибка чтения файла');
        };
        reader.readAsDataURL(fileToSend);
        
        // Очищаем после отправки
        attachedFile = null;
        document.getElementById('previewContainer').innerHTML = '';
        document.getElementById('previewContainer').style.display = 'none';
    }
    
    input.focus();
}

        function handleFileSelect(input) {
            const file = input.files[0];
            if (!file) return;
            if (file.size > 20 * 1024 * 1024) { alert('Файл слишком большой (макс 20MB)'); input.value = ''; return; }
            attachedFile = file;
            const preview = document.getElementById('previewContainer');
            preview.innerHTML = `<div class="file-preview"><span>📎 ${file.name} (${(file.size/1024).toFixed(1)} KB)</span><button onclick="cancelFile()" class="text-red-400 hover:text-red-300 ml-2">✕</button><button onclick="sendChatMessage()" class="bg-green-500/20 hover:bg-green-500/30 px-3 py-1 rounded ml-2">📤 Отправить</button></div>`;
            preview.style.display = 'block';
            input.value = '';
        }
        function cancelFile() { attachedFile = null; document.getElementById('previewContainer').innerHTML = ''; document.getElementById('previewContainer').style.display = 'none'; }

        async function startVoiceRecording() {
            const voiceBtn = document.getElementById('voiceBtn');
            const indicator = document.getElementById('recordingIndicator');
            if (isRecording) {
                if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
                isRecording = false;
                indicator.style.display = 'none';
                voiceBtn.innerHTML = '🎤';
                return;
            }
            if (!navigator.mediaDevices) { alert('Ваш браузер не поддерживает запись'); return; }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
                mediaRecorder.onstop = () => {
                    if (audioChunks.length === 0) return;
                    recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const preview = document.getElementById('previewContainer');
                    preview.innerHTML = `<div class="voice-preview"><audio controls src="${URL.createObjectURL(recordedBlob)}" class="h-8"></audio><button onclick="cancelVoice()" class="text-red-400 hover:text-red-300 ml-2">✕</button><button onclick="sendChatMessage()" class="bg-green-500/20 hover:bg-green-500/30 px-3 py-1 rounded ml-2">📤 Отправить</button></div>`;
                    preview.style.display = 'block';
                    stream.getTracks().forEach(t => t.stop());
                };
                mediaRecorder.start();
                isRecording = true;
                indicator.style.display = 'flex';
                voiceBtn.innerHTML = '⏹️';
            } catch(e) { alert('Не удалось получить доступ к микрофону'); }
        }
        function cancelVoice() { recordedBlob = null; document.getElementById('previewContainer').innerHTML = ''; document.getElementById('previewContainer').style.display = 'none'; }

        function initEmojiPicker() {
            const container = document.querySelector('#emojiPicker .emoji-list');
            if (!container || container.children.length > 0) return;
            const emojis = ['😀','😃','😄','😁','😆','😅','😂','🤣','😊','😇','🙂','🙃','😉','😌','😍','🥰','😘','😗','😙','😚','😋','😛','😜','🤪','😝','🎶','🤔','🤨','😐','😑','😶','🙄','😏','😣','😥','😮','🤐','😯','😪','😫','😴','😌','😒','😓','😔','😕','🙃','🤑','😲','☹️','🙁','😖','😞','😟','😤','😢','😭','😦','😧','😨','😩','🤯','😬','😰','😱','🥵','🥶','😳','🤪','😵','😡','😠','🤬','😷','🤒','🤕','🤢','🤮','🤧','🥴','🤐','🥳','🥸','🥺','😥','😪','🤤','😴','👍','👎','❤️','💔','🔥','⭐','🎉','✨'];
            emojis.forEach(e => { const span = document.createElement('span'); span.textContent = e; span.className = 'emoji-item'; span.onclick = () => { const input = document.getElementById('chatInput'); const pos = input.selectionStart; input.value = input.value.substring(0, pos) + e + input.value.substring(pos); input.focus(); document.getElementById('emojiPicker').style.display = 'none'; }; container.appendChild(span); });
        }
        document.getElementById('emojiBtn')?.addEventListener('click', () => { const p = document.getElementById('emojiPicker'); if (p.style.display === 'block') p.style.display = 'none'; else { initEmojiPicker(); p.style.display = 'block'; } });
        document.getElementById('fileBtn')?.addEventListener('click', () => document.getElementById('fileInput').click());
        document.getElementById('voiceBtn')?.addEventListener('click', startVoiceRecording);
        document.addEventListener('click', (e) => { const p = document.getElementById('emojiPicker'); if (p && !p.contains(e.target) && e.target !== document.getElementById('emojiBtn')) p.style.display = 'none'; });

        checkAuth();
        
                // ============================================================
        // РАСШИРЕННАЯ МОДЕРАЦИЯ БАНА - НОВЫЕ ФУНКЦИИ
        // ============================================================

        // Переход в раздел модерации с указанным ID
        function goToAdvancedBan(userId) {
    // Переключаем на вкладку модерации
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.tab-btn[data-tab="moderation"]').classList.add('active');
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.getElementById('moderationTab').classList.add('active');
    
    // Устанавливаем ID в поле расширенного бана
    const input = document.getElementById('advancedBanUserId');
    if (input) {
        input.value = userId;
    }
    
    // Автоматически загружаем информацию о пользователе
    setTimeout(() => loadAdvancedBanUser(), 500);
}

// Загрузка информации о пользователе для расширенного бана
async function loadAdvancedBanUser() {
    const input = document.getElementById('advancedBanUserId');
    if (!input) {
        console.error('❌ advancedBanUserId not found');
        return;
    }
    
    const userId = input.value;
    if (!userId) {
        alert('Введите ID пользователя');
        return;
    }
    
    try {
        // ===== ИСПОЛЬЗУЕМ НОВЫЙ ЭНДПОИНТ =====
        const res = await fetch(`${BACKEND_URL}/admin/user-info/${userId}`, { headers: getAuthHeaders() });
        const infoDiv = document.getElementById('advancedBanUserInfo');
        if (!infoDiv) return;
        
        if (res.ok) {
            const u = await res.json();
            
            document.getElementById('advBanUserId').textContent = u.user_id || userId;
            
            const statusEl = document.getElementById('advBanUserStatus');
            if (u.in_online) {
                statusEl.textContent = u.status || 'в сети';
                statusEl.className = u.status === 'connected' ? 'text-green-400' : (u.status === 'searching' ? 'text-yellow-400' : 'text-blue-400');
            } else {
                statusEl.textContent = 'Не в сети';
                statusEl.className = 'text-gray-400';
            }
            
            document.getElementById('advBanUserIp').textContent = u.ip_address || 'неизвестен';
            document.getElementById('advBanUserBrowser').textContent = u.browser_id || 'неизвестен';
            document.getElementById('advBanUserFingerprint').textContent = u.fingerprint || 'не обнаружен';
            
            // ===== Показываем, если уже забанен =====
            const alreadyBanned = [];
            if (u.is_banned_user) alreadyBanned.push('👤 по ID');
            if (u.is_banned_ip) alreadyBanned.push('🌐 по IP');
            if (u.is_banned_browser) alreadyBanned.push('🖥️ по Browser');
            if (u.is_banned_fingerprint) alreadyBanned.push('🆔 по Fingerprint');
            if (u.is_banned_subnet) alreadyBanned.push('🔗 по подсети');
            
            let extraHtml = '';
            if (alreadyBanned.length > 0) {
                extraHtml = `<div class="mt-2 p-2 bg-red-500/20 border border-red-500/40 rounded text-xs text-red-300">
                    ⚠️ Уже забанен: ${alreadyBanned.join(', ')}
                </div>`;
            }
            
            infoDiv.innerHTML = `
                <div class="bg-gray-800/30 p-3 rounded-xl text-sm">
                    <div class="flex justify-between"><span class="text-gray-400">ID:</span><span id="advBanUserId" class="font-mono"></span></div>
                    <div class="flex justify-between"><span class="text-gray-400">Статус:</span><span id="advBanUserStatus"></span></div>
                    <div class="flex justify-between"><span class="text-gray-400">IP:</span><span id="advBanUserIp" class="font-mono text-xs"></span></div>
                    <div class="flex justify-between"><span class="text-gray-400">Browser ID:</span><span id="advBanUserBrowser" class="font-mono text-xs"></span></div>
                    <div class="flex justify-between"><span class="text-gray-400">Fingerprint:</span><span id="advBanUserFingerprint" class="font-mono text-xs"></span></div>
                    ${extraHtml}
                </div>
            `;
            
            // Восстанавливаем значения в новые span-ы
            document.getElementById('advBanUserId').textContent = u.user_id || userId;
            statusEl.textContent = u.in_online ? (u.status || 'в сети') : 'Не в сети';
            statusEl.className = u.in_online ? (u.status === 'connected' ? 'text-green-400' : (u.status === 'searching' ? 'text-yellow-400' : 'text-blue-400')) : 'text-gray-400';
            document.getElementById('advBanUserIp').textContent = u.ip_address || 'неизвестен';
            document.getElementById('advBanUserBrowser').textContent = u.browser_id || 'неизвестен';
            document.getElementById('advBanUserFingerprint').textContent = u.fingerprint || 'не обнаружен';
            
            infoDiv.classList.remove('hidden');
        } else {
            // Fallback на старый способ, если эндпоинт упал
            console.warn('⚠️ [ADMIN] user-info endpoint failed, using fallback');
            await loadAdvancedBanUserFallback(userId);
        }
    } catch(e) {
        console.error('Ошибка загрузки пользователя:', e);
        await loadAdvancedBanUserFallback(userId);
    }
}

// ===== FALLBACK (старый способ через /admin/connections) =====
async function loadAdvancedBanUserFallback(userId) {
    try {
        const res = await fetch(`${BACKEND_URL}/admin/connections`, { headers: getAuthHeaders() });
        if (res.ok) {
            const conns = await res.json();
            const user = conns.find(c => c.user_id === userId);
            
            const infoDiv = document.getElementById('advancedBanUserInfo');
            if (!infoDiv) return;
            
            if (user) {
                infoDiv.innerHTML = `
                    <div class="bg-gray-800/30 p-3 rounded-xl text-sm">
                        <div class="flex justify-between"><span class="text-gray-400">ID:</span><span class="font-mono">${user.user_id || userId}</span></div>
                        <div class="flex justify-between"><span class="text-gray-400">Статус:</span><span class="text-green-400">${user.status || 'в сети'}</span></div>
                        <div class="flex justify-between"><span class="text-gray-400">IP:</span><span class="font-mono text-xs">${user.ip_address || 'неизвестен'}</span></div>
                        <div class="flex justify-between"><span class="text-gray-400">Browser ID:</span><span class="font-mono text-xs">${user.browser_id || 'неизвестен'}</span></div>
                        <div class="flex justify-between"><span class="text-gray-400">Fingerprint:</span><span class="font-mono text-xs">${user.fingerprint || 'не обнаружен'}</span></div>
                    </div>
                `;
            } else {
                infoDiv.innerHTML = `
                    <div class="bg-gray-800/30 p-3 rounded-xl text-sm">
                        <div class="flex justify-between"><span class="text-gray-400">ID:</span><span class="font-mono">${userId}</span></div>
                        <div class="flex justify-between"><span class="text-gray-400">Статус:</span><span class="text-gray-400">Не в сети</span></div>
                        <div class="flex justify-between"><span class="text-gray-400">IP:</span><span class="font-mono text-xs">неизвестен</span></div>
                        <div class="flex justify-between"><span class="text-gray-400">Browser ID:</span><span class="font-mono text-xs">неизвестен</span></div>
                        <div class="flex justify-between"><span class="text-gray-400">Fingerprint:</span><span class="font-mono text-xs">не обнаружен</span></div>
                    </div>
                `;
            }
            infoDiv.classList.remove('hidden');
        }
    } catch(e) {
        console.error('Fallback error:', e);
    }
}

// Выполнение расширенного бана
async function executeAdvancedBan() {
    const input = document.getElementById('advancedBanUserId');
    if (!input) {
        alert('Ошибка: элемент advancedBanUserId не найден');
        return;
    }
    
    const userId = input.value;
    if (!userId) {
        alert('Введите ID пользователя для бана');
        return;
    }
    
    // Собираем ТОЛЬКО выбранные типы бана
    const banTypeUser = document.getElementById('banTypeUser');
    const banTypeIp = document.getElementById('banTypeIp');
    const banTypeBrowser = document.getElementById('banTypeBrowser');
    const banTypeFingerprint = document.getElementById('banTypeFingerprint');
    const banTypeSubnet = document.getElementById('banTypeSubnet');
    const banTypeAll = document.getElementById('banTypeAll');
    
    const banTypes = {
        user: banTypeUser ? banTypeUser.checked : false,
        ip: banTypeIp ? banTypeIp.checked : false,
        browser: banTypeBrowser ? banTypeBrowser.checked : false,
        fingerprint: banTypeFingerprint ? banTypeFingerprint.checked : false,
        subnet: banTypeSubnet ? banTypeSubnet.checked : false,
        all: banTypeAll ? banTypeAll.checked : false
    };
    
    // Если выбран "Максимальный бан" - включаем всё
    if (banTypes.all) {
        banTypes.user = true;
        banTypes.ip = true;
        banTypes.browser = true;
        banTypes.fingerprint = true;
        banTypes.subnet = true;
    }
    
    // Проверяем, что выбран хотя бы один тип
    if (!banTypes.user && !banTypes.ip && !banTypes.browser && !banTypes.fingerprint && !banTypes.subnet) {
        alert('Выберите хотя бы один тип блокировки');
        return;
    }
    
    // Получаем длительность
    const durationInput = document.getElementById('advancedBanDuration');
    const unitSelect = document.getElementById('advancedBanUnit');
    
    let duration = 24;
    if (durationInput) {
        duration = parseInt(durationInput.value) || 24;
    }
    
    let unit = 'hours';
    if (unitSelect) {
        unit = unitSelect.value;
    }
    
    let durationText = '';
    let durationHours = 0;
    
    switch(unit) {
        case 'hours':
            durationHours = duration;
            durationText = `${duration} час(ов)`;
            break;
        case 'days':
            durationHours = duration * 24;
            durationText = `${duration} день(ей)`;
            break;
        case 'weeks':
            durationHours = duration * 24 * 7;
            durationText = `${duration} недель(и)`;
            break;
        case 'months':
            durationHours = duration * 24 * 30;
            durationText = `${duration} месяц(ев)`;
            break;
        case 'permanent':
            durationHours = 0;
            durationText = 'Навсегда';
            break;
    }
    
    // Получаем причину
    const reasonSelect = document.getElementById('advancedBanReason');
    const customReasonInput = document.getElementById('advancedBanCustomReason');
    
    let reason = 'spam';
    if (reasonSelect) {
        reason = reasonSelect.value;
    }
    
    if (customReasonInput && customReasonInput.value) {
        reason = customReasonInput.value;
    }
    
    // Формируем данные для отправки
    const banData = {
        user_id: userId,
        reason: reason,
        duration_hours: durationHours,
        ban_types: banTypes,
        duration_text: durationText
    };
    
    console.log('📤 [ADMIN] Sending ban data:', banData);
    
    // Подтверждение
    const banTypeNames = [];
    if (banTypes.user) banTypeNames.push('👤 ID');
    if (banTypes.ip) banTypeNames.push('🌐 IP');
    if (banTypes.browser) banTypeNames.push('🖥️ Browser');
    if (banTypes.fingerprint) banTypeNames.push('🆔 Fingerprint');
    if (banTypes.subnet) banTypeNames.push('🔗 Подсеть');
    if (banTypes.all) banTypeNames.push('💀 ВСЁ!');
    
    if (!confirm(`⚠️ ВНИМАНИЕ! Вы собираетесь забанить пользователя ${userId}\n\nТипы бана: ${banTypeNames.join(', ')}\nДлительность: ${durationText}\nПричина: ${reason}\n\nПродолжить?`)) {
        return;
    }
    
    // Отправляем запрос
    try {
        const res = await fetch(`${BACKEND_URL}/admin/advanced-ban`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify(banData)
        });
        
        const result = await res.json();
        console.log('📥 [ADMIN] Ban result:', result);
        
        const resultDiv = document.getElementById('advancedBanResult');
        if (!resultDiv) return;
        
        resultDiv.classList.remove('hidden');
        
        if (res.ok && result.status === 'success') {
            resultDiv.className = 'mt-3 p-4 bg-green-500/20 border border-green-500/50 rounded-xl';
            let resultsHtml = result.results ? result.results.map(r => `<div class="text-sm text-green-300">${r}</div>`).join('') : '';
            resultDiv.innerHTML = `
                <div class="flex items-start gap-3">
                    <span class="text-2xl">✅</span>
                    <div>
                        <div class="font-bold text-green-400">БАН ВЫПОЛНЕН УСПЕШНО!</div>
                        <div class="text-sm text-gray-300 mt-1">Пользователь: <span class="font-mono">${userId}</span></div>
                        <div class="text-sm text-gray-300">Типы бана: ${banTypeNames.join(', ')}</div>
                        <div class="text-sm text-gray-300">Длительность: ${durationText}</div>
                        <div class="text-sm text-gray-300">Причина: ${reason}</div>
                        ${resultsHtml}
                        ${result.message ? `<div class="text-sm text-green-300 mt-2">${result.message}</div>` : ''}
                    </div>
                </div>
            `;
            
            // Обновляем списки
            if (typeof loadBlockedUsers === 'function') loadBlockedUsers();
            if (typeof loadStats === 'function') loadStats();
            if (typeof loadConnections === 'function') loadConnections();
        } else {
            resultDiv.className = 'mt-3 p-4 bg-red-500/20 border border-red-500/50 rounded-xl';
            resultDiv.innerHTML = `
                <div class="flex items-start gap-3">
                    <span class="text-2xl">❌</span>
                    <div>
                        <div class="font-bold text-red-400">ОШИБКА БАНА</div>
                        <div class="text-sm text-gray-300">${result.detail || result.message || 'Неизвестная ошибка'}</div>
                        ${result.results ? result.results.map(r => `<div class="text-sm text-yellow-300">${r}</div>`).join('') : ''}
                    </div>
                </div>
            `;
        }
    } catch(e) {
        console.error('❌ [ADMIN] Ban error:', e);
        const resultDiv = document.getElementById('advancedBanResult');
        if (!resultDiv) return;
        
        resultDiv.classList.remove('hidden');
        resultDiv.className = 'mt-3 p-4 bg-red-500/20 border border-red-500/50 rounded-xl';
        resultDiv.innerHTML = `
            <div class="flex items-start gap-3">
                <span class="text-2xl">❌</span>
                <div>
                    <div class="font-bold text-red-400">ОШИБКА</div>
                    <div class="text-sm text-gray-300">${e.message || 'Ошибка соединения'}</div>
                </div>
            </div>
        `;
    }
}

// Очистка формы расширенного бана
function clearAdvancedBan() {
    const input = document.getElementById('advancedBanUserId');
    if (input) input.value = '';
    
    const infoDiv = document.getElementById('advancedBanUserInfo');
    if (infoDiv) infoDiv.classList.add('hidden');
    
    const resultDiv = document.getElementById('advancedBanResult');
    if (resultDiv) {
        resultDiv.classList.add('hidden');
        resultDiv.innerHTML = '';
    }
    
    // Сбрасываем чекбоксы - ВСЕ СНЯТЫ
    const userCheckbox = document.getElementById('banTypeUser');
    const ipCheckbox = document.getElementById('banTypeIp');
    const browserCheckbox = document.getElementById('banTypeBrowser');
    const fpCheckbox = document.getElementById('banTypeFingerprint');
    const subnetCheckbox = document.getElementById('banTypeSubnet');
    const allCheckbox = document.getElementById('banTypeAll');
    
    if (userCheckbox) userCheckbox.checked = false;
    if (ipCheckbox) ipCheckbox.checked = false;
    if (browserCheckbox) browserCheckbox.checked = false;
    if (fpCheckbox) fpCheckbox.checked = false;
    if (subnetCheckbox) subnetCheckbox.checked = false;
    if (allCheckbox) allCheckbox.checked = false;
    
    // Разблокируем все чекбоксы
    if (userCheckbox) userCheckbox.disabled = false;
    if (ipCheckbox) ipCheckbox.disabled = false;
    if (browserCheckbox) browserCheckbox.disabled = false;
    if (fpCheckbox) fpCheckbox.disabled = false;
    if (subnetCheckbox) subnetCheckbox.disabled = false;
    
    const durationInput = document.getElementById('advancedBanDuration');
    if (durationInput) durationInput.value = 24;
    
    const unitSelect = document.getElementById('advancedBanUnit');
    if (unitSelect) unitSelect.value = 'hours';
    
    const reasonSelect = document.getElementById('advancedBanReason');
    if (reasonSelect) reasonSelect.value = 'spam';
    
    const customReasonInput = document.getElementById('advancedBanCustomReason');
    if (customReasonInput) customReasonInput.value = '';
}

// ============================================================
// ИНИЦИАЛИЗАЦИЯ РАСШИРЕННОЙ МОДЕРАЦИИ
// ============================================================

function initAdvancedBan() {
    console.log('🔒 [ADMIN] Initializing Advanced Ban...');
    
    const allCheckbox = document.getElementById('banTypeAll');
    if (allCheckbox) {
        // Удаляем старые обработчики
        const newCheckbox = allCheckbox.cloneNode(true);
        allCheckbox.parentNode.replaceChild(newCheckbox, allCheckbox);
        
        newCheckbox.addEventListener('change', function() {
            const userCb = document.getElementById('banTypeUser');
            const ipCb = document.getElementById('banTypeIp');
            const browserCb = document.getElementById('banTypeBrowser');
            const fpCb = document.getElementById('banTypeFingerprint');
            const subnetCb = document.getElementById('banTypeSubnet');
            
            if (this.checked) {
                // Включаем все и блокируем
                if (userCb) { userCb.checked = true; userCb.disabled = true; }
                if (ipCb) { ipCb.checked = true; ipCb.disabled = true; }
                if (browserCb) { browserCb.checked = true; browserCb.disabled = true; }
                if (fpCb) { fpCb.checked = true; fpCb.disabled = true; }
                if (subnetCb) { subnetCb.checked = true; subnetCb.disabled = true; }
            } else {
                // Разблокируем
                if (userCb) userCb.disabled = false;
                if (ipCb) ipCb.disabled = false;
                if (browserCb) browserCb.disabled = false;
                if (fpCb) fpCb.disabled = false;
                if (subnetCb) subnetCb.disabled = false;
            }
        });
    }
    
    console.log('✅ [ADMIN] Advanced Ban initialized');
}

// Инициализация
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAdvancedBan);
} else {
    initAdvancedBan();
}

        // ============================================================
        // ФУНКЦИИ ДЛЯ РАБОТЫ С ЧЕКБОКСАМИ (ДОБАВЛЕНО)
        // ============================================================

        // Функция для переключения чекбокса
        // Функция для переключения чекбокса
function toggleCheckbox(id) {
    const checkbox = document.getElementById(id);
    if (checkbox) {
        checkbox.checked = !checkbox.checked;
        
        if (id === 'banTypeAll') {
            handleMaxBanToggle(checkbox.checked);
        }
    }
}

function handleMaxBanToggle(checked) {
    const ids = ['banTypeUser', 'banTypeIp', 'banTypeBrowser', 'banTypeFingerprint', 'banTypeSubnet'];
    ids.forEach(id => {
        const cb = document.getElementById(id);
        if (cb) {
            if (checked) {
                cb.checked = true;
                cb.disabled = true;
            } else {
                cb.disabled = false;
            }
        }
    });
}

// ============================================================
// НАСТРОЙКА ПЕРИОДА ПОКАЗА РЕКЛАМЫ
// ============================================================

let adScheduleState = {
    id: null,
    type: null,  // 'audio' | 'video'
    preset: 'always',
    from: null,
    until: null
};

// Открыть модалку для аудио
function openAudioScheduleModal(adId) {
    openAdScheduleModal(adId, 'audio');
}

// Открыть модалку для видео
function openVideoScheduleModal(adId) {
    openAdScheduleModal(adId, 'video');
}

async function openAdScheduleModal(adId, type) {
    console.log('⚙️ [SCHEDULE] Открываю модалку:', adId, type);
    
    adScheduleState.id = adId;
    adScheduleState.type = type;
    
    // Загружаем данные записи
    try {
        const listEndpoint = type === 'audio' ? '/admin/ads/audio/list' : '/admin/ads/video/list';
        const res = await fetch(`${BACKEND_URL}${listEndpoint}`, { headers: getAuthHeaders() });
        if (res.ok) {
            const data = await res.json();
            const ad = (data.ads || []).find(a => a.id === adId);
            if (ad) {
                adScheduleState.preset = ad.period_preset || 'always';
                adScheduleState.from = ad.show_from || null;
                adScheduleState.until = ad.show_until || null;
                
                // Заполняем форму
                document.getElementById('adScheduleId').value = adId;
                document.getElementById('adScheduleType').value = type;
                document.getElementById('adScheduleTitle').textContent = `⚙️ Период показа — ${ad.name || adId.slice(0, 8)}`;
                document.getElementById('adScheduleActive').checked = ad.is_active !== false;
                
                // Устанавливаем значения datetime-local
                if (ad.show_from) {
                    document.getElementById('adScheduleFrom').value = toLocalDatetime(ad.show_from);
                } else {
                    document.getElementById('adScheduleFrom').value = '';
                }
                if (ad.show_until) {
                    document.getElementById('adScheduleUntil').value = toLocalDatetime(ad.show_until);
                } else {
                    document.getElementById('adScheduleUntil').value = '';
                }
                
                // Подсвечиваем активный preset
                highlightPreset(adScheduleState.preset);
            }
        }
    } catch (e) {
        console.error('❌ [SCHEDULE] Ошибка загрузки:', e);
    }
    
    document.getElementById('adScheduleModal').classList.remove('hidden');
}

function closeAdScheduleModal() {
    document.getElementById('adScheduleModal').classList.add('hidden');
    adScheduleState = { id: null, type: null, preset: 'always', from: null, until: null };
}

// Устанавливает пресет и автоматически заполняет даты
function setAdSchedulePreset(preset) {
    adScheduleState.preset = preset;
    highlightPreset(preset);
    
    const now = new Date();
    const fromInput = document.getElementById('adScheduleFrom');
    const untilInput = document.getElementById('adScheduleUntil');
    
    if (preset === 'always') {
        fromInput.value = '';
        untilInput.value = '';
        return;
    }
    
    if (preset === 'custom') {
        // Не трогаем поля, пользователь сам выберет
        return;
    }
    
    // Заполняем "сейчас"
    fromInput.value = toLocalDatetime(now.toISOString());
    
    // Заполняем "через N"
    const until = new Date(now);
    switch (preset) {
        case 'day':   until.setDate(until.getDate() + 1); break;
        case 'week':  until.setDate(until.getDate() + 7); break;
        case 'month': until.setMonth(until.getMonth() + 1); break;
        case 'year':  until.setFullYear(until.getFullYear() + 1); break;
    }
    untilInput.value = toLocalDatetime(until.toISOString());
}

function highlightPreset(preset) {
    document.querySelectorAll('.preset-btn').forEach(btn => {
        if (btn.dataset.preset === preset) {
            btn.classList.add('bg-blue-500/40', 'border-blue-400');
        } else {
            btn.classList.remove('bg-blue-500/40', 'border-blue-400');
        }
    });
}

// Преобразование ISO в формат для datetime-local
function toLocalDatetime(isoString) {
    if (!isoString) return '';
    try {
        const d = new Date(isoString);
        // Формат: YYYY-MM-DDTHH:MM
        const pad = (n) => String(n).padStart(2, '0');
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
    } catch (e) {
        return '';
    }
}

// Сохранение
async function saveAdSchedule() {
    const adId = adScheduleState.id;
    const type = adScheduleState.type;
    if (!adId || !type) {
        alert('Ошибка: не выбран объект');
        return;
    }
    
    const preset = adScheduleState.preset;
    const from = document.getElementById('adScheduleFrom').value;
    const until = document.getElementById('adScheduleUntil').value;
    const isActive = document.getElementById('adScheduleActive').checked;
    
    // Валидация
    if (preset === 'custom') {
        if (!from || !until) {
            alert('Для "своего" периода укажите обе даты');
            return;
        }
        if (new Date(from) >= new Date(until)) {
            alert('Дата "По какое" должна быть позже "С какого"');
            return;
        }
    }
    
    // Если preset не always и не custom — отправляем как есть
    let showFrom = null;
    let showUntil = null;
    
    if (preset === 'custom') {
        showFrom = from ? new Date(from).toISOString() : null;
        showUntil = until ? new Date(until).toISOString() : null;
    } else if (preset !== 'always') {
        // Автопресет — вычисляем на сервере
        showFrom = from ? new Date(from).toISOString() : new Date().toISOString();
        showUntil = until ? new Date(until).toISOString() : null;
    }
    
    const endpoint = type === 'audio'
        ? `/admin/ads/audio/${adId}/schedule`
        : `/admin/ads/video/${adId}/schedule`;
    
    console.log('💾 [SCHEDULE] Сохраняю:', { preset, showFrom, showUntil, isActive });
    
    try {
        const res = await fetch(`${BACKEND_URL}${endpoint}`, {
            method: 'PUT',
            headers: getAuthHeaders(),
            body: JSON.stringify({
                period_preset: preset,
                show_from: showFrom,
                show_until: showUntil,
                is_active: isActive
            })
        });
        
        const result = await res.json();
        console.log('📥 [SCHEDULE] Ответ:', res.status, result);
        
        if (res.ok) {
            alert('✅ Период показа сохранён');
            closeAdScheduleModal();
            // Обновляем список
            if (type === 'audio') loadAudioAds();
            else loadVideoAds();
        } else {
            alert('❌ Ошибка: ' + (result.detail || 'Unknown error'));
        }
    } catch (e) {
        console.error('❌ [SCHEDULE] Ошибка:', e);
        alert('❌ Ошибка сети: ' + e.message);
    }
}
    

    </script>
</body>
</html>
"""



# ============================================================
# ПОДКЛЮЧЕНИЕ РОУТЕРОВ
# ============================================================
app.include_router(api_router)
app.include_router(admin_router)
app.include_router(moderation_router)

# ============================================================
# ЗАПУСК СЕРВЕРА
# ============================================================


# ============================================================
# ЗАГРУЗКА ОБНОВЛЕНИЙ ФРОНТЕНДА
# ============================================================
# Путь к папке build вашего фронтенда
FRONTEND_BUILD_DIR = Path(r"C:\Users\NACHISTO\Desktop\anon_chat\frontend\build")
FRONTEND_BACKUP_DIR = Path(r"C:\Users\NACHISTO\Desktop\anon_chat\frontend\build_backup")
FRONTEND_VERSION_FILE = FRONTEND_BUILD_DIR / "version.txt"


@app.post("/admin/upload-frontend")
async def upload_frontend_update(
    file: UploadFile = File(...),
    version: str = Form(""),
    moderator: dict = Depends(authenticate_admin)
):
    """
    Загружает ZIP с фронтендом и версией.
    version — строка версии, введённая в админке.
    """
    logger.info("=" * 60)
    logger.info(f"[OTA] Upload started by {moderator['username']}")
    logger.info(f"[OTA] Version: {version or '(auto)'}")
    logger.info("=" * 60)

    temp_zip = Path(__file__).parent / "_upload.zip"
    temp_extract = Path(__file__).parent / "_extract"

    try:
        # ===== 1. Читаем файл =====
        content = await file.read()
        logger.info(f"[OTA] Received {len(content)} bytes")

        if len(content) < 100:
            raise HTTPException(400, "File too small (not a valid ZIP)")

        # ===== 2. Проверяем версию =====
        version_str = (version or "").strip()
        if version_str:
            # Валидация — только буквы/цифры/точки/дефисы/подчёркивания, до 50 символов
            import re
            if not re.match(r'^[a-zA-Z0-9.\-_а-яА-Я ]{1,50}$', version_str):
                raise HTTPException(400, "Invalid version format (only letters, digits, .-_)")
            logger.info(f"[OTA] User provided version: {version_str}")
        else:
            version_str = datetime.utcnow().strftime("%Y.%m.%d-%H%M%S")
            logger.info(f"[OTA] Auto version: {version_str}")

        # ===== 3. Сохраняем ZIP =====
        with open(temp_zip, "wb") as f:
            f.write(content)

        # ===== 4. Проверяем ZIP =====
        if not zipfile.is_zipfile(temp_zip):
            temp_zip.unlink(missing_ok=True)
            raise HTTPException(400, "Uploaded file is not a valid ZIP")

        # ===== 5. Бэкап текущего build =====
        if FRONTEND_BUILD_DIR.exists() and any(FRONTEND_BUILD_DIR.iterdir()):
            if FRONTEND_BACKUP_DIR.exists():
                shutil.rmtree(FRONTEND_BACKUP_DIR)
            shutil.copytree(FRONTEND_BUILD_DIR, FRONTEND_BACKUP_DIR)
            logger.info("[OTA] Backup created")

        # ===== 6. Распаковка =====
        if temp_extract.exists():
            shutil.rmtree(temp_extract)
        temp_extract.mkdir()

        with zipfile.ZipFile(temp_zip, "r") as z:
            z.extractall(temp_extract)

        # ===== 7. Ищем index.html =====
        index_candidates = list(temp_extract.rglob("index.html"))
        if not index_candidates:
            shutil.rmtree(temp_extract, ignore_errors=True)
            temp_zip.unlink(missing_ok=True)
            raise HTTPException(400, "ZIP does not contain index.html")

        index_file = sorted(index_candidates, key=lambda p: len(p.parts))[0]
        source_root = index_file.parent
        logger.info(f"[OTA] Found index.html at: {source_root}")

        # ===== 8. Заменяем build =====
        if FRONTEND_BUILD_DIR.exists():
            shutil.rmtree(FRONTEND_BUILD_DIR)
        shutil.move(str(source_root), str(FRONTEND_BUILD_DIR))

        # ===== 9. Записываем версию и метаданные =====
        version_data = {
            "version": version_str,
            "uploaded_by": moderator["username"],
            "uploaded_at": datetime.utcnow().isoformat(),
            "source": "manual" if version else "auto"
        }
        (FRONTEND_BUILD_DIR / "version.txt").write_text(
            version_str,
            encoding="utf-8"
        )
        (FRONTEND_BUILD_DIR / "version.json").write_text(
            json.dumps(version_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # ===== 10. Чистим =====
        shutil.rmtree(temp_extract, ignore_errors=True)
        temp_zip.unlink(missing_ok=True)

        # ===== 11. Уведомляем пользователей =====
        try:
            await manager.broadcast_message({
                "type": "frontend_update_available",
                "version": version_str,
                "message": f"🔄 Доступно обновление {version_str}!"
            })
        except Exception as e:
            logger.warning(f"[OTA] Broadcast error: {e}")

        file_count = len(list(FRONTEND_BUILD_DIR.rglob("*")))
        logger.info(f"[OTA] Success! Version: {version_str}, Files: {file_count}")
        logger.info("=" * 60)

        return {
            "status": "success",
            "message": f"✅ Фронтенд обновлён до {version_str}",
            "version": version_str,
            "uploaded_by": moderator["username"],
            "files_count": file_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[OTA] Error: {e}")
        import traceback
        traceback.print_exc()

        # Откат
        if FRONTEND_BACKUP_DIR.exists():
            try:
                if FRONTEND_BUILD_DIR.exists():
                    shutil.rmtree(FRONTEND_BUILD_DIR)
                shutil.copytree(FRONTEND_BACKUP_DIR, FRONTEND_BUILD_DIR)
                logger.info("[OTA] Rollback successful")
            except Exception as rollback_err:
                logger.error(f"[OTA] Rollback failed: {rollback_err}")

        shutil.rmtree(temp_extract, ignore_errors=True)
        temp_zip.unlink(missing_ok=True)

        raise HTTPException(500, f"Update failed: {str(e)}")


@app.get("/admin/frontend-info")
async def get_frontend_info(moderator: dict = Depends(authenticate_admin)):
    """Информация о текущем build с версией и метаданными"""
    if not FRONTEND_BUILD_DIR.exists():
        return {
            "installed": False,
            "message": "Папка build не найдена",
            "path": str(FRONTEND_BUILD_DIR)
        }

    index = FRONTEND_BUILD_DIR / "index.html"
    stat = index.stat() if index.exists() else None

    # ===== Версия =====
    version = None
    uploaded_by = None
    uploaded_at = None
    version_source = "unknown"

    # 1. Пробуем version.json (богатые метаданные)
    version_json = FRONTEND_BUILD_DIR / "version.json"
    if version_json.exists():
        try:
            data = json.loads(version_json.read_text(encoding="utf-8"))
            version = data.get("version")
            uploaded_by = data.get("uploaded_by")
            uploaded_at = data.get("uploaded_at")
            version_source = "version.json"
        except Exception as e:
            logger.warning(f"[FRONTEND-INFO] version.json read error: {e}")

    # 2. Fallback: version.txt
    if not version and FRONTEND_VERSION_FILE.exists():
        try:
            version = FRONTEND_VERSION_FILE.read_text(encoding="utf-8").strip()
            version_source = "version.txt"
        except Exception:
            pass

    # 3. Fallback: package.json
    if not version:
        try:
            package_json_path = FRONTEND_BUILD_DIR.parent / "package.json"
            if package_json_path.exists():
                with open(package_json_path, "r", encoding="utf-8") as f:
                    pkg = json.load(f)
                    version = pkg.get("version", "1.0.0")
                    version_source = "package.json"
        except Exception:
            pass

    # 4. Совсем fallback
    if not version:
        version = "неизвестно"
        version_source = "none"

    return {
        "installed": index.exists(),
        "version": version,
        "version_source": version_source,
        "uploaded_by": uploaded_by,
        "uploaded_at": uploaded_at,
        "path": str(FRONTEND_BUILD_DIR),
        "index_size": stat.st_size if stat else 0,
        "last_modified": stat.st_mtime if stat else None,
        "files_count": len(list(FRONTEND_BUILD_DIR.rglob("*"))),
        "has_backup": FRONTEND_BACKUP_DIR.exists()
    }


@app.get("/api/frontend-version")
async def get_frontend_version():
    """Версия фронтенда (для уведомления об обновлении)"""
    if FRONTEND_VERSION_FILE.exists():
        try:
            return {"version": FRONTEND_VERSION_FILE.read_text(encoding="utf-8").strip()}
        except Exception:
            pass
    return {"version": "0.0.1"}


@app.post("/admin/frontend-rollback")
async def rollback_frontend(moderator: dict = Depends(authenticate_admin)):
    """Откат build к бэкапу"""
    try:
        if not FRONTEND_BACKUP_DIR.exists():
            raise HTTPException(404, "Backup not found")

        if FRONTEND_BUILD_DIR.exists():
            shutil.rmtree(FRONTEND_BUILD_DIR)
        shutil.copytree(FRONTEND_BACKUP_DIR, FRONTEND_BUILD_DIR)

        logger.info(f"[OTA] Rollback successful by {moderator['username']}")
        return {"status": "success", "message": "✅ Откат выполнен"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Rollback failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.103", port=8000, ws_max_size=20 * 1024 * 1024, reload=True)