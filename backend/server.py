from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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
from datetime import datetime
from enum import Enum

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

app = FastAPI(lifespan=lifespan)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

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

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, dict] = {}
        self.search_queues: Dict[str, List[str]] = {
            'all': [], 'dating': [], 'friends': [], 'fun': [], 'advice': [], 'games': []
        }
        self.active_rooms: Dict[str, dict] = {}
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        async with self.lock:
            self.active_connections[user_id] = websocket
            self.user_sessions[user_id] = {
                "user_id": user_id,
                "status": UserStatus.IDLE,
                "room_id": None,
                "partner_id": None,
                "search_params": None,
                "connected_at": datetime.utcnow()
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
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
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
                "created_at": datetime.utcnow()
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

    def get_online_stats(self) -> dict:
        """Get current online statistics"""
        return {
            "online_users": len(self.active_connections),
            "searching_users": sum(len(queue) for queue in self.search_queues.values()),
            "active_rooms": len(self.active_rooms),
            "queues_by_category": {
                category: len(queue) for category, queue in self.search_queues.items()
            }
        }

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
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
                await manager.handle_disconnect_request(
                    user_id=user_id,
                    room_id=message["room_id"],
                    partner_id=message["partner_id"]
                )
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
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)

@app.get("/")
async def read_index():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Chat Roulette</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .stats { background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
            .stat-item { margin: 5px 0; }
            button { padding: 10px; margin: 5px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            #messages { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h2>Voice Chat Roulette</h2>
        <div id="stats" class="stats"></div>
        <button onclick="connect()">Connect</button>
        <button onclick="startSearch()">Start Search</button>
        <button onclick="stopSearch()">Stop Search</button>
        <div id="messages"></div>
        <script>
            let socket;
            const userId = 'user-' + Math.random().toString(36).substr(2, 9);
            
            function connect() {
                socket = new WebSocket('ws://' + window.location.host + '/ws/' + userId);
                socket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    addMessage('Received: ' + JSON.stringify(data));
                    
                    if (data.type === 'online_stats') {
                        updateStats(data.online_stats);
                    }
                };
            }
            
            function startSearch() {
                if (socket) {
                    const searchParams = {
                        category: 'all',
                        user_gender: 'male',
                        search_gender: 'any',
                        topics: []
                    };
                    socket.send(JSON.stringify({
                        type: 'start_search',
                        search_params: searchParams
                    }));
                }
            }
            
            function stopSearch() {
                if (socket) {
                    socket.send(JSON.stringify({type: 'stop_search'}));
                }
            }
            
            function updateStats(stats) {
                const statsDiv = document.getElementById('stats');
                statsDiv.innerHTML = `
                    <h3>📊 Онлайн сейчас:</h3>
                    <div>👥 Пользователей: ${stats.online_users}</div>
                    <div>🔍 В поиске: ${stats.searching_users}</div>
                    <div>💬 Комнат: ${stats.active_rooms}</div>
                `;
            }
            
            function addMessage(text) {
                const messages = document.getElementById('messages');
                messages.innerHTML += '<div>' + new Date().toLocaleTimeString() + ' - ' + text + '</div>';
                messages.scrollTop = messages.scrollHeight;
            }
            
            // Auto connect
            connect();
        </script>
    </body>
    </html>
    """)

@api_router.get("/stats")
async def get_stats():
    return manager.get_online_stats()

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)