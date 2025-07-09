from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# User states
class UserStatus(str, Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, dict] = {}
        self.waiting_queue: List[str] = []
        self.active_rooms: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_sessions[user_id] = {
            "user_id": user_id,
            "status": UserStatus.IDLE,
            "room_id": None,
            "partner_id": None,
            "connected_at": datetime.utcnow()
        }
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connected",
            "user_id": user_id,
            "message": "Connected to voice chat roulette"
        }, user_id)

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            # Remove from waiting queue if present
            if user_id in self.waiting_queue:
                self.waiting_queue.remove(user_id)
            
            # Handle disconnection from active room
            if user_id in self.user_sessions:
                session = self.user_sessions[user_id]
                if session.get("room_id") and session.get("partner_id"):
                    asyncio.create_task(self.handle_partner_disconnect(user_id, session["partner_id"]))
            
            # Clean up
            del self.active_connections[user_id]
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]

    async def handle_partner_disconnect(self, disconnected_user_id: str, partner_id: str):
        if partner_id in self.user_sessions:
            partner_session = self.user_sessions[partner_id]
            partner_session["status"] = UserStatus.IDLE
            partner_session["room_id"] = None
            partner_session["partner_id"] = None
            
            await self.send_personal_message({
                "type": "partner_disconnected",
                "message": "Your partner has disconnected"
            }, partner_id)

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(json.dumps(message))

    async def send_to_room(self, message: dict, room_id: str, exclude_user: str = None):
        if room_id in self.active_rooms:
            room = self.active_rooms[room_id]
            for participant_id in room["participants"]:
                if participant_id != exclude_user:
                    await self.send_personal_message(message, participant_id)

    async def start_search(self, user_id: str):
        if user_id not in self.user_sessions:
            return False
        
        session = self.user_sessions[user_id]
        session["status"] = UserStatus.SEARCHING
        
        # Check if there's someone waiting
        if self.waiting_queue:
            partner_id = self.waiting_queue.pop(0)
            await self.create_room(user_id, partner_id)
        else:
            # Add to waiting queue
            self.waiting_queue.append(user_id)
            await self.send_personal_message({
                "type": "searching",
                "message": "Searching for a chat partner..."
            }, user_id)
        
        return True

    async def create_room(self, user1_id: str, user2_id: str):
        room_id = str(uuid.uuid4())
        
        # Update user sessions
        self.user_sessions[user1_id]["status"] = UserStatus.CONNECTED
        self.user_sessions[user1_id]["room_id"] = room_id
        self.user_sessions[user1_id]["partner_id"] = user2_id
        
        self.user_sessions[user2_id]["status"] = UserStatus.CONNECTED
        self.user_sessions[user2_id]["room_id"] = room_id
        self.user_sessions[user2_id]["partner_id"] = user1_id
        
        # Create room
        self.active_rooms[room_id] = {
            "room_id": room_id,
            "participants": [user1_id, user2_id],
            "created_at": datetime.utcnow()
        }
        
        # Notify both users
        await self.send_personal_message({
            "type": "match_found",
            "room_id": room_id,
            "partner_id": user2_id,
            "message": "Match found! You can start voice chat."
        }, user1_id)
        
        await self.send_personal_message({
            "type": "match_found",
            "room_id": room_id,
            "partner_id": user1_id,
            "message": "Match found! You can start voice chat."
        }, user2_id)

    async def stop_search(self, user_id: str):
        if user_id in self.waiting_queue:
            self.waiting_queue.remove(user_id)
        
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            session["status"] = UserStatus.IDLE
            
            await self.send_personal_message({
                "type": "search_stopped",
                "message": "Search stopped"
            }, user_id)

    async def next_partner(self, user_id: str):
        if user_id not in self.user_sessions:
            return
        
        session = self.user_sessions[user_id]
        partner_id = session.get("partner_id")
        room_id = session.get("room_id")
        
        # Disconnect current partner
        if partner_id:
            await self.handle_partner_disconnect(user_id, partner_id)
        
        # Clean up room
        if room_id and room_id in self.active_rooms:
            del self.active_rooms[room_id]
        
        # Reset session
        session["status"] = UserStatus.IDLE
        session["room_id"] = None
        session["partner_id"] = None
        
        # Start new search
        await self.start_search(user_id)

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

# Global connection manager
manager = ConnectionManager()

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_search":
                await manager.start_search(user_id)
            elif message["type"] == "stop_search":
                await manager.stop_search(user_id)
            elif message["type"] == "next_partner":
                await manager.next_partner(user_id)
            elif message["type"] == "webrtc_signal":
                await manager.handle_webrtc_signal(user_id, message["signal"])
            elif message["type"] == "ping":
                await manager.send_personal_message({"type": "pong"}, user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Voice Chat Roulette API"}

@api_router.get("/stats")
async def get_stats():
    return {
        "active_connections": len(manager.active_connections),
        "waiting_queue": len(manager.waiting_queue),
        "active_rooms": len(manager.active_rooms)
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()