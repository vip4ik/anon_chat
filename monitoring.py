from fastapi import FastAPI
import logging
from datetime import datetime

# Импортируем менеджер из основного файла
from main import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Chat Monitor API")

@app.get("/")
async def get_stats():
    """Возвращает статистику сервера"""
    stats = manager.get_online_stats()
    return {
        **stats,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/users")
async def get_users():
    """Получить информацию о всех пользователях"""
    return {
        "total_users": len(manager.user_data),
        "users": {
            uid: {
                "status": data.get('status'),
                "room_id": data.get('room_id'),
                "partner_id": data.get('partner_id'),
                "connected_at": data.get('connected_at').isoformat() if data.get('connected_at') else None,
                "search_category": data.get('search_params', {}).get('category') if data.get('search_params') else None
            }
            for uid, data in manager.user_data.items()
        }
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервера"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "online_users": len(manager.active_connections)
    }

@app.get("/queues")
async def get_queues():
    """Получить информацию об очередях поиска"""
    return {
        "queues": {
            category: {
                "size": len(queue),
                "users": queue
            }
            for category, queue in manager.search_queues.items()
        }
    }

@app.get("/rooms")
async def get_rooms():
    """Получить информацию о комнатах"""
    return {
        "active_rooms": len(manager.rooms),
        "rooms": {
            room_id: {
                "user1": room_data['user1'],
                "user2": room_data['user2'],
                "created_at": room_data['created_at'].isoformat()
            }
            for room_id, room_data in manager.rooms.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5001,
        log_level="info",
        use_colors=False
    )