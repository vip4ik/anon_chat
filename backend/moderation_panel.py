from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import aiohttp
import asyncio
import logging
from datetime import datetime
from typing import Optional
import secrets
import os
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Moderation Panel with Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAIN_SERVER_URL = "http://192.168.0.103:8000"
security = HTTPBasic()

cache = {}
CACHE_TTL = 5

# ============================================================
# КЭШ
# ============================================================
def is_cache_valid(key: str) -> bool:
    if key not in cache:
        return False
    return (datetime.now() - cache[key]["timestamp"]).total_seconds() < CACHE_TTL

def update_cache(key: str, data):
    cache[key] = {"data": data, "timestamp": datetime.now()}

def invalidate_cache(*keys):
    """Сброс кэша после модифицирующих действий"""
    for k in keys:
        cache.pop(k, None)
    # Сбрасываем также кэши логов
    for k in list(cache.keys()):
        if k.startswith("logs_"):
            cache.pop(k, None)

# ============================================================
# АУТЕНТИФИКАЦИЯ МОДЕРАТОРА
# Проверяет через /moderation/check-auth на основном сервере
# (доступно модераторам, а не только админам)
# ============================================================
async def authenticate_moderator(credentials: HTTPBasicCredentials = Depends(security)):
    try:
        async with aiohttp.ClientSession() as session:
            auth = aiohttp.BasicAuth(credentials.username, credentials.password)
            async with session.get(
                f"{MAIN_SERVER_URL}/moderation/check-auth",
                auth=auth,
                timeout=10
            ) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                    except Exception:
                        data = {}
                    return {
                        "username": credentials.username,
                        "password": credentials.password,   # для проксирования
                        "role": data.get("role", "moderator"),
                        "permissions": data.get("permissions", []),
                    }
                elif response.status == 401:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                else:
                    logger.warning(f"[AUTH] check-auth → {response.status}")
                    raise HTTPException(status_code=401, detail="Invalid credentials")
    except HTTPException:
        raise
    except aiohttp.ClientConnectorError:
        raise HTTPException(status_code=503, detail="Main server is not running")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Main server timeout")
    except Exception as e:
        logger.error(f"[AUTH] error: {e}")
        raise HTTPException(status_code=503, detail=f"Connection error: {str(e)}")


# ============================================================
# ПРОКСИ К ОСНОВНОМУ СЕРВЕРУ
# Использует РЕАЛЬНЫЕ credentials модератора
# ============================================================
async def call_main_api(endpoint: str, method: str = "GET", data: dict = None, moderator: dict = None):
    if not moderator:
        logger.warning(f"[PROXY] no moderator passed to {endpoint}")
        return None
    try:
        async with aiohttp.ClientSession() as session:
            auth = aiohttp.BasicAuth(
                moderator["username"],
                moderator.get("password", "")
            )
            async with session.request(
                method=method,
                url=f"{MAIN_SERVER_URL}{endpoint}",
                auth=auth,
                json=data,
                timeout=10
            ) as response:
                if response.status == 200:
                    try:
                        return await response.json()
                    except Exception:
                        return None
                else:
                    text = await response.text()
                    logger.warning(f"[PROXY] {method} {endpoint} → {response.status}: {text[:200]}")
                    return None
    except aiohttp.ClientConnectorError:
        logger.error(f"[PROXY] {endpoint}: main server unreachable")
        return None
    except asyncio.TimeoutError:
        logger.error(f"[PROXY] {endpoint}: timeout")
        return None
    except Exception as e:
        logger.error(f"[PROXY] {endpoint}: {e}")
        return None


# ============================================================
# ЭНДПОИНТЫ
# ============================================================

@app.get("/moderation/check-auth")
async def check_auth(moderator: dict = Depends(authenticate_moderator)):
    return {
        "username": moderator["username"],
        "role": moderator.get("role", "moderator"),
        "permissions": moderator.get("permissions", [])
    }


@app.get("/moderation/stats")
async def get_stats(moderator: dict = Depends(authenticate_moderator)):
    if is_cache_valid("stats"):
        return cache["stats"]["data"]

    stats = await call_main_api("/api/stats", moderator=moderator)
    reports = await call_main_api("/moderation/stats", moderator=moderator)

    result = {
        "online_users": stats.get("online_users", 0) if stats else 0,
        "unique_users": stats.get("unique_users", 0) if stats else 0,
        "active_rooms": stats.get("active_rooms", 0) if stats else 0,
        "searching_users": stats.get("searching_users", 0) if stats else 0,
        "active_reports": reports.get("active_reports", 0) if reports else 0,
        "banned_users": reports.get("banned_users", 0) if reports else 0,
        "muted_users": reports.get("muted_users", 0) if reports else 0,
        "moderation_actions_today": reports.get("moderation_actions_today", 0) if reports else 0,
        "server_status": "online" if stats else "offline"
    }
    update_cache("stats", result)
    return result


@app.get("/moderation/connections")
async def get_connections(moderator: dict = Depends(authenticate_moderator)):
    if is_cache_valid("connections"):
        return cache["connections"]["data"]

    connections = await call_main_api("/admin/connections", moderator=moderator)
    result = connections if connections else []
    update_cache("connections", result)
    return result


@app.get("/moderation/rooms")
async def get_rooms(moderator: dict = Depends(authenticate_moderator)):
    if is_cache_valid("rooms"):
        return cache["rooms"]["data"]

    connections = await call_main_api("/admin/connections", moderator=moderator)
    if not connections:
        return {"rooms": []}

    rooms = {}
    for c in connections:
        rid = c.get("room_id")
        if rid:
            if rid not in rooms:
                rooms[rid] = {"room_id": rid, "participants": [], "participants_count": 0}
            rooms[rid]["participants"].append({
                "user_id": c.get("user_id"),
                "status": c.get("status"),
            })
            rooms[rid]["participants_count"] = len(rooms[rid]["participants"])

    result = {"rooms": list(rooms.values())}
    update_cache("rooms", result)
    return result


@app.get("/moderation/reports")
async def get_reports(
    moderator: dict = Depends(authenticate_moderator),
    resolved: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
):
    url = f"/moderation/reports?limit={limit}&offset={offset}"
    if resolved is not None:
        url += f"&resolved={str(resolved).lower()}"

    reports = await call_main_api(url, moderator=moderator)
    return reports if reports else {"reports": [], "total": 0}


@app.get("/moderation/reports/recent")
async def get_recent_reports(moderator: dict = Depends(authenticate_moderator), limit: int = 5):
    reports = await call_main_api(f"/moderation/reports/recent?limit={limit}", moderator=moderator)
    return reports if reports else []


@app.post("/moderation/reports/{report_id}/resolve")
async def resolve_report(report_id: str, request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        body = await request.json()
        result = await call_main_api(
            f"/moderation/reports/{report_id}/resolve",
            method="POST",
            data=body,
            moderator=moderator
        )
        invalidate_cache("stats", "blocked", "connections", "rooms")
        return result if result else {"status": "success"}
    except Exception as e:
        logger.error(f"[RESOLVE] {e}")
        return {"status": "error", "message": str(e)}


@app.post("/moderation/warn")
async def warn_user(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        body = await request.json()
        result = await call_main_api("/moderation/warn", method="POST", data=body, moderator=moderator)
        invalidate_cache("stats", "connections")
        return result if result else {"status": "success"}
    except Exception as e:
        logger.error(f"[WARN] {e}")
        return {"status": "error", "message": str(e)}


@app.post("/moderation/mute")
async def mute_user(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        body = await request.json()
        result = await call_main_api("/moderation/mute", method="POST", data=body, moderator=moderator)
        invalidate_cache("stats", "connections")
        return result if result else {"status": "success"}
    except Exception as e:
        logger.error(f"[MUTE] {e}")
        return {"status": "error", "message": str(e)}


@app.post("/moderation/unmute")
async def unmute_user(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        body = await request.json()
        result = await call_main_api("/moderation/unmute", method="POST", data=body, moderator=moderator)
        invalidate_cache("stats", "connections")
        return result if result else {"status": "success"}
    except Exception as e:
        logger.error(f"[UNMUTE] {e}")
        return {"status": "error", "message": str(e)}


@app.post("/moderation/kick")
async def kick_user(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        body = await request.json()
        result = await call_main_api("/moderation/kick", method="POST", data=body, moderator=moderator)
        invalidate_cache("stats", "connections", "rooms")
        return result if result else {"status": "success"}
    except Exception as e:
        logger.error(f"[KICK] {e}")
        return {"status": "error", "message": str(e)}


@app.post("/moderation/ban")
async def ban_user(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        body = await request.json()
        result = await call_main_api("/moderation/ban", method="POST", data=body, moderator=moderator)
        invalidate_cache("stats", "blocked", "connections", "rooms")
        return result if result else {"status": "success"}
    except Exception as e:
        logger.error(f"[BAN] {e}")
        return {"status": "error", "message": str(e)}


@app.post("/moderation/unban")
async def unban_user(request: Request, moderator: dict = Depends(authenticate_moderator)):
    try:
        body = await request.json()
        result = await call_main_api("/moderation/unban", method="POST", data=body, moderator=moderator)
        invalidate_cache("stats", "blocked", "connections", "rooms")
        return result if result else {"status": "success"}
    except Exception as e:
        logger.error(f"[UNBAN] {e}")
        return {"status": "error", "message": str(e)}

@app.post("/moderation/advanced-ban")
async def advanced_ban_proxy(request: Request, moderator: dict = Depends(authenticate_moderator)):
    """Прокси расширенного бана"""
    try:
        body = await request.json()
        result = await call_main_api(
            "/moderation/advanced-ban",
            method="POST",
            data=body,
            moderator=moderator
        )
        invalidate_cache("stats", "blocked", "connections", "rooms")
        return result if result else {"status": "error", "message": "Main server error"}
    except Exception as e:
        logger.error(f"[ADV-BAN] {e}")
        return {"status": "error", "message": str(e)}        


@app.get("/moderation/user/{user_id}")
async def get_user_status(user_id: str, moderator: dict = Depends(authenticate_moderator)):
    status = await call_main_api(f"/moderation/user/{user_id}", moderator=moderator)
    if status:
        return status
    return {
        "user_id": user_id,
        "is_muted": False,
        "is_banned": False,
        "warnings_count": 0,
        "active_warnings": [],
        "active_mutes": [],
        "total_reports": 0
    }


@app.get("/moderation/logs")
async def get_logs(moderator: dict = Depends(authenticate_moderator), limit: int = 100, offset: int = 0):
    cache_key = f"logs_{limit}_{offset}"
    if is_cache_valid(cache_key):
        return cache[cache_key]["data"]

    logs = await call_main_api(
        f"/moderation/logs?limit={limit}&offset={offset}",
        moderator=moderator
    )
    result = logs if logs else {"logs": [], "total": 0}
    update_cache(cache_key, result)
    return result


@app.get("/moderation/blocked")
async def get_blocked(moderator: dict = Depends(authenticate_moderator)):
    if is_cache_valid("blocked"):
        return cache["blocked"]["data"]

    blocked = await call_main_api("/admin/blocked", moderator=moderator)
    result = blocked if blocked else {
        "blocked_users": [],
        "blocked_ips": [],
        "blocked_browsers": [],
        "blocked_fingerprints": [],
        "blocked_subnets": [],
        "total": 0
    }
    update_cache("blocked", result)
    return result

@app.post("/moderation/unblock-ip/{ip:path}")
async def unblock_ip(ip: str, moderator: dict = Depends(authenticate_moderator)):
    """Прокси разблокировки IP"""
    result = await call_main_api(
        f"/admin/unblock-ip/{ip}",
        method="POST",
        moderator=moderator
    )
    invalidate_cache("blocked", "stats")
    return result if result else {"status": "error", "message": "Main server error"}


@app.post("/moderation/unblock-browser/{browser_id}")
async def unblock_browser(browser_id: str, moderator: dict = Depends(authenticate_moderator)):
    """Прокси разблокировки Browser ID"""
    result = await call_main_api(
        f"/admin/unblock-browser/{browser_id}",
        method="POST",
        moderator=moderator
    )
    invalidate_cache("blocked", "stats")
    return result if result else {"status": "error", "message": "Main server error"}


@app.post("/moderation/unblock-fingerprint/{fingerprint}")
async def unblock_fingerprint(fingerprint: str, moderator: dict = Depends(authenticate_moderator)):
    """Прокси разблокировки Fingerprint"""
    result = await call_main_api(
        f"/admin/unblock-fingerprint/{fingerprint}",
        method="POST",
        moderator=moderator
    )
    invalidate_cache("blocked", "stats")
    return result if result else {"status": "error", "message": "Main server error"}


@app.post("/moderation/unblock-subnet/{subnet:path}")
async def unblock_subnet(subnet: str, moderator: dict = Depends(authenticate_moderator)):
    """Прокси разблокировки подсети"""
    result = await call_main_api(
        f"/admin/unblock-subnet/{subnet}",
        method="POST",
        moderator=moderator
    )
    invalidate_cache("blocked", "stats")
    return result if result else {"status": "error", "message": "Main server error"}


@app.get("/moderation/chat/token")
async def get_chat_token(moderator: dict = Depends(authenticate_moderator)):
    return {"token": secrets.token_urlsafe(32), "username": moderator["username"]}


@app.get("/moderation/health")
async def moderation_health():
    """Публичная проверка, что прокси жив"""
    return {"status": "ok", "service": "moderation-proxy", "main_server": MAIN_SERVER_URL}


# ============================================================
# HTML панель
# ============================================================
HTML_CONTENT = None


@app.on_event("startup")
async def load_html():
    global HTML_CONTENT
    html_path = os.path.join(os.path.dirname(__file__), "panel.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            HTML_CONTENT = f.read()
        logger.info(f"✅ panel.html загружен ({len(HTML_CONTENT)} байт)")
    else:
        logger.warning("⚠️ panel.html не найден")
        HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Moderation Panel</title>
</head>
<body>
    <h1>Moderation Panel</h1>
    <p>panel.html not found. Please create panel.html file.</p>
</body>
</html>
        """


@app.get("/")
async def serve_panel():
    return HTMLResponse(HTML_CONTENT)


@app.get("/panel.html")
async def serve_panel_html():
    return HTMLResponse(HTML_CONTENT)


# ============================================================
# ЗАПУСК
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 ЗАПУСК ПАНЕЛИ МОДЕРАЦИИ С ЧАТОМ")
    print("=" * 60)
    print("Panel:      http://192.168.0.103:8001")
    print("Main server: http://192.168.0.103:8000")
    print("=" * 60)
    print("⚠️  Убедитесь, что основной сервер запущен на порту 8000!")
    print("⚠️  Чат работает напрямую: ws://192.168.0.103:8000/ws/chat")
    print("=" * 60)
    uvicorn.run(app, host="192.168.0.103", port=8001, reload=True)