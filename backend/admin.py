import asyncio
import aiohttp
import json
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import bcrypt
import secrets
import traceback
from pathlib import Path

# Настройки
ADMIN_DB_PATH = "admin_db.json"
MODERATION_PANEL_URL = "http://192.168.0.102:8001"  # URL панели модерации
INTERNAL_AUTH_KEY = "moderation_panel_secret_2024"  # Должен совпадать с ключом в moderation_panel.py

# Модели данных
class ModerationAction(BaseModel):
    user_id: str
    action: str  # "warn", "mute", "kick", "ban", "unmute", "unban"
    reason: str
    duration_minutes: Optional[int] = None
    moderator: str
    timestamp: str

class UserReport(BaseModel):
    reporter_id: str
    reported_user_id: str
    reason: str
    room_id: str
    evidence: Optional[str] = None
    timestamp: str
    status: str = "active"

class AdminUser(BaseModel):
    username: str
    password_hash: str
    role: str = "admin"
    permissions: List[str] = ["view", "manage_users", "moderate", "view_reports", "system_control"]

class InternalReport(BaseModel):
    report_data: dict
    auth_key: str

# Инициализация FastAPI
app = FastAPI(title="Admin Panel")
security = HTTPBasic()

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаем необходимые директории
Path("admin_templates").mkdir(exist_ok=True)
Path("admin_static").mkdir(exist_ok=True)

# Создаем базовый CSS файл для админ панели
ADMIN_CSS_CONTENT = """
/* Стили админ панели */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    background-color: #0f172a;
    color: #f1f5f9;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 1rem;
}

.header {
    background-color: #1e293b;
    border-bottom: 1px solid #334155;
    padding: 1rem 0;
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header h1 {
    margin: 0;
    font-size: 1.5rem;
    color: #e2e8f0;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.stat-card {
    background-color: #1e293b;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #334155;
    transition: transform 0.2s;
}

.stat-card:hover {
    transform: translateY(-2px);
}

.stat-card h3 {
    margin: 0 0 0.5rem 0;
    font-size: 0.875rem;
    color: #94a3b8;
}

.stat-value {
    font-size: 2rem;
    font-weight: bold;
    margin: 0;
}

.nav-tabs {
    display: flex;
    border-bottom: 1px solid #334155;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}

.tab-btn {
    padding: 1rem 1.5rem;
    background: none;
    border: none;
    color: #94a3b8;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    white-space: nowrap;
}

.tab-btn:hover {
    color: #e2e8f0;
}

.tab-btn.active {
    color: #e2e8f0;
    border-bottom-color: #3b82f6;
}

.tab-content {
    display: none;
}

.tab-content.active {
    display: block;
}

.table-container {
    background-color: #1e293b;
    border-radius: 0.5rem;
    overflow: hidden;
    margin-bottom: 2rem;
}

.table-header {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #334155;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.table {
    width: 100%;
    border-collapse: collapse;
}

.table th,
.table td {
    padding: 0.75rem 1.5rem;
    text-align: left;
    border-bottom: 1px solid #334155;
}

.table th {
    background-color: #0f172a;
    font-weight: 600;
    color: #94a3b8;
    position: sticky;
    top: 0;
}

.table tr:hover {
    background-color: #334155;
}

.btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 0.375rem;
    cursor: pointer;
    font-size: 0.875rem;
    transition: background-color 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.btn-primary {
    background-color: #3b82f6;
    color: white;
}

.btn-primary:hover {
    background-color: #2563eb;
}

.btn-danger {
    background-color: #ef4444;
    color: white;
}

.btn-danger:hover {
    background-color: #dc2626;
}

.btn-warning {
    background-color: #f59e0b;
    color: white;
}

.btn-warning:hover {
    background-color: #d97706;
}

.btn-success {
    background-color: #10b981;
    color: white;
}

.btn-success:hover {
    background-color: #059669;
}

.form-group {
    margin-bottom: 1rem;
}

.form-label {
    display: block;
    margin-bottom: 0.5rem;
    color: #e2e8f0;
    font-size: 0.875rem;
    font-weight: 500;
}

.form-input {
    width: 100%;
    padding: 0.5rem 0.75rem;
    background-color: #334155;
    border: 1px solid #475569;
    border-radius: 0.375rem;
    color: #f1f5f9;
    font-size: 0.875rem;
}

.form-input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.status-badge {
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}

.status-connected {
    background-color: #065f46;
    color: #34d399;
}

.status-searching {
    background-color: #92400e;
    color: #fbbf24;
}

.status-idle {
    background-color: #374151;
    color: #9ca3af;
}

.status-online {
    background-color: #065f46;
    color: #34d399;
}

.status-offline {
    background-color: #991b1b;
    color: #f87171;
}

.login-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.login-form {
    background-color: #1e293b;
    padding: 2rem;
    border-radius: 0.5rem;
    width: 100%;
    max-width: 400px;
    border: 1px solid #334155;
}

.card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.card-title {
    margin: 0;
    font-size: 1.125rem;
    color: #e2e8f0;
}

.report-item {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.report-item:hover {
    border-color: #475569;
}

.report-item.resolved {
    opacity: 0.7;
    border-color: #065f46;
}

.no-data {
    text-align: center;
    color: #94a3b8;
    padding: 2rem;
    font-style: italic;
}

.loading {
    text-align: center;
    color: #94a3b8;
    padding: 2rem;
}

.btn-sm {
    padding: 0.375rem 0.75rem;
    font-size: 0.75rem;
}

.connection-status {
    padding: 0.75rem;
    border-radius: 0.375rem;
    margin-bottom: 1rem;
    text-align: center;
    font-weight: 500;
}

.connection-online {
    background-color: #065f46;
    color: #34d399;
    border: 1px solid #065f46;
}

.connection-offline {
    background-color: #991b1b;
    color: #f87171;
    border: 1px solid #991b1b;
}

.w-full {
    width: 100%;
}

.grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

@media (max-width: 768px) {
    .grid-2 {
        grid-template-columns: 1fr;
    }
    
    .nav-tabs {
        overflow-x: auto;
    }
    
    .table-header {
        flex-direction: column;
        gap: 1rem;
        align-items: stretch;
    }
}

.overflow-x-auto {
    overflow-x: auto;
}

.mb-4 {
    margin-bottom: 1rem;
}

.mt-4 {
    margin-top: 1rem;
}

.text-sm {
    font-size: 0.875rem;
}

.text-xs {
    font-size: 0.75rem;
}

.flex {
    display: flex;
}

.items-center {
    align-items: center;
}

.justify-between {
    justify-content: space-between;
}

.gap-2 {
    gap: 0.5rem;
}

.gap-4 {
    gap: 1rem;
}

.modal-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.modal-content {
    background-color: #1e293b;
    border-radius: 0.5rem;
    padding: 2rem;
    width: 90%;
    max-width: 500px;
    border: 1px solid #334155;
}

.hidden {
    display: none;
}

.alert {
    padding: 1rem;
    border-radius: 0.375rem;
    margin-bottom: 1rem;
}

.alert-success {
    background-color: #065f46;
    color: #34d399;
    border: 1px solid #065f46;
}

.alert-error {
    background-color: #991b1b;
    color: #f87171;
    border: 1px solid #991b1b;
}

.alert-warning {
    background-color: #92400e;
    color: #fbbf24;
    border: 1px solid #92400e;
}
"""

# Сохраняем CSS файл
with open("admin_static/styles.css", "w", encoding="utf-8") as f:
    f.write(ADMIN_CSS_CONTENT)

templates = Jinja2Templates(directory="admin_templates")

# База данных администраторов
admins_db = {}
moderation_logs = []
user_reports = []
moderation_panel_online = False

# Инициализация базы данных
def init_database():
    global admins_db
    if Path(ADMIN_DB_PATH).exists():
        try:
            with open(ADMIN_DB_PATH, 'r', encoding='utf-8') as f:
                admins_db = json.load(f)
        except:
            admins_db = {}
    else:
        # Создаем администратора по умолчанию
        default_password = "admin123"
        password_hash = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        admins_db = {
            "admin": {
                "username": "admin",
                "password_hash": password_hash,
                "role": "superadmin",
                "permissions": ["view", "manage_users", "moderate", "view_reports", "system_control", "manage_admins"]
            }
        }
        save_database()
    
    # Загружаем логи и жалобы
    load_moderation_data()

def save_database():
    with open(ADMIN_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(admins_db, f, ensure_ascii=False, indent=2)

def load_moderation_data():
    global moderation_logs, user_reports
    try:
        if Path("moderation_logs.json").exists():
            with open("moderation_logs.json", "r", encoding="utf-8") as f:
                moderation_logs = json.load(f)
    except:
        moderation_logs = []
    
    try:
        if Path("user_reports.json").exists():
            with open("user_reports.json", "r", encoding="utf-8") as f:
                user_reports = json.load(f)
    except:
        user_reports = []

def save_moderation_data():
    try:
        with open("moderation_logs.json", "w", encoding="utf-8") as f:
            json.dump(moderation_logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Ошибка сохранения логов: {e}")
    
    try:
        with open("user_reports.json", "w", encoding="utf-8") as f:
            json.dump(user_reports, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Ошибка сохранения жалоб: {e}")

# Функции аутентификации
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except:
        return False

async def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    user = admins_db.get(credentials.username)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user

def verify_internal_auth(auth_key: str):
    """Проверка внутреннего ключа аутентификации"""
    return auth_key == INTERNAL_AUTH_KEY

# Функции для работы с панелью модерации
async def call_moderation_panel(endpoint: str, method: str = "GET", data: dict = None):
    """Вызов панели модерации"""
    global moderation_panel_online
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Auth": INTERNAL_AUTH_KEY,
        "X-Request-Source": "admin_panel"
    }
    timeout = aiohttp.ClientTimeout(total=10)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"{MODERATION_PANEL_URL}{endpoint}"
            print(f"🔗 Вызов панели модерации: {method} {url}")
            
            if method.upper() == "GET":
                async with session.get(url, headers=headers, ssl=False) as response:
                    return await handle_moderation_response(response)
            elif method.upper() == "POST":
                async with session.post(url, json=data, headers=headers, ssl=False) as response:
                    return await handle_moderation_response(response)
                    
    except aiohttp.ClientConnectorError as e:
        moderation_panel_online = False
        print(f"❌ Ошибка подключения к панели модерации: {e}")
        return {"error": "Connection refused", "details": str(e)}
    except Exception as e:
        moderation_panel_online = False
        print(f"💥 Ошибка вызова панели модерации: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

async def handle_moderation_response(response):
    """Обработка ответа от панели модерации"""
    global moderation_panel_online
    try:
        if response.status == 200:
            moderation_panel_online = True
            result = await response.json()
            print(f"✅ Успешный ответ от панели модерации")
            return result
        else:
            moderation_panel_online = False
            error_text = await response.text()
            print(f"❌ Ошибка HTTP {response.status} от панели модерации: {error_text}")
            return {"error": f"HTTP {response.status}", "details": error_text}
    except Exception as e:
        moderation_panel_online = False
        print(f"❌ Ошибка парсинга ответа от панели модерации: {e}")
        return {"error": f"Response parsing error: {str(e)}"}

# Роуты админ панели
@app.get("/")
async def admin_panel(request: Request):
    return templates.TemplateResponse("admin_panel.html", {
        "request": request,
        "moderation_panel_online": moderation_panel_online
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "admin_panel", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/status")
async def get_system_status(user: dict = Depends(authenticate_admin)):
    """Получить статус системы"""
    # Проверяем статус панели модерации
    moderation_status = await call_moderation_panel("/health")
    
    return {
        "admin_panel_online": True,
        "moderation_panel_online": moderation_panel_online,
        "moderation_panel_status": moderation_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/stats")
async def get_system_stats(user: dict = Depends(authenticate_admin)):
    """Получить статистику системы"""
    # Получаем статистику от панели модерации
    moderation_stats = await call_moderation_panel("/api/stats")
    
    if isinstance(moderation_stats, dict) and not moderation_stats.get("error"):
        return {
            "online_users": moderation_stats.get("online_users", 0),
            "unique_users": moderation_stats.get("unique_users", 0),
            "searching_users": moderation_stats.get("searching_users", 0),
            "active_rooms": moderation_stats.get("active_rooms", 0),
            "monitored_rooms": moderation_stats.get("monitored_rooms", 0),
            "banned_users": moderation_stats.get("banned_users", 0),
            "active_reports": moderation_stats.get("active_reports", 0),
            "moderation_panel_online": moderation_panel_online,
            "total_moderators": len([u for u in admins_db.values() if u.get("role") in ["moderator", "admin"]])
        }
    else:
        return {
            "online_users": 0,
            "unique_users": 0,
            "searching_users": 0,
            "active_rooms": 0,
            "monitored_rooms": 0,
            "banned_users": 0,
            "active_reports": len([r for r in user_reports if r.get("status") != "resolved"]),
            "moderation_panel_online": False,
            "total_moderators": len([u for u in admins_db.values() if u.get("role") in ["moderator", "admin"]])
        }

@app.get("/api/connections")
async def get_connections(user: dict = Depends(authenticate_admin)):
    """Получить подключения через панель модерации"""
    connections = await call_moderation_panel("/api/connections")
    if isinstance(connections, list):
        return connections
    else:
        return []

@app.get("/api/user/{user_id}")
async def get_user_status(user_id: str, user: dict = Depends(authenticate_admin)):
    """Получить статус пользователя через панель модерации"""
    user_status = await call_moderation_panel(f"/api/user/{user_id}")
    if isinstance(user_status, dict) and not user_status.get("error"):
        return user_status
    else:
        return {
            "user_id": user_id,
            "is_muted": False,
            "is_banned": False,
            "warnings_count": 0,
            "active_reports": 0
        }

@app.post("/api/moderate")
async def moderate_user(action: ModerationAction, user: dict = Depends(authenticate_admin)):
    """Применить действие модерации через панель модерации"""
    # Логируем действие
    log_moderation_action(
        user["username"], 
        action.action, 
        action.user_id,
        f"Причина: {action.reason}, Длительность: {action.duration_minutes} мин"
    )
    
    # Отправляем в панель модерации
    result = await call_moderation_panel("/api/moderate", "POST", action.dict())
    
    if result and not result.get("error"):
        return {
            "status": "success", 
            "message": "Действие успешно применено",
            "moderation_panel_response": result
        }
    else:
        return {
            "status": "error", 
            "message": "Не удалось применить действие",
            "moderation_panel_response": result
        }

@app.get("/api/moderation/reports")
async def get_moderation_reports(
    user: dict = Depends(authenticate_admin),
    limit: int = 100,
    offset: int = 0,
    status: str = "active"
):
    """Получить жалобы через панель модерации"""
    reports_data = await call_moderation_panel(f"/api/reports/list?limit={limit}&offset={offset}&status={status}")
    
    if isinstance(reports_data, dict) and 'reports' in reports_data:
        return {
            "reports": reports_data['reports'],
            "total": reports_data.get('total', len(reports_data['reports'])),
            "moderation_panel_online": moderation_panel_online
        }
    else:
        # Возвращаем локальные данные если панель недоступна
        filtered_reports = [r for r in user_reports if r.get("status") == status] if status != "all" else user_reports
        paginated_reports = filtered_reports[offset:offset + limit]
        
        return {
            "reports": paginated_reports,
            "total": len(filtered_reports),
            "moderation_panel_online": False
        }

@app.post("/api/moderation/reports")
async def create_moderation_report(report: UserReport, user: dict = Depends(authenticate_admin)):
    """Создать жалобу через панель модерации"""
    report_data = report.dict()
    report_data["id"] = len(user_reports) + 1
    report_data["created_by"] = user["username"]
    
    # Сохраняем локально
    user_reports.append(report_data)
    save_moderation_data()
    
    # Отправляем в панель модерации
    result = await call_moderation_panel("/api/reports", "POST", report_data)
    
    return {
        "status": "success",
        "message": "Жалоба создана",
        "report_id": report_data["id"],
        "moderation_panel_response": result
    }

@app.post("/api/moderation/reports/{report_id}/resolve")
async def resolve_moderation_report(report_id: int, user: dict = Depends(authenticate_admin)):
    """Решить жалобу через панель модерации"""
    # Обновляем локально
    for report in user_reports:
        if report.get("id") == report_id:
            report["status"] = "resolved"
            report["resolved_by"] = user["username"]
            report["resolved_at"] = datetime.now().isoformat()
            break
    
    save_moderation_data()
    
    # Отправляем в панель модерации
    result = await call_moderation_panel(f"/api/reports/{report_id}/resolve", "POST")
    
    return {
        "status": "success",
        "message": "Жалоба отмечена как решенная",
        "moderation_panel_response": result
    }

@app.get("/api/moderation/logs")
async def get_moderation_logs(
    user: dict = Depends(authenticate_admin),
    limit: int = 50,
    offset: int = 0
):
    """Получить логи модерации через панель модерации"""
    logs_data = await call_moderation_panel(f"/api/logs?limit={limit}&offset={offset}")
    
    if isinstance(logs_data, dict) and 'logs' in logs_data:
        return {
            "logs": logs_data['logs'],
            "total": logs_data.get('total', len(logs_data['logs'])),
            "moderation_panel_online": moderation_panel_online
        }
    else:
        # Возвращаем локальные логи если панель недоступна
        paginated_logs = moderation_logs[offset:offset + limit]
        
        return {
            "logs": paginated_logs,
            "total": len(moderation_logs),
            "moderation_panel_online": False
        }

@app.post("/api/moderation/actions")
async def handle_moderation_action(action: ModerationAction, user: dict = Depends(authenticate_admin)):
    """Обработка действий модерации от панели модерации"""
    # Логируем действие
    log_moderation_action(
        action.moderator,
        action.action,
        action.user_id,
        f"Причина: {action.reason}, Длительность: {action.duration_minutes} мин"
    )
    
    # Здесь можно добавить дополнительную логику обработки действий
    
    return {
        "status": "success",
        "message": "Действие модерации обработано",
        "action_id": len(moderation_logs),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/report")
async def receive_user_report(report: UserReport):
    """Прием жалоб от пользователей (эндпоинт для основного сервера)"""
    try:
        print(f"🔍 Received user report: {report.dict()}")
        
        # Сохраняем жалобу
        report_data = report.dict()
        report_data["id"] = len(user_reports) + 1
        report_data["timestamp"] = datetime.now().isoformat()
        report_data["status"] = "active"
        
        user_reports.append(report_data)
        save_moderation_data()
        
        # Логируем получение жалобы
        log_moderation_action(
            "system", 
            "report_received", 
            report.reported_user_id,
            f"Жалоба от {report.reporter_id}: {report.reason}"
        )
        
        # Пытаемся отправить в панель модерации
        try:
            moderation_result = await call_moderation_panel("/api/internal/reports", "POST", {
                "report_data": report_data,
                "auth_key": INTERNAL_AUTH_KEY
            })
            print(f"✅ Жалоба отправлена в панель модерации: {moderation_result}")
        except Exception as e:
            print(f"⚠️ Не удалось отправить жалобу в панель модерации: {e}")
        
        print(f"✅ Жалоба сохранена: ID {report_data['id']}")
        
        return {
            "status": "success", 
            "message": "Жалоба успешно получена",
            "report_id": report_data["id"],
            "received_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Ошибка при обработке жалобы: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/internal/reports")
async def receive_internal_report(internal_report: InternalReport):
    """Прием жалоб от панели модерации"""
    try:
        if not verify_internal_auth(internal_report.auth_key):
            raise HTTPException(status_code=401, detail="Invalid authentication key")
        
        report_data = internal_report.report_data
        
        # Сохраняем жалобу
        report_data["id"] = len(user_reports) + 1
        if "timestamp" not in report_data:
            report_data["timestamp"] = datetime.now().isoformat()
        report_data["status"] = "active"
        
        user_reports.append(report_data)
        save_moderation_data()
        
        print(f"✅ Внутренняя жалоба получена от панели модерации: ID {report_data['id']}")
        
        return {
            "status": "success", 
            "message": "Жалоба успешно получена",
            "report_id": report_data["id"]
        }
        
    except Exception as e:
        print(f"❌ Ошибка при обработке внутренней жалобы: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/system/users")
async def get_system_users(user: dict = Depends(authenticate_admin)):
    """Получить список пользователей системы (администраторов/модераторов)"""
    users_list = []
    for username, user_data in admins_db.items():
        users_list.append({
            "username": username,
            "role": user_data.get("role", "user"),
            "permissions": user_data.get("permissions", []),
            "last_login": user_data.get("last_login", "never")
        })
    
    return {"users": users_list}

@app.post("/api/system/users")
async def create_system_user(
    user_data: dict, 
    current_user: dict = Depends(authenticate_admin)
):
    """Создать нового пользователя системы"""
    if "manage_admins" not in current_user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    username = user_data.get("username")
    password = user_data.get("password")
    role = user_data.get("role", "moderator")
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    if username in admins_db:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Хэшируем пароль
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Определяем permissions по роли
    if role == "superadmin":
        permissions = ["view", "manage_users", "moderate", "view_reports", "system_control", "manage_admins"]
    elif role == "admin":
        permissions = ["view", "manage_users", "moderate", "view_reports", "system_control"]
    else:  # moderator
        permissions = ["view", "moderate", "view_reports"]
    
    admins_db[username] = {
        "username": username,
        "password_hash": password_hash,
        "role": role,
        "permissions": permissions,
        "created_by": current_user["username"],
        "created_at": datetime.now().isoformat()
    }
    
    save_database()
    
    return {"status": "success", "message": "User created successfully"}

# Вспомогательные функции
def log_moderation_action(moderator: str, action: str, target_user: str, details: str = ""):
    """Логирование действий модерации"""
    log_entry = {
        "moderator": moderator,
        "action": action,
        "target_user": target_user,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    moderation_logs.append(log_entry)
    save_moderation_data()

# Статические файлы
app.mount("/static", StaticFiles(directory="admin_static"), name="static")

# HTML шаблон админ панели
ADMIN_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Админ панель</title>
    <link rel="stylesheet" href="/static/styles.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <div class="header">
        <div class="container header-content">
            <h1>
                <i class="fas fa-cogs mr-3"></i>
                Админ панель
            </h1>
            <div>
                <span id="currentUser" style="margin-right: 1rem;"></span>
                <button onclick="logout()" class="btn btn-danger">
                    <i class="fas fa-sign-out-alt mr-2"></i>Выйти
                </button>
            </div>
        </div>
    </div>

    <div class="container">
        <div id="connectionStatus" class="connection-status connection-offline">
            <i class="fas fa-exclamation-triangle mr-2"></i>Панель модерации недоступна
        </div>

        <!-- Статистика -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Онлайн пользователей</h3>
                <p class="stat-value" id="onlineUsers">0</p>
            </div>
            <div class="stat-card">
                <h3>Активные комнаты</h3>
                <p class="stat-value" id="activeRooms">0</p>
            </div>
            <div class="stat-card">
                <h3>Забанено</h3>
                <p class="stat-value" id="bannedUsers">0</p>
            </div>
            <div class="stat-card">
                <h3>Модераторы</h3>
                <p class="stat-value" id="totalModerators">0</p>
            </div>
        </div>

        <!-- Навигация -->
        <div class="nav-tabs">
            <button class="tab-btn active" data-tab="overview">
                <i class="fas fa-tachometer-alt mr-2"></i>Обзор
            </button>
            <button class="tab-btn" data-tab="moderation">
                <i class="fas fa-gavel mr-2"></i>Модерация
            </button>
            <button class="tab-btn" data-tab="reports">
                <i class="fas fa-flag mr-2"></i>Жалобы
            </button>
            <button class="tab-btn" data-tab="logs">
                <i class="fas fa-history mr-2"></i>Логи
            </button>
            <button class="tab-btn" data-tab="system">
                <i class="fas fa-cog mr-2"></i>Система
            </button>
        </div>

        <!-- Контент вкладок -->
        <div id="overviewTab" class="tab-content active">
            <div class="grid-2">
                <!-- Статус системы -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Статус системы</h3>
                    </div>
                    <div id="systemStatus">
                        <div class="loading">Загрузка статуса...</div>
                    </div>
                </div>

                <!-- Быстрые действия -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Быстрые действия</h3>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <button onclick="showUserLookup()" class="btn btn-primary">
                            <i class="fas fa-search mr-2"></i>Найти пользователя
                        </button>
                        <button onclick="showQuickModeration()" class="btn btn-warning">
                            <i class="fas fa-gavel mr-2"></i>Быстрая модерация
                        </button>
                        <button onclick="loadConnections()" class="btn btn-success">
                            <i class="fas fa-plug mr-2"></i>Просмотр подключений
                        </button>
                        <button onclick="syncWithModerationPanel()" class="btn btn-primary">
                            <i class="fas fa-sync-alt mr-2"></i>Синхронизировать с панелью модерации
                        </button>
                    </div>
                </div>
            </div>

            <!-- Активные подключения -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Активные подключения</h3>
                    <button onclick="loadConnections()" class="btn btn-primary btn-sm">
                        <i class="fas fa-sync-alt mr-2"></i>Обновить
                    </button>
                </div>
                <div class="overflow-x-auto">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>ID пользователя</th>
                                <th>Статус</th>
                                <th>Комната</th>
                                <th>Действия</th>
                            </tr>
                        </thead>
                        <tbody id="connectionsTable">
                            <tr>
                                <td colspan="4" class="no-data">Загрузка данных...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div id="moderationTab" class="tab-content">
            <div class="grid-2">
                <!-- Модерация пользователя -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Модерация пользователя</h3>
                    </div>
                    <div class="form-group">
                        <label class="form-label">ID пользователя</label>
                        <input type="text" id="moderateUserId" class="form-input" placeholder="Введите ID пользователя">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Действие</label>
                        <select id="moderateAction" class="form-input">
                            <option value="warn">Предупреждение</option>
                            <option value="mute">Заглушить</option>
                            <option value="kick">Выгнать</option>
                            <option value="ban">Забанить</option>
                            <option value="unmute">Разглушить</option>
                            <option value="unban">Разбанить</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Длительность (минут)</label>
                        <input type="number" id="moderateDuration" value="60" class="form-input">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Причина</label>
                        <textarea id="moderateReason" rows="3" class="form-input" placeholder="Укажите причину действия"></textarea>
                    </div>
                    <button onclick="moderateUser()" class="btn btn-danger w-full">
                        <i class="fas fa-gavel mr-2"></i>Применить действие
                    </button>
                </div>

                <!-- Статус пользователя -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Статус пользователя</h3>
                    </div>
                    <div class="form-group">
                        <label class="form-label">ID пользователя</label>
                        <div class="flex gap-2">
                            <input type="text" id="checkUserId" class="form-input" placeholder="Введите ID пользователя">
                            <button onclick="checkUserStatus()" class="btn btn-primary">
                                <i class="fas fa-search mr-2"></i>Проверить
                            </button>
                        </div>
                    </div>
                    <div id="userStatusResult">
                        <!-- Результат проверки статуса -->
                    </div>
                </div>
            </div>
        </div>

        <div id="reportsTab" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Жалобы пользователей</h3>
                    <div class="flex gap-2">
                        <button onclick="loadReports()" class="btn btn-primary">
                            <i class="fas fa-sync-alt mr-2"></i>Обновить
                        </button>
                        <select id="reportStatusFilter" onchange="loadReports()" class="form-input" style="width: auto;">
                            <option value="active">Активные</option>
                            <option value="resolved">Решенные</option>
                            <option value="all">Все</option>
                        </select>
                    </div>
                </div>
                <div id="reportsList">
                    <div class="loading">Загрузка жалоб...</div>
                </div>
            </div>
        </div>

        <div id="logsTab" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Логи модерации</h3>
                    <button onclick="loadLogs()" class="btn btn-primary">
                        <i class="fas fa-sync-alt mr-2"></i>Обновить
                    </button>
                </div>
                <div class="overflow-x-auto">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Время</th>
                                <th>Модератор</th>
                                <th>Действие</th>
                                <th>Пользователь</th>
                                <th>Детали</th>
                            </tr>
                        </thead>
                        <tbody id="logsTable">
                            <tr>
                                <td colspan="5" class="no-data">Загрузка логов...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div id="systemTab" class="tab-content">
            <div class="grid-2">
                <!-- Пользователи системы -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Пользователи системы</h3>
                        <button onclick="showCreateUserModal()" class="btn btn-primary btn-sm">
                            <i class="fas fa-plus mr-2"></i>Добавить
                        </button>
                    </div>
                    <div id="systemUsersList">
                        <div class="loading">Загрузка пользователей...</div>
                    </div>
                </div>

                <!-- Настройки системы -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Настройки системы</h3>
                    </div>
                    <div class="form-group">
                        <label class="form-label">URL панели модерации</label>
                        <input type="text" id="moderationPanelUrl" value="http://192.168.0.102:8001" class="form-input">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Ключ аутентификации</label>
                        <input type="text" id="authKey" value="moderation_panel_secret_2024" class="form-input">
                    </div>
                    <button onclick="updateSystemSettings()" class="btn btn-primary w-full">
                        <i class="fas fa-save mr-2"></i>Сохранить настройки
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Модальные окна -->
    <div id="loginModal" class="login-modal">
        <div class="login-form">
            <h3 style="margin-top: 0; text-align: center;">
                <i class="fas fa-cogs mr-2"></i>Админ панель
            </h3>
            <div class="form-group">
                <label class="form-label">Логин</label>
                <input type="text" id="username" class="form-input" placeholder="Введите логин">
            </div>
            <div class="form-group">
                <label class="form-label">Пароль</label>
                <input type="password" id="password" class="form-input" placeholder="Введите пароль">
            </div>
            <button onclick="login()" class="btn btn-primary w-full">
                <i class="fas fa-sign-in-alt mr-2"></i>Войти
            </button>
            <div style="margin-top: 1rem; font-size: 0.875rem; color: #94a3b8; text-align: center;">
                По умолчанию: admin / admin123
            </div>
        </div>
    </div>

    <div id="createUserModal" class="modal-backdrop hidden">
        <div class="modal-content">
            <h3 style="margin-top: 0;">Создать пользователя</h3>
            <div class="form-group">
                <label class="form-label">Логин</label>
                <input type="text" id="newUsername" class="form-input">
            </div>
            <div class="form-group">
                <label class="form-label">Пароль</label>
                <input type="password" id="newPassword" class="form-input">
            </div>
            <div class="form-group">
                <label class="form-label">Роль</label>
                <select id="newUserRole" class="form-input">
                    <option value="moderator">Модератор</option>
                    <option value="admin">Администратор</option>
                    <option value="superadmin">Супер администратор</option>
                </select>
            </div>
            <div class="flex gap-2">
                <button onclick="createSystemUser()" class="btn btn-primary w-full">Создать</button>
                <button onclick="hideCreateUserModal()" class="btn btn-primary">Отмена</button>
            </div>
        </div>
    </div>

    <script>
        let currentUser = null;
        let authHeader = null;

        // Инициализация
        document.addEventListener('DOMContentLoaded', function() {
            showLoginModal();
            setupTabNavigation();
        });

        // Навигация по вкладкам
        function setupTabNavigation() {
            const tabBtns = document.querySelectorAll('.tab-btn');
            tabBtns.forEach(btn => {
                btn.addEventListener('click', function() {
                    const targetTab = this.getAttribute('data-tab');
                    
                    tabBtns.forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    document.getElementById(targetTab + 'Tab').classList.add('active');
                    
                    // Загружаем данные для активной вкладки
                    if (targetTab === 'overview') {
                        loadSystemStatus();
                        loadConnections();
                    }
                    if (targetTab === 'reports') loadReports();
                    if (targetTab === 'logs') loadLogs();
                    if (targetTab === 'system') loadSystemUsers();
                });
            });
        }

        // Аутентификация
        function showLoginModal() {
            document.getElementById('loginModal').style.display = 'flex';
        }

        function hideLoginModal() {
            document.getElementById('loginModal').style.display = 'none';
        }

        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            if (!username || !password) {
                alert('Введите логин и пароль');
                return;
            }
            
            try {
                authHeader = 'Basic ' + btoa(username + ':' + password);
                
                const response = await fetch('/api/status', {
                    headers: { 'Authorization': authHeader }
                });
                
                if (response.ok) {
                    currentUser = username;
                    document.getElementById('currentUser').textContent = username;
                    hideLoginModal();
                    loadStats();
                    loadSystemStatus();
                } else {
                    alert('Ошибка авторизации');
                }
            } catch (error) {
                console.error('Ошибка входа:', error);
                alert('Ошибка подключения к серверу');
            }
        }

        function logout() {
            currentUser = null;
            authHeader = null;
            showLoginModal();
        }

        // API функции
        async function apiCall(endpoint, options = {}) {
            if (!authHeader) throw new Error('Not authenticated');
            
            const defaultOptions = {
                headers: {
                    'Authorization': authHeader,
                    'Content-Type': 'application/json'
                }
            };
            
            const response = await fetch(endpoint, { ...defaultOptions, ...options });
            
            if (response.status === 401) {
                logout();
                throw new Error('Authentication failed');
            }
            
            return response.json();
        }

        // Загрузка статистики
        async function loadStats() {
            try {
                const stats = await apiCall('/api/stats');
                
                document.getElementById('onlineUsers').textContent = stats.online_users || 0;
                document.getElementById('activeRooms').textContent = stats.active_rooms || 0;
                document.getElementById('bannedUsers').textContent = stats.banned_users || 0;
                document.getElementById('totalModerators').textContent = stats.total_moderators || 0;
                
                // Статус подключения
                const statusElement = document.getElementById('connectionStatus');
                if (stats.moderation_panel_online) {
                    statusElement.className = 'connection-status connection-online';
                    statusElement.innerHTML = '<i class="fas fa-check-circle mr-2"></i>Панель модерации доступна';
                } else {
                    statusElement.className = 'connection-status connection-offline';
                    statusElement.innerHTML = '<i class="fas fa-exclamation-triangle mr-2"></i>Панель модерации недоступна';
                }
                
            } catch (error) {
                console.error('Ошибка загрузки статистики:', error);
            }
        }

        // Статус системы
        async function loadSystemStatus() {
            try {
                const status = await apiCall('/api/status');
                const container = document.getElementById('systemStatus');
                
                container.innerHTML = `
                    <div class="flex justify-between items-center mb-4">
                        <span>Админ панель:</span>
                        <span class="status-badge status-online">
                            <i class="fas fa-check-circle mr-2"></i>Online
                        </span>
                    </div>
                    <div class="flex justify-between items-center mb-4">
                        <span>Панель модерации:</span>
                        <span class="status-badge ${status.moderation_panel_online ? 'status-online' : 'status-offline'}">
                            <i class="fas fa-${status.moderation_panel_online ? 'check' : 'times'}-circle mr-2"></i>
                            ${status.moderation_panel_online ? 'Online' : 'Offline'}
                        </span>
                    </div>
                    <div class="text-sm text-gray-400">
                        Последняя проверка: ${new Date().toLocaleString()}
                    </div>
                `;
                
            } catch (error) {
                console.error('Ошибка загрузки статуса:', error);
            }
        }

        // Подключения
        async function loadConnections() {
            try {
                const connections = await apiCall('/api/connections');
                const table = document.getElementById('connectionsTable');
                
                if (!connections || connections.length === 0) {
                    table.innerHTML = '<tr><td colspan="4" class="no-data">Нет активных подключений</td></tr>';
                    return;
                }
                
                table.innerHTML = connections.map(conn => `
                    <tr>
                        <td>${conn.user_id || 'N/A'}</td>
                        <td>
                            <span class="status-badge ${getStatusClass(conn.status)}">
                                ${getStatusText(conn.status)}
                            </span>
                        </td>
                        <td>${conn.room_id || 'N/A'}</td>
                        <td>
                            <button onclick="moderateUserById('${conn.user_id}')" class="btn btn-sm btn-warning">
                                <i class="fas fa-gavel mr-2"></i>Модерировать
                            </button>
                        </td>
                    </tr>
                `).join('');
                
            } catch (error) {
                console.error('Ошибка загрузки подключений:', error);
                document.getElementById('connectionsTable').innerHTML = 
                    '<tr><td colspan="4" class="no-data" style="color: #ef4444;">Ошибка загрузки данных</td></tr>';
            }
        }

        function getStatusClass(status) {
            switch (status) {
                case 'connected': return 'status-connected';
                case 'searching': return 'status-searching';
                default: return 'status-idle';
            }
        }

        function getStatusText(status) {
            switch (status) {
                case 'connected': return 'В комнате';
                case 'searching': return 'Поиск';
                default: return 'Неактивен';
            }
        }

        // Модерация
        async function moderateUser() {
            const userId = document.getElementById('moderateUserId').value;
            const action = document.getElementById('moderateAction').value;
            const duration = document.getElementById('moderateDuration').value;
            const reason = document.getElementById('moderateReason').value;
            
            if (!userId || !reason) {
                alert('Заполните все обязательные поля');
                return;
            }
            
            try {
                const result = await apiCall('/api/moderate', {
                    method: 'POST',
                    body: JSON.stringify({
                        user_id: userId,
                        action: action,
                        duration_minutes: parseInt(duration),
                        reason: reason,
                        moderator: currentUser,
                        timestamp: new Date().toISOString()
                    })
                });
                
                alert(result.message || 'Действие применено');
                document.getElementById('moderateUserId').value = '';
                document.getElementById('moderateReason').value = '';
                
            } catch (error) {
                console.error('Ошибка модерации:', error);
                alert('Ошибка применения действия');
            }
        }

        function moderateUserById(userId) {
            document.getElementById('moderateUserId').value = userId;
            document.querySelector('[data-tab="moderation"]').click();
        }

        // Статус пользователя
        async function checkUserStatus() {
            const userId = document.getElementById('checkUserId').value;
            
            if (!userId) {
                alert('Введите ID пользователя');
                return;
            }
            
            try {
                const status = await apiCall(`/api/user/${userId}`);
                const resultDiv = document.getElementById('userStatusResult');
                
                resultDiv.innerHTML = `
                    <div class="card">
                        <h4>Статус пользователя ${userId}</h4>
                        <p>Заглушен: ${status.is_muted ? 'Да' : 'Нет'}</p>
                        <p>Забанен: ${status.is_banned ? 'Да' : 'Нет'}</p>
                        <p>Предупреждения: ${status.warnings_count || 0}</p>
                        <p>Активные жалобы: ${status.active_reports || 0}</p>
                    </div>
                `;
                
            } catch (error) {
                console.error('Ошибка проверки статуса:', error);
                alert('Ошибка получения статуса пользователя');
            }
        }

        // Жалобы
        async function loadReports() {
            try {
                const statusFilter = document.getElementById('reportStatusFilter').value;
                const data = await apiCall(`/api/moderation/reports?status=${statusFilter}&limit=50`);
                const container = document.getElementById('reportsList');
                
                if (!data.reports || data.reports.length === 0) {
                    container.innerHTML = '<div class="no-data">Нет жалоб</div>';
                    return;
                }
                
                container.innerHTML = data.reports.map(report => `
                    <div class="report-item ${report.status === 'resolved' ? 'resolved' : ''}">
                        <div class="flex justify-between items-start">
                            <div>
                                <strong>Жалоба #${report.id}</strong>
                                <div class="text-sm">
                                    От: ${report.reporter_id || 'N/A'} → На: ${report.reported_user_id || 'N/A'}
                                    <br>Причина: ${report.reason || 'N/A'}
                                    <br>Комната: ${report.room_id || 'N/A'}
                                    <br>Время: ${new Date(report.timestamp).toLocaleString()}
                                    ${report.status === 'resolved' ? `<br>Решено: ${report.resolved_by} (${new Date(report.resolved_at).toLocaleString()})` : ''}
                                </div>
                            </div>
                            ${report.status !== 'resolved' ? `
                                <button onclick="resolveReport(${report.id})" class="btn btn-sm btn-primary">
                                    <i class="fas fa-check mr-2"></i>Решено
                                </button>
                            ` : ''}
                        </div>
                        ${report.evidence ? `<div class="mt-4"><strong>Доказательства:</strong> ${report.evidence}</div>` : ''}
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('Ошибка загрузки жалоб:', error);
                document.getElementById('reportsList').innerHTML = 
                    '<div class="no-data" style="color: #ef4444;">Ошибка загрузки жалоб</div>';
            }
        }

        async function resolveReport(reportId) {
            if (!confirm('Отметить жалобу как решенную?')) return;
            
            try {
                await apiCall(`/api/moderation/reports/${reportId}/resolve`, { method: 'POST' });
                loadReports();
            } catch (error) {
                console.error('Ошибка разрешения жалобы:', error);
                alert('Ошибка обновления жалобы');
            }
        }

        // Логи
        async function loadLogs() {
            try {
                const data = await apiCall('/api/moderation/logs?limit=50');
                const table = document.getElementById('logsTable');
                
                if (!data.logs || data.logs.length === 0) {
                    table.innerHTML = '<tr><td colspan="5" class="no-data">Нет записей в логах</td></tr>';
                    return;
                }
                
                table.innerHTML = data.logs.slice().reverse().map(log => `
                    <tr>
                        <td>${new Date(log.timestamp).toLocaleString()}</td>
                        <td>${log.moderator || 'system'}</td>
                        <td>${log.action || 'N/A'}</td>
                        <td>${log.target_user || 'N/A'}</td>
                        <td>${log.details || ''}</td>
                    </tr>
                `).join('');
                
            } catch (error) {
                console.error('Ошибка загрузки логов:', error);
                document.getElementById('logsTable').innerHTML = 
                    '<tr><td colspan="5" class="no-data" style="color: #ef4444;">Ошибка загрузки логов</td></tr>';
            }
        }

        // Системные пользователи
        async function loadSystemUsers() {
            try {
                const data = await apiCall('/api/system/users');
                const container = document.getElementById('systemUsersList');
                
                container.innerHTML = data.users.map(user => `
                    <div class="report-item">
                        <div class="flex justify-between items-center">
                            <div>
                                <strong>${user.username}</strong>
                                <div class="text-sm">
                                    Роль: ${user.role}
                                    <br>Разрешения: ${user.permissions.join(', ')}
                                    <br>Последний вход: ${user.last_login}
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('Ошибка загрузки пользователей:', error);
                document.getElementById('systemUsersList').innerHTML = 
                    '<div class="no-data" style="color: #ef4444;">Ошибка загрузки пользователей</div>';
            }
        }

        function showCreateUserModal() {
            document.getElementById('createUserModal').classList.remove('hidden');
        }

        function hideCreateUserModal() {
            document.getElementById('createUserModal').classList.add('hidden');
        }

        async function createSystemUser() {
            const username = document.getElementById('newUsername').value;
            const password = document.getElementById('newPassword').value;
            const role = document.getElementById('newUserRole').value;
            
            if (!username || !password) {
                alert('Заполните все поля');
                return;
            }
            
            try {
                await apiCall('/api/system/users', {
                    method: 'POST',
                    body: JSON.stringify({
                        username: username,
                        password: password,
                        role: role
                    })
                });
                
                alert('Пользователь создан');
                hideCreateUserModal();
                loadSystemUsers();
                
            } catch (error) {
                console.error('Ошибка создания пользователя:', error);
                alert('Ошибка создания пользователя');
            }
        }

        // Синхронизация
        async function syncWithModerationPanel() {
            try {
                await loadStats();
                await loadSystemStatus();
                alert('Синхронизация завершена');
            } catch (error) {
                console.error('Ошибка синхронизации:', error);
                alert('Ошибка синхронизации');
            }
        }

        // Автообновление
        setInterval(loadStats, 15000);
        setInterval(loadSystemStatus, 30000);

        // Обработка клавиш
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                if (document.getElementById('loginModal').style.display !== 'none') {
                    login();
                }
            }
        });
    </script>
</body>
</html>
"""

# Сохраняем HTML шаблон
with open("admin_templates/admin_panel.html", "w", encoding="utf-8") as f:
    f.write(ADMIN_HTML)

# Инициализация при запуске
init_database()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Запуск админ панели...")
    print(f"📊 URL панели: http://localhost:8002")
    print(f"🔗 Панель модерации: {MODERATION_PANEL_URL}")
    print("👮 Логин по умолчанию: admin / admin123")
    
    uvicorn.run(
        app, 
        host="192.168.0.102", 
        port=8002,
        log_level="info"
    )