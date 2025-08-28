import requests
import time
import matplotlib.pyplot as plt
from datetime import datetime

def monitor_server(host, interval=5):
    timestamps = []
    online_users = []
    searching_users = []
    active_rooms = []
    
    try:
        while True:
            try:
                response = requests.get(f"{host}/api/stats", timeout=5)
                stats = response.json()
                
                current_time = datetime.now().strftime("%H:%M:%S")
                timestamps.append(current_time)
                online_users.append(stats.get('online_users', 0))
                searching_users.append(stats.get('searching_users', 0))
                active_rooms.append(stats.get('active_rooms', 0))
                
                # Сохраняем только последние 60 точек
                if len(timestamps) > 60:
                    timestamps = timestamps[-60:]
                    online_users = online_users[-60:]
                    searching_users = searching_users[-60:]
                    active_rooms = active_rooms[-60:]
                
                # Обновляем график
                plt.clf()
                plt.plot(timestamps, online_users, label='Online Users')
                plt.plot(timestamps, searching_users, label='Searching Users')
                plt.plot(timestamps, active_rooms, label='Active Rooms')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.pause(0.1)
                
                print(f"{current_time} - Online: {online_users[-1]}, Searching: {searching_users[-1]}, Rooms: {active_rooms[-1]}")
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("Monitoring stopped")

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    monitor_server(host)