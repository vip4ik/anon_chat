#!/usr/bin/env python3
import asyncio
import json
import websockets
import requests
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the backend URL from frontend .env
BACKEND_URL = "https://8023ac5f-9f53-4e31-adda-551cf1c4fdfd.preview.emergentagent.com"
WS_URL = f"wss://{BACKEND_URL.split('//')[1]}/ws"
API_URL = f"{BACKEND_URL}/api"

class TestResult:
    def __init__(self):
        self.success = True
        self.messages = []
    
    def add_success(self, message):
        self.messages.append(f"✅ {message}")
    
    def add_failure(self, message):
        self.success = False
        self.messages.append(f"❌ {message}")
    
    def print_results(self):
        for message in self.messages:
            print(message)
        print(f"\nOverall test result: {'SUCCESS' if self.success else 'FAILURE'}")
        return self.success

class WebSocketClient:
    def __init__(self, user_id):
        self.user_id = user_id
        self.ws = None
        self.connected = False
        self.messages = []
        self.partner_id = None
        self.room_id = None
    
    async def connect(self):
        try:
            self.ws = await websockets.connect(f"{WS_URL}/{self.user_id}")
            self.connected = True
            logger.info(f"User {self.user_id} connected")
            return True
        except Exception as e:
            logger.error(f"Connection error for user {self.user_id}: {e}")
            return False
    
    async def disconnect(self):
        if self.ws:
            await self.ws.close()
            self.connected = False
            logger.info(f"User {self.user_id} disconnected")
    
    async def send_message(self, message):
        if not self.connected:
            logger.error(f"Cannot send message, user {self.user_id} not connected")
            return False
        
        try:
            await self.ws.send(json.dumps(message))
            logger.info(f"User {self.user_id} sent: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending message for user {self.user_id}: {e}")
            return False
    
    async def receive_message(self, timeout=5):
        if not self.connected:
            logger.error(f"Cannot receive message, user {self.user_id} not connected")
            return None
        
        try:
            message = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            message_data = json.loads(message)
            self.messages.append(message_data)
            logger.info(f"User {self.user_id} received: {message_data}")
            
            # Update client state based on message
            if message_data.get("type") == "match_found":
                self.partner_id = message_data.get("partner_id")
                self.room_id = message_data.get("room_id")
            
            return message_data
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for message for user {self.user_id}")
            return None
        except Exception as e:
            logger.error(f"Error receiving message for user {self.user_id}: {e}")
            return None
    
    async def start_search(self):
        return await self.send_message({"type": "start_search"})
    
    async def stop_search(self):
        return await self.send_message({"type": "stop_search"})
    
    async def next_partner(self):
        return await self.send_message({"type": "next_partner"})
    
    async def send_webrtc_signal(self, signal_data):
        return await self.send_message({
            "type": "webrtc_signal",
            "signal": signal_data
        })
    
    async def ping(self):
        return await self.send_message({"type": "ping"})

async def test_websocket_connection():
    """Test WebSocket connection and basic session management"""
    test_result = TestResult()
    user_id = str(uuid.uuid4())
    client = WebSocketClient(user_id)
    
    # Test connection
    if await client.connect():
        test_result.add_success(f"WebSocket connection established for user {user_id}")
        
        # Test welcome message
        welcome_msg = await client.receive_message()
        if welcome_msg and welcome_msg.get("type") == "connected":
            test_result.add_success("Received welcome message")
        else:
            test_result.add_failure("Did not receive welcome message")
        
        # Test ping-pong
        await client.ping()
        pong_msg = await client.receive_message()
        if pong_msg and pong_msg.get("type") == "pong":
            test_result.add_success("Ping-pong mechanism working")
        else:
            test_result.add_failure("Ping-pong mechanism not working")
        
        # Test disconnection
        await client.disconnect()
        test_result.add_success("WebSocket disconnection successful")
    else:
        test_result.add_failure("Failed to establish WebSocket connection")
    
    return test_result

async def test_user_matching():
    """Test user matching system and room creation"""
    test_result = TestResult()
    
    # Create two users
    user1_id = str(uuid.uuid4())
    user2_id = str(uuid.uuid4())
    client1 = WebSocketClient(user1_id)
    client2 = WebSocketClient(user2_id)
    
    # Connect both users
    if await client1.connect() and await client2.connect():
        test_result.add_success("Both users connected successfully")
        
        # Skip welcome messages
        await client1.receive_message()
        await client2.receive_message()
        
        # User 1 starts searching
        await client1.start_search()
        search_msg = await client1.receive_message()
        if search_msg and search_msg.get("type") == "searching":
            test_result.add_success("User 1 searching status confirmed")
        else:
            test_result.add_failure("User 1 searching status not confirmed")
        
        # User 2 starts searching (should match with user 1)
        await client2.start_search()
        
        # Both users should receive match_found messages
        match_msg1 = await client1.receive_message()
        match_msg2 = await client2.receive_message()
        
        if (match_msg1 and match_msg1.get("type") == "match_found" and 
            match_msg2 and match_msg2.get("type") == "match_found"):
            test_result.add_success("Users matched successfully")
            
            # Verify partner IDs are correct
            if (match_msg1.get("partner_id") == user2_id and 
                match_msg2.get("partner_id") == user1_id):
                test_result.add_success("Partner IDs correctly assigned")
            else:
                test_result.add_failure("Partner IDs incorrectly assigned")
            
            # Verify room IDs match
            if match_msg1.get("room_id") == match_msg2.get("room_id"):
                test_result.add_success("Room IDs match for both users")
            else:
                test_result.add_failure("Room IDs don't match for both users")
        else:
            test_result.add_failure("Users not matched properly")
        
        # Test stop_search functionality with a third user
        user3_id = str(uuid.uuid4())
        client3 = WebSocketClient(user3_id)
        
        if await client3.connect():
            await client3.receive_message()  # Skip welcome message
            
            # Start searching
            await client3.start_search()
            search_msg = await client3.receive_message()
            if search_msg and search_msg.get("type") == "searching":
                test_result.add_success("User 3 searching status confirmed")
            else:
                test_result.add_failure("User 3 searching status not confirmed")
            
            # Stop searching
            await client3.stop_search()
            stop_msg = await client3.receive_message()
            if stop_msg and stop_msg.get("type") == "search_stopped":
                test_result.add_success("User 3 search stopped successfully")
            else:
                test_result.add_failure("User 3 search not stopped properly")
            
            await client3.disconnect()
        else:
            test_result.add_failure("Failed to connect user 3")
        
        # Clean up
        await client1.disconnect()
        await client2.disconnect()
    else:
        test_result.add_failure("Failed to connect both users")
    
    return test_result

async def test_webrtc_signaling():
    """Test WebRTC signaling between matched users"""
    test_result = TestResult()
    
    # Create two users
    user1_id = str(uuid.uuid4())
    user2_id = str(uuid.uuid4())
    client1 = WebSocketClient(user1_id)
    client2 = WebSocketClient(user2_id)
    
    # Connect both users
    if await client1.connect() and await client2.connect():
        # Skip welcome messages
        await client1.receive_message()
        await client2.receive_message()
        
        # Match users
        await client1.start_search()
        await client1.receive_message()  # searching message
        await client2.start_search()
        
        # Both users should receive match_found messages
        await client1.receive_message()
        await client2.receive_message()
        
        # Test WebRTC signaling from user1 to user2
        test_offer = {"type": "offer", "sdp": "test_sdp_offer"}
        await client1.send_webrtc_signal(test_offer)
        
        # User2 should receive the signal
        signal_msg = await client2.receive_message()
        if (signal_msg and 
            signal_msg.get("type") == "webrtc_signal" and 
            signal_msg.get("from_user") == user1_id and 
            signal_msg.get("signal", {}).get("type") == "offer"):
            test_result.add_success("WebRTC offer signal correctly relayed")
        else:
            test_result.add_failure("WebRTC offer signal not correctly relayed")
        
        # Test WebRTC signaling from user2 to user1
        test_answer = {"type": "answer", "sdp": "test_sdp_answer"}
        await client2.send_webrtc_signal(test_answer)
        
        # User1 should receive the signal
        signal_msg = await client1.receive_message()
        if (signal_msg and 
            signal_msg.get("type") == "webrtc_signal" and 
            signal_msg.get("from_user") == user2_id and 
            signal_msg.get("signal", {}).get("type") == "answer"):
            test_result.add_success("WebRTC answer signal correctly relayed")
        else:
            test_result.add_failure("WebRTC answer signal not correctly relayed")
        
        # Test ICE candidate exchange
        test_ice = {"type": "ice-candidate", "candidate": "test_ice_candidate"}
        await client1.send_webrtc_signal(test_ice)
        
        # User2 should receive the ICE candidate
        signal_msg = await client2.receive_message()
        if (signal_msg and 
            signal_msg.get("type") == "webrtc_signal" and 
            signal_msg.get("from_user") == user1_id and 
            signal_msg.get("signal", {}).get("type") == "ice-candidate"):
            test_result.add_success("ICE candidate correctly relayed")
        else:
            test_result.add_failure("ICE candidate not correctly relayed")
        
        # Clean up
        await client1.disconnect()
        await client2.disconnect()
    else:
        test_result.add_failure("Failed to connect both users")
    
    return test_result

async def test_next_partner():
    """Test next_partner functionality"""
    test_result = TestResult()
    
    # Create three users
    user1_id = str(uuid.uuid4())
    user2_id = str(uuid.uuid4())
    user3_id = str(uuid.uuid4())
    client1 = WebSocketClient(user1_id)
    client2 = WebSocketClient(user2_id)
    client3 = WebSocketClient(user3_id)
    
    # Connect all users
    if await client1.connect() and await client2.connect() and await client3.connect():
        # Skip welcome messages
        await client1.receive_message()
        await client2.receive_message()
        await client3.receive_message()
        
        # Match user1 and user2
        await client1.start_search()
        await client1.receive_message()  # searching message
        await client2.start_search()
        
        # Both users should receive match_found messages
        await client1.receive_message()
        await client2.receive_message()
        
        # User3 starts searching but won't match yet
        await client3.start_search()
        await client3.receive_message()  # searching message
        
        # User1 requests next partner
        await client1.next_partner()
        
        # User2 should receive partner_disconnected message
        disconnect_msg = await client2.receive_message()
        if disconnect_msg and disconnect_msg.get("type") == "partner_disconnected":
            test_result.add_success("User2 received partner disconnection notification")
        else:
            test_result.add_failure("User2 did not receive partner disconnection notification")
        
        # User1 should receive searching message
        searching_msg = await client1.receive_message()
        if searching_msg and searching_msg.get("type") == "searching":
            test_result.add_success("User1 returned to searching state")
        else:
            test_result.add_failure("User1 did not return to searching state")
        
        # User1 and User3 should match
        match_msg1 = await client1.receive_message()
        match_msg3 = await client3.receive_message()
        
        if (match_msg1 and match_msg1.get("type") == "match_found" and 
            match_msg3 and match_msg3.get("type") == "match_found"):
            test_result.add_success("User1 successfully matched with User3")
            
            # Verify partner IDs are correct
            if (match_msg1.get("partner_id") == user3_id and 
                match_msg3.get("partner_id") == user1_id):
                test_result.add_success("New partner IDs correctly assigned")
            else:
                test_result.add_failure("New partner IDs incorrectly assigned")
        else:
            test_result.add_failure("User1 and User3 not matched properly")
        
        # Clean up
        await client1.disconnect()
        await client2.disconnect()
        await client3.disconnect()
    else:
        test_result.add_failure("Failed to connect all users")
    
    return test_result

async def test_disconnection_handling():
    """Test handling of user disconnection during active chat"""
    test_result = TestResult()
    
    # Create two users
    user1_id = str(uuid.uuid4())
    user2_id = str(uuid.uuid4())
    client1 = WebSocketClient(user1_id)
    client2 = WebSocketClient(user2_id)
    
    # Connect both users
    if await client1.connect() and await client2.connect():
        # Skip welcome messages
        await client1.receive_message()
        await client2.receive_message()
        
        # Match users
        await client1.start_search()
        await client1.receive_message()  # searching message
        await client2.start_search()
        
        # Both users should receive match_found messages
        await client1.receive_message()
        await client2.receive_message()
        
        # Simulate user1 disconnecting
        await client1.disconnect()
        
        # User2 should receive partner_disconnected message
        disconnect_msg = await client2.receive_message()
        if disconnect_msg and disconnect_msg.get("type") == "partner_disconnected":
            test_result.add_success("Partner disconnection handled correctly")
        else:
            test_result.add_failure("Partner disconnection not handled correctly")
        
        # Clean up
        await client2.disconnect()
    else:
        test_result.add_failure("Failed to connect both users")
    
    return test_result

async def test_api_endpoints():
    """Test API endpoints"""
    test_result = TestResult()
    
    # Test root endpoint
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200 and "message" in response.json():
            test_result.add_success("Root API endpoint working")
        else:
            test_result.add_failure(f"Root API endpoint not working: {response.status_code}")
    except Exception as e:
        test_result.add_failure(f"Error accessing root API endpoint: {e}")
    
    # Test stats endpoint
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            if all(key in data for key in ["active_connections", "waiting_queue", "active_rooms"]):
                test_result.add_success("Stats API endpoint working")
            else:
                test_result.add_failure("Stats API endpoint missing expected fields")
        else:
            test_result.add_failure(f"Stats API endpoint not working: {response.status_code}")
    except Exception as e:
        test_result.add_failure(f"Error accessing stats API endpoint: {e}")
    
    return test_result

async def test_multiple_users():
    """Test multiple users searching simultaneously"""
    test_result = TestResult()
    num_users = 6  # Create 6 users (3 pairs)
    clients = []
    
    # Create and connect all users
    for i in range(num_users):
        user_id = str(uuid.uuid4())
        client = WebSocketClient(user_id)
        if await client.connect():
            await client.receive_message()  # Skip welcome message
            clients.append(client)
        else:
            test_result.add_failure(f"Failed to connect user {i+1}")
    
    if len(clients) < num_users:
        test_result.add_failure(f"Only {len(clients)} of {num_users} users connected")
        # Clean up connected clients
        for client in clients:
            await client.disconnect()
        return test_result
    
    # Start search for all users
    for client in clients:
        await client.start_search()
    
    # All users should receive either searching or match_found messages
    matched_users = 0
    for client in clients:
        msg = await client.receive_message()
        if msg and (msg.get("type") == "searching" or msg.get("type") == "match_found"):
            if msg.get("type") == "match_found":
                matched_users += 1
        else:
            test_result.add_failure(f"User {client.user_id} did not receive proper search status")
    
    # Wait for remaining match_found messages
    for client in clients:
        if client.partner_id is None:  # This client hasn't been matched yet
            msg = await client.receive_message()
            if msg and msg.get("type") == "match_found":
                matched_users += 1
    
    # We should have all users matched (each match counts as 2 users)
    if matched_users == num_users:
        test_result.add_success(f"All {num_users} users matched successfully")
    else:
        test_result.add_failure(f"Only {matched_users} of {num_users} users matched")
    
    # Clean up
    for client in clients:
        await client.disconnect()
    
    return test_result

async def run_all_tests():
    """Run all tests and print results"""
    print("\n=== Running WebSocket Connection Tests ===")
    connection_result = await test_websocket_connection()
    connection_result.print_results()
    
    print("\n=== Running User Matching Tests ===")
    matching_result = await test_user_matching()
    matching_result.print_results()
    
    print("\n=== Running WebRTC Signaling Tests ===")
    signaling_result = await test_webrtc_signaling()
    signaling_result.print_results()
    
    print("\n=== Running Next Partner Tests ===")
    next_partner_result = await test_next_partner()
    next_partner_result.print_results()
    
    print("\n=== Running Disconnection Handling Tests ===")
    disconnection_result = await test_disconnection_handling()
    disconnection_result.print_results()
    
    print("\n=== Running API Endpoint Tests ===")
    api_result = await test_api_endpoints()
    api_result.print_results()
    
    print("\n=== Running Multiple Users Tests ===")
    multiple_users_result = await test_multiple_users()
    multiple_users_result.print_results()
    
    # Overall result
    overall_success = all([
        connection_result.success,
        matching_result.success,
        signaling_result.success,
        next_partner_result.success,
        disconnection_result.success,
        api_result.success,
        multiple_users_result.success
    ])
    
    print("\n=== OVERALL TEST RESULTS ===")
    print(f"WebSocket Connection: {'✅' if connection_result.success else '❌'}")
    print(f"User Matching: {'✅' if matching_result.success else '❌'}")
    print(f"WebRTC Signaling: {'✅' if signaling_result.success else '❌'}")
    print(f"Next Partner: {'✅' if next_partner_result.success else '❌'}")
    print(f"Disconnection Handling: {'✅' if disconnection_result.success else '❌'}")
    print(f"API Endpoints: {'✅' if api_result.success else '❌'}")
    print(f"Multiple Users: {'✅' if multiple_users_result.success else '❌'}")
    print(f"\nOverall Test Result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    return overall_success

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import websockets
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
        import websockets
    
    # Run the tests
    asyncio.run(run_all_tests())