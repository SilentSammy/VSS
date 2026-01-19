import numpy as np
import asyncio
from bleak import BleakScanner, BleakClient
from threading import Thread
import socket
import struct
import json

def to_byte(val):
    """Convert -1.0 to +1.0 range to 0-255 byte using two's complement"""
    signed = max(-128, min(127, int(val * 128.0)))
    return signed & 0xFF

def merge_proportional(cmd_primary, cmd_secondary):
    # Primary command overrides secondary based on its magnitude
    cmd_final = {}
    
    # Handle all axes from both commands
    all_axes = set(cmd_primary.keys()) | set(cmd_secondary.keys())
    
    for axis in all_axes:
        primary_input = cmd_primary.get(axis, 0.0)    # Default to 0 if missing
        secondary_input = cmd_secondary.get(axis, 0.0) # Default to 0 if missing
        
        if abs(primary_input) < 0.05:  # No primary input
            cmd_final[axis] = secondary_input
        else:
            # Primary input interpolates between secondary and desired value
            # abs(primary_input) determines how much override (0 to 1)
            # sign(primary_input) determines direction
            override_strength = abs(primary_input)
            desired_value = 1.0 if primary_input > 0 else -1.0
            cmd_final[axis] = (1 - override_strength) * secondary_input + override_strength * desired_value
    
    return cmd_final

def get_user_cmd():
    import combined_input as inp
    slow = 0.5
    fast = 1.0
    scale = fast if inp.is_pressed('c') else slow  # 'C' key for full speed
    return {
        'x': inp.get_bipolar_ctrl('w', 's', 'LY') * scale,
        'y': inp.get_bipolar_ctrl('d', 'a', 'LX') * scale,
        'w': inp.get_bipolar_ctrl('e', 'q', 'RX') * scale
    }

def get_manual_override(cmd):
    user_cmd = get_user_cmd()
    return merge_proportional(user_cmd, cmd)

class MecanumBLEClient:
    def __init__(self, device_name="MP_BLE_Device", resolution=0.05):
        self.device_name = device_name
        self.device = None
        self.client = None
        
        # Velocity control fields
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        
        # Control resolution - round commands to nearest multiple
        self.resolution = resolution
        
        # UUID for combined velocity characteristic
        self.uuid_velocity = "12345678-1234-5678-1234-56789abcdef1"
        
        # Cache last sent values to avoid redundant BLE writes
        self._cached_velocity = None
        
        # Track pending non-blocking writes and queued values
        self._pending = None
        self._queued = None
        
        # Start background event loop
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
    
    # Internal async methods
    async def _async_find_device(self):
        """Scan for and return the target BLE device"""
        print(f"Scanning for {self.device_name}...")
        devices = await BleakScanner.discover(timeout=10.0)
        device = next((d for d in devices if d.name == self.device_name), None)
        if not device:
            print(f"✗ Device not found")
            return None
        print(f"✓ Found at {device.address}")
        return device
    
    async def _async_connect(self):
        """Connect to the BLE device"""
        self.device = await self._async_find_device()
        if not self.device:
            raise Exception("Device not found")
        
        print(f"Connecting...")
        self.client = BleakClient(self.device.address)
        await self.client.connect()
        if not self.client.is_connected:
            raise Exception("Connection failed")
        print(f"✓ Connected\n")
    
    async def _async_disconnect(self):
        """Disconnect from the BLE device"""
        if self.client:
            await self.client.disconnect()
            print("Disconnected.")
    
    async def _async_send_velocity(self, x, y, w, force=False):
        """Send velocity command (3 bytes: x, y, w)"""
        velocity_bytes = (to_byte(x), to_byte(y), to_byte(w))
        
        # Check cache to avoid redundant writes (unless forced)
        if not force and self._cached_velocity == velocity_bytes:
            return  # No change, skip write
        
        await self.client.write_gatt_char(self.uuid_velocity, bytes(velocity_bytes))
        self._cached_velocity = velocity_bytes
    
    # Public synchronous methods
    def connect(self):
        """Connect to the BLE device (blocking)"""
        future = asyncio.run_coroutine_threadsafe(self._async_connect(), self.loop)
        future.result()
    
    def disconnect(self):
        """Disconnect from the BLE device (blocking)"""
        future = asyncio.run_coroutine_threadsafe(self._async_disconnect(), self.loop)
        future.result()
    
    def send(self, force=False):
        """Send current velocity fields (x, y, w) to robot (non-blocking, queues if busy)
        
        Args:
            force: If True, bypasses cache and forces BLE write even if values unchanged
        """
        velocity_tuple = (self.x, self.y, self.w, force)
        
        if self._pending is None or self._pending.done():
            # Not busy - start new write
            self._pending = asyncio.run_coroutine_threadsafe(
                self._async_send_velocity(*velocity_tuple), 
                self.loop
            )
            # Add callback to process queued value when done
            self._pending.add_done_callback(lambda f: self._on_send_complete())
        else:
            # Busy - queue this value (replaces any previous queued value)
            self._queued = velocity_tuple
    
    def set_velocity(self, velocity, force=False):
        """Set velocity from dictionary and send command
        
        Args:
            velocity: Dictionary with keys 'x', 'y', 'w' (values -1.0 to 1.0)
            force: If True, bypasses cache and forces BLE write even if values unchanged
        """
        # Update internal state with rounding
        self.x = round(velocity.get('x', 0.0) / self.resolution) * self.resolution
        self.y = round(velocity.get('y', 0.0) / self.resolution) * self.resolution
        self.w = round(velocity.get('w', 0.0) / self.resolution) * self.resolution
        
        self.send(force=force)
    
    def _on_send_complete(self):
        """Callback when send completes - send queued values if exist"""
        if self._queued is not None:
            x, y, w, force = self._queued
            self._queued = None  # Clear queue
            # Update fields and send
            self.x = x
            self.y = y
            self.w = w
            self.send(force=force)
    
    def stop(self):
        """Stop all motors (blocking)"""
        future = asyncio.run_coroutine_threadsafe(
            self._async_send_velocity(0.0, 0.0, 0.0, force=True), 
            self.loop
        )
        future.result()
        # Update internal state
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0

class MecanumSocketClient:
    """TCP socket client for sending commands to BLE proxy server"""
    
    def __init__(self, host="localhost", port=5000, resolution=0.05):
        self.host = host
        self.port = port
        self.resolution = resolution
        self.sock = None
        
        # Velocity control fields
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
    
    def connect(self):
        """Connect to the proxy server (blocking)"""
        print(f"Connecting to proxy at {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print("✓ Connected to proxy\n")
    
    def disconnect(self):
        """Disconnect from the proxy server"""
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Disconnected from proxy.")
    
    def send(self, force=False):
        """Send current velocity fields (x, y, w) to proxy server
        
        Args:
            force: If True, forces send even if values unchanged
        """
        if not self.sock:
            raise Exception("Not connected to proxy server")
        
        # Send as JSON for simplicity
        cmd = {
            'x': self.x,
            'y': self.y,
            'w': self.w,
            'force': force
        }
        msg = json.dumps(cmd) + '\n'
        self.sock.sendall(msg.encode('utf-8'))
    
    def set_velocity(self, velocity, force=False):
        """Set velocity from dictionary and send command
        
        Args:
            velocity: Dictionary with keys 'x', 'y', 'w' (values -1.0 to 1.0)
            force: If True, bypasses cache and forces send
        """
        # Update internal state with rounding
        self.x = round(velocity.get('x', 0.0) / self.resolution) * self.resolution
        self.y = round(velocity.get('y', 0.0) / self.resolution) * self.resolution
        self.w = round(velocity.get('w', 0.0) / self.resolution) * self.resolution
        
        self.send(force=force)
    
    def stop(self):
        """Stop all motors"""
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.send(force=True)

class MecanumClientAuto:
    """Auto-fallback client: tries socket proxy first, then direct BLE"""
    
    def __init__(self, socket_client=None, ble_client=None):
        self.socket_client = socket_client or MecanumSocketClient()
        self.ble_client = ble_client or MecanumBLEClient()
        self.active_client = None
    
    def connect(self):
        """Try socket proxy first, fallback to direct BLE"""
        try:
            self.socket_client.connect()
            self.active_client = self.socket_client
            print("✓ Using proxy connection")
        except Exception as e:
            print(f"✗ Proxy unavailable ({e}), connecting directly to BLE...")
            self.ble_client.connect()
            self.active_client = self.ble_client
    
    def disconnect(self):
        """Disconnect from active client"""
        if self.active_client:
            self.active_client.disconnect()
    
    def set_velocity(self, velocity, force=False):
        """Send velocity command through active client"""
        if self.active_client:
            self.active_client.set_velocity(velocity, force)
    
    def stop(self):
        """Stop all motors"""
        if self.active_client:
            self.active_client.stop()

if __name__ == "__main__":
    import combined_input as inp
    import time
    import threading
    
    print("Mecanum BLE Hybrid Control Server")
    print("==================================\n")
    
    # Shared state for socket client connection
    current_client = None
    client_lock = threading.Lock()
    
    def socket_server_thread(port=5000):
        """Background thread that accepts socket connections"""
        global current_client
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('localhost', port))
        server.listen(1)
        print(f"Socket server listening on localhost:{port}")
        print("Waiting for connections...\n")
        
        try:
            while True:
                client_sock, addr = server.accept()
                with client_lock:
                    current_client = client_sock
                print(f"✓ Client connected from {addr} - Switching to REMOTE CONTROL")
                
                try:
                    # Keep connection alive until client disconnects
                    while True:
                        data = client_sock.recv(1)
                        if not data:
                            break
                except:
                    pass
                finally:
                    with client_lock:
                        current_client = None
                    client_sock.close()
                    print(f"✗ Client disconnected - Switching to MANUAL CONTROL")
        finally:
            server.close()
    
    def read_socket_command():
        """Read one JSON command from current socket client"""
        with client_lock:
            if not current_client:
                return None
            sock = current_client
        
        try:
            # Read until newline
            buffer = b''
            while b'\n' not in buffer:
                chunk = sock.recv(1024)
                if not chunk:
                    return None
                buffer += chunk
            
            # Parse JSON
            line = buffer.split(b'\n')[0]
            line = "{" + line.decode('utf-8').split('{',1)[1]  # Ensure valid JSON
            cmd = json.loads(line)
            return {
                'x': cmd.get('x', 0.0),
                'y': cmd.get('y', 0.0),
                'w': cmd.get('w', 0.0)
            }
        except:
            return None
    
    # Start socket server in background
    server_thread = threading.Thread(target=socket_server_thread, daemon=True)
    server_thread.start()
    
    # Connect to BLE device
    car = MecanumBLEClient()
    car.connect()
    
    try:
        print("Manual control active! Use WASD/gamepad to control.")
        print("  W/S or Left Stick Y: Forward/Backward (X-axis)")
        print("  A/D or Left Stick X: Strafe Left/Right (Y-axis)")
        print("  Q/E or Right Stick X: Rotate (W-axis)")
        print("  C: Full speed mode")
        print("\n")
        
        while True:
            # Check if we have a socket client
            with client_lock:
                has_client = current_client is not None
            
            if has_client:
                # Remote control mode - read from socket
                cmd = read_socket_command()
                if cmd is None:
                    # Connection lost or error - will be handled by server thread
                    time.sleep(0.02)
                    continue
            else:
                # Manual control mode - read from keyboard/gamepad
                cmd = get_user_cmd()
            
            # Send velocity command
            car.set_velocity(cmd)
            
            time.sleep(0.02)  # update rate
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        car.stop()
        car.disconnect()
