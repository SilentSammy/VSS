# Dual motor driver test - PWM on direction pins, enable always HIGH
from machine import Pin, PWM
import time
import ble_server
from mecanum import MecanumCar

if __name__ == "__main__":
    print("Mecanum car BLE control starting...")
    
    # Create mecanum car
    car = MecanumCar()
    
    # Define BLE callbacks
    def on_x(data):
        """Handle X-axis (forward/backward) updates"""
        car.x = ble_server.to_bipolar(data[0])
    
    def on_y(data):
        """Handle Y-axis (strafe) updates"""
        car.y = ble_server.to_bipolar(data[0])
    
    def on_w(data):
        """Handle W-axis (rotation) updates"""
        car.w = ble_server.to_bipolar(data[0])
    
    # Register callbacks with UUIDs
    ble_server.control_callbacks = {
        '12345678-1234-5678-1234-56789abcdef1': on_x,  # X-axis (forward/backward)
        '12345678-1234-5678-1234-56789abcdef2': on_y,  # Y-axis (strafe)
        '12345678-1234-5678-1234-56789abcdef3': on_w,  # W-axis (rotation)
    }
    
    # Start BLE server
    ble_server.start()
    
    print("\nBLE control active. Use BLE client to control the car.")
    print("Characteristics:")
    print("  X-axis (fwd/back):   ...def1")
    print("  Y-axis (strafe):     ...def2")
    print("  W-axis (rotation):   ...def3")
    
