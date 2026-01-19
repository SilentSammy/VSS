from mecanum_client import MecanumSocketClient, MecanumBLEClient, MecanumClientAuto
from combined_input import is_pressed

client = MecanumSocketClient()
client.connect()

print("Client connected")

while True:
    if is_pressed('w'):
        client.set_velocity({'x': 0.5})
    if is_pressed('x'):
        client.set_velocity({'x': 0})
    
