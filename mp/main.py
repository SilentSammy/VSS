# Dual motor driver test - PWM on direction pins, enable always HIGH
from machine import Pin, PWM
import time

# Motor pin assignments (PWM control)
MOTOR_PINS = [
    (3, 2),    # Motor 1
    (1, 0),    # Motor 2
    (9, 10),   # Motor 3
    (20, 21),  # Motor 4
]


class Motor:
    def __init__(self, pin1, pin2, freq=1000):
        """Initialize motor with two pins for H-bridge control."""
        self.pin1_num = pin1
        self.pin2_num = pin2
        self.pin1 = Pin(pin1, Pin.OUT)
        self.pin2 = Pin(pin2, Pin.OUT)
        self.pwm1 = None
        self.pwm2 = None
        self.freq = freq
        self.stop()
    
    def set_power(self, power):
        """Set motor power from -1 (full reverse) to 1 (full forward)."""
        # Clamp power to valid range
        power = max(-1.0, min(1.0, power))
        
        if power > 0:
            # Forward - PWM on pin1, pin2 LOW
            # Deinit pin2 PWM if active
            if self.pwm2:
                self.pwm2.deinit()
                self.pwm2 = None
                self.pin2 = Pin(self.pin2_num, Pin.OUT)
            self.pin2.value(0)
            
            # Create PWM on pin1 if not exists
            if not self.pwm1:
                self.pwm1 = PWM(self.pin1, freq=self.freq)
            
            duty = int(power * 65535)
            self.pwm1.duty_u16(duty)
            
        elif power < 0:
            # Reverse - PWM on pin2, pin1 LOW
            # Deinit pin1 PWM if active
            if self.pwm1:
                self.pwm1.deinit()
                self.pwm1 = None
                self.pin1 = Pin(self.pin1_num, Pin.OUT)
            self.pin1.value(0)
            
            # Create PWM on pin2 if not exists
            if not self.pwm2:
                self.pwm2 = PWM(self.pin2, freq=self.freq)
            
            duty = int(abs(power) * 65535)
            self.pwm2.duty_u16(duty)
            
        else:
            # Stop
            self.stop()
    
    def stop(self):
        """Stop the motor."""
        if self.pwm1:
            self.pwm1.deinit()
            self.pwm1 = None
            self.pin1 = Pin(self.pin1_num, Pin.OUT)
        if self.pwm2:
            self.pwm2.deinit()
            self.pwm2 = None
            self.pin2 = Pin(self.pin2_num, Pin.OUT)
        self.pin1.value(0)
        self.pin2.value(0)

class MecanumCar:
    def __init__(self, motor_fl=None, motor_fr=None, motor_bl=None, motor_br=None):
        """Initialize mecanum car with 4 motors (front-left, front-right, back-left, back-right)."""
        self.motor_fl = motor_fl if motor_fl else Motor(*MOTOR_PINS[0])
        self.motor_fr = motor_fr if motor_fr else Motor(*MOTOR_PINS[1])
        self.motor_bl = motor_bl if motor_bl else Motor(*MOTOR_PINS[2])
        self.motor_br = motor_br if motor_br else Motor(*MOTOR_PINS[3])
        self._x = 0.0
        self._y = 0.0
        self._w = 0.0
    
    @property
    def x(self):
        """Strafe velocity (-1 to 1, left to right)."""
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = max(-1.0, min(1.0, value))
        self._update_motors()
    
    @property
    def y(self):
        """Forward velocity (-1 to 1, backward to forward)."""
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = max(-1.0, min(1.0, value))
        self._update_motors()
    
    @property
    def w(self):
        """Rotation velocity (-1 to 1, counter-clockwise to clockwise)."""
        return self._w
    
    @w.setter
    def w(self, value):
        self._w = max(-1.0, min(1.0, value))
        self._update_motors()
    
    def _update_motors(self):
        """Apply mecanum kinematics to calculate and set motor speeds."""
        fl = self._y + self._x - self._w
        fr = self._y - self._x + self._w
        bl = self._y - self._x - self._w
        br = self._y + self._x + self._w
        
        # Normalize if any value exceeds ±1
        max_val = max(abs(fl), abs(fr), abs(bl), abs(br))
        if max_val > 1.0:
            fl /= max_val
            fr /= max_val
            bl /= max_val
            br /= max_val
        
        self.motor_fl.set_power(fl)
        self.motor_fr.set_power(fr)
        self.motor_bl.set_power(bl)
        self.motor_br.set_power(br)
    
    def stop(self):
        """Stop all motors."""
        self._x = 0.0
        self._y = 0.0
        self._w = 0.0
        self._update_motors()


if __name__ == "__main__":
    print("Mecanum car axis test starting...")
    
    # Create mecanum car
    car = MecanumCar()
    
    # Test Y-axis (forward/backward)
    print("\nTesting Y-axis (forward/backward)...")
    print("  Forward 50%...")
    car.y = 0.5
    time.sleep(2)
    
    print("  Stop...")
    car.stop()
    time.sleep(1)
    
    print("  Backward 50%...")
    car.y = -0.5
    time.sleep(2)
    
    print("  Stop...")
    car.stop()
    time.sleep(1)
    
    # Test X-axis (strafe left/right)
    print("\nTesting X-axis (strafe left/right)...")
    print("  Strafe right 50%...")
    car.x = 0.5
    time.sleep(2)
    
    print("  Stop...")
    car.stop()
    time.sleep(1)
    
    print("  Strafe left 50%...")
    car.x = -0.5
    time.sleep(2)
    
    print("  Stop...")
    car.stop()
    time.sleep(1)
    
    # Test W-axis (rotation)
    print("\nTesting W-axis (rotation)...")
    print("  Rotate clockwise 50%...")
    car.w = 0.5
    time.sleep(2)
    
    print("  Stop...")
    car.stop()
    time.sleep(1)
    
    print("  Rotate counter-clockwise 50%...")
    car.w = -0.5
    time.sleep(2)
    
    print("  Stop...")
    car.stop()
    
    print("\nAll tests complete!")
