import machine
from machine import Pin, PWM

class Car:
    def __init__(self, ena_pin=5, in1_pin=4, in2_pin=0, enb_pin=14, in3_pin=12, in4_pin=13):
        self.min_freq = 5   # Minimum PWM frequency at low speeds
        self.max_freq = 100  # Maximum PWM frequency at full throttle

        # Motor A (Left)
        self.ENA = PWM(Pin(ena_pin), freq=20, duty=0)
        self.IN1 = Pin(in1_pin, Pin.OUT)
        self.IN2 = Pin(in2_pin, Pin.OUT)
        self.MtrA = (self.ENA, self.IN1, self.IN2)

        # Motor B (Right)
        self.ENB = PWM(Pin(enb_pin), freq=20, duty=0)
        self.IN3 = Pin(in3_pin, Pin.OUT)
        self.IN4 = Pin(in4_pin, Pin.OUT)
        self.MtrB = (self.ENB, self.IN3, self.IN4)

        # Control variables
        self._throttle = 0
        self._steering = 0
        self.turn_strength = 1  # Adjust how sharp turns are

        # Initialize both motors to stop
        self.update_motors()

    @property
    def throttle(self):
        return self._throttle

    @throttle.setter
    def throttle(self, value):
        """ Set overall throttle (-1 to 1) """
        self._throttle = max(-1, min(1, value))  # Clamp between -1 and 1
        self.update_motors()

    @property
    def steering(self):
        return self._steering

    @steering.setter
    def steering(self, value):
        """ Set steering value (-1 to 1) """
        self._steering = max(-1, min(1, value))  # Clamp between -1 and 1
        self.update_motors()

    def update_motors(self):
        """ Compute and apply motor speeds based on throttle and steering, smoothly transitioning between movement and in-place turning. """
        left_speed = self._throttle * (1 - abs(self._steering)) + self._steering
        right_speed = self._throttle * (1 - abs(self._steering)) - self._steering

        # Clamp values to -1 to 1 range
        left_speed = max(-1, min(1, left_speed))
        right_speed = max(-1, min(1, right_speed))

        # Apply speeds to motors
        self.update_motor(self.MtrA, left_speed)
        self.update_motor(self.MtrB, right_speed)

    def update_motor(self, motor, throttle):
        """ Update motor speed and mode based on throttle """
        speed = abs(throttle)
        mode = 0 if throttle == 0 else (1 if throttle > 0 else 2)  # 0 = Stop, 1 = Forward, 2 = Reverse

        # Direct multiplication scaling, clamped to min_freq
        new_freq = max(self.min_freq, int(self.max_freq * speed))
        motor[0].freq(new_freq)  # Dynamically update PWM frequency

        set_mode(mode, motor)
        set_speed(speed, motor)

        print("left" if motor == self.MtrA else "right", "speed:", speed, "mode:", mode, "freq:", new_freq)

    def stop(self):
        """ Stop both motors """
        self._throttle = 0
        self._steering = 0
        self.update_motors()

def set_mode(mode, motor):
    """ Set motor direction """
    _, IN1, IN2 = motor
    IN1.value(mode & 1)
    IN2.value(mode >> 1)

def set_speed(speed, motor):
    """ Set motor speed (0 to 1 scaled to 1023 PWM) """
    motor[0].duty(int(min(speed * 1023, 1023)))

car = Car()