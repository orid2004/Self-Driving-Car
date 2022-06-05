import math
import threading
import time
from collections import namedtuple

Values = namedtuple("Values", ["steer", "throttle", "brake"])


class Wheel:
    def __init__(self):
        self.steer_target = 0  # between -1.0 1.0
        self.steer = 0  # between -1.0 1.0
        self.throttle = 0  # between 0 1.0
        self.brake_val = 0  # between 0 1.0
        self.v = 0
        self.turn = False
        self.forward_speed = 0
        self.max_speed = 0
        self.on_release = False
        self.on_gas_release = False
        self.on_gas = False
        self.last_max = 0
        self.maintain_start = 0
        self.next_interval = 1

        self.throttle_target = 0

        self.steer_back_slowly = False
        self.steer_target = 0

        self.brake_target = 0

        self.brake_mul = 1

        threading.Thread(target=self._gas).start()
        threading.Thread(target=self._update_steer).start()
        threading.Thread(target=self._brake_to).start()

    def right(self, value):
        self.turn = True
        if self.steer < value or not self.steer_back_slowly:
            self.steer_back_slowly = False
            self.steer_target = 0
            self.steer = value
        else:
            self.steer_target = value

    def left(self, value):
        self.turn = True
        if self.steer > value or not self.steer_back_slowly:
            self.steer_back_slowly = False
            self.steer_target = 0
            self.steer = value * -1
        else:
            self.steer_target = value * -1

    def _update_steer(self):
        while True:
            if self.steer_target != 0:
                if self.steer < self.steer_target:
                    value = 0.1 if self.steer > 0 else 0.02
                    self.steer += value
                    if self.steer > self.steer_target:
                        self.steer_back_slowly = False
                        self.steer = self.steer_target
                        print("Done Wb")
                else:
                    value = 0.1 if self.steer < 0 else 0.02
                    self.steer -= value
                    if self.steer < self.steer_target:
                        self.steer_back_slowly = False
                        self.steer = self.steer_target
                        print("Done wb")
                time.sleep(0.2)
            else:
                time.sleep(0.1)

    def brake(self, value, mul=1):
        self.brake_val = value
        self.brake_mul = mul
        self.last_max = 0
        self.throttle = 0
        self.throttle_target = 0

    def accelerate(self, value):
        self.throttle_target = value
        self.last_max = 0

    def _gas(self):
        while True:
            if self.throttle < self.throttle_target:
                self.throttle += 0.075
                if self.throttle > self.throttle_target:
                    self.throttle = self.throttle_target
                #print("throt is", self.throttle)
            time.sleep(0.25)

    def set_brake_target(self, speed):
        self.brake_target = speed

    def _brake_to(self):
        while True:
            if self.brake_target != 0 and self.brake_target < (self.forward_speed * 3.6):
                speed = self.forward_speed * 3.6
                diff = speed - self.brake_target
                val = 0
                if diff > 20:
                    val = 0.5
                elif diff > 15:
                    vall = 0.25
                elif diff > 10:
                    val = 0.15
                elif diff > 5:
                    val = diff / 100
                else:
                    val = diff / 200
                if self.brake_target < 15:
                    val *= 1.5
                self.brake(val)
                print("braking", val, "to", self.brake_target)
            time.sleep(0.2)

    def set_maintain_settings(self, max_speed=None):
        if self.last_max == 0 and self.forward_speed > 0.1:
            if not max_speed:
                speed = int(self.forward_speed * 3.6)
            else:
                speed = max_speed
            if speed > self.max_speed:
                speed = self.max_speed
            self.last_max = speed
            if speed <= 12:
                self.throttle = 0.45
            if speed <= 16:
                self.throttle = 0.5
            elif speed <= 22:
                self.throttle = 0.55
            elif speed <= 35:
                self.throttle = 0.6
            else:
                self.throttle = 0.7


    def maintain_speed(self):
        if time.time() - self.maintain_start > self.next_interval and self.last_max > 0:
            speed = int(self.forward_speed * 3.6)
            if abs(speed - self.last_max) > 20:
                self.brake(0.3)
                self.next_interval = 1
            elif abs(speed - self.last_max) > 10:
                if speed > self.last_max:
                    self.throttle -= 0.07 if self.last_max > 15 else 0.05
                else:
                    self.throttle += 0.07 if self.last_max > 15 else 0.05
                self.next_interval = 3 if self.last_max > 15 else 1
            elif abs(speed - self.last_max) > 5:
                if speed > self.last_max:
                    self.throttle -= 0.04
                else:
                    self.throttle += 0.04
                self.next_interval = 3 if self.last_max > 15 else 1

            self.maintain_start = time.time()
            '''
            self.maintain_start = time.time()
            speed = int(self.forward_speed * 3.6)
            diff = abs(speed - self.last_max)
            if self.last_max > 0 and speed > 0 and diff > 2:
                if diff > 10:
                    value = 0.1
                elif 5 < diff < 10:
                    value = 0.05
                else:
                    value = 0.025
                if speed > self.last_max:
                    print('down', speed, self.last_max)
                    self.throttle -= value
                    print('t now', self.throttle)
                else:
                    print('up', speed, self.last_max)
                    self.throttle += value
                    print('t now', self.throttle)'''

    def release(self):
        if self.steer != 0:
            self.turn = False
            if not self.on_release:
                print("RELEASE!")
                self.on_release = True
                threading.Thread(target=self._release).start()

    def has_values(self):
        return self.steer != 0 or self.throttle != 0 or self.brake != 0

    def get_values(self):
        return Values(steer=self.steer, throttle=self.throttle, brake=self.brake_val)

    def _release(self):
        while self.steer != 0 and not self.turn:
            """
            value = round(self.forward_speed / 10, 2)
            if self.steer > 0:
                value = value * -1
            self.steer += value
            if (self.steer < 0 and value < 0) or (self.steer > 0 and value > 0):
                self.steer = 0
                self.on_release = False
                return
            """
            if self.forward_speed > 0:
                value = 0.2
                if self.steer > 0:
                    value = -0.2
                self.steer += value
            time.sleep(0.1)
        self.on_release = False
