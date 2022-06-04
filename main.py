"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.
    Drive
    -----
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    Toggle Reverse
    --------------
    Q            : reverse
    E            : drive

    R            : restart level
"""

from __future__ import print_function

import pickle
import queue
import threading
import uuid
import argparse
import logging
import time

import carla.client
import lane_detection
import cv2

from timer import SecInterval

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_k
    from pygame.locals import K_l
    from pygame.locals import K_c
    from pygame.locals import K_t
    from pygame.locals import K_y
    from pygame.locals import K_e
    from pygame.locals import K_g

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter, sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

from wheel import *

# import selfdrive
# from selfdrive.image.filters import CROP_SPEEDLIMIT_AREA
# from selfdrive.image.editor import ImageEditor
# from selfdrive.image.buffer import ImageBuffer

from disnet.client import Admin
from disnet.job import Job

from collections import namedtuple

Program_Control = namedtuple("Program_Control", ["up"])

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
IS_ONLINE = False
if IS_ONLINE:
    ADMIN = Admin(host="127.0.0.1")

'''
area_filter = CROP_SPEEDLIMIT_AREA
area_filter.cords = (
    (0, 0), (50, 0), (100, 0), (150, 0), (200, 0), (250, 0), (300, 0), (350, 0), (400, 0),
    (0, 50), (50, 50), (100, 50), (150, 50), (200, 50), (250, 50), (300, 50), (350, 50), (400, 50),
    (0, 100), (50, 100), (100, 100), (150, 100), (200, 100), (250, 100), (300, 100), (350, 100), (400, 100),
    (0, 150), (50, 150), (100, 150), (150, 150), (200, 150), (250, 150), (300, 150), (350, 150), (400, 150),
    (0, 200), (50, 200), (100, 200), (150, 200), (200, 200), (250, 200), (300, 200), (350, 200), (400, 200),
)
'''


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=30,
        WeatherId=1,
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, -15, 0.0)

    camera1 = sensor.Camera('side')
    camera1.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera1.set_position(5.0, 0.0, 1)
    camera1.set_rotation(-20, -90, 0)

    settings.add_sensor(camera0)
    settings.add_sensor(camera1)
    if args.lidar:
        lidar = sensor.Lidar('Lidar32')
        lidar.set_position(0, 0, 2.5)
        lidar.set_rotation(0, 0, 0)
        lidar.set(
            Channels=32,
            Range=50,
            PointsPerSecond=100000,
            RotationFrequency=10,
            UpperFovLimit=10,
            LowerFovLimit=-30)
        settings.add_sensor(lidar)
    return settings


class Timer(object):
    def __init__(self):
        """
        Constructor for the timer class.
        """
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class Interval:
    def __init__(self, func, ms=0, s=0, m=0):
        self.interval = s + m * 60 + ms / 1000
        self.func = func
        self.s_time = time.time()

    def try_calling(self):
        if time.time() - self.s_time >= self.interval:
            self.func()
            self.s_time = time.time()


'''
class SpeedLimitEditor(ImageEditor):
    """
    SpeedLimit editor that inherits from the base editor.
    It uses specific cropping settings for circle detection.
    """

    def __init__(self):
        """
        Constructor for the speedlimit editor.
        """
        super().__init__()
        self.circle_crop = selfdrive.image.editor.CIRCLE_CROP(
            minDist=6, dp=1.1, param1=150,
            param2=50, minRadius=20, maxRadius=50
        )
'''


class CarlaGame(object):
    def __init__(self, carla_client: carla.client.CarlaClient, args):
        """
        Carla settings
        --------------
        """
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._side_image = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        """
        Self driving car settings
        ------------------------
        """
        self.num_detections = 0
        self.timer = SecInterval()
        self.timer.start()
        self.on_capture = False  # capture mode flag
        # self.im_buff = ImageBuffer("data")  # image buffer for data saving
        self.on_auto = False  # toggle auto-pilot
        # self.editor = SpeedLimitEditor()  # speedlimit image editor for preprocessing
        self.player_measurements = None
        self.max_speed = 25
        self.is_online = IS_ONLINE
        self.required_jobs = ["speedlimit"]
        if self.is_online:
            self.admin = ADMIN
            self.admin.connect(supported_jobs=self.required_jobs)
        else:
            self.admin = None
        self.jobs_queue = []
        self.sl_jobs = 0
        self.image_queue = queue.Queue()
        self.frame = 0
        self.wheel = Wheel()
        self.adjust_index = 0
        self.adjust_direction = ""

        self.control = [0, 0, 0]
        self.control_wait = False
        self.last_adjust = None
        self.trigger_adjust = False

        self.level = 0
        self.down = False

        frameSize = (1600, 900)

        self.export = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
        self.released = False

        self.print_steer = False
        self.last_turn = 0
        self.on_turn = False

        self.wheel_set_target = False

        self.maintain_on_turn = False

        def steer_back():
            if self.wheel.steer > 0:
                self.wheel.right(self.wheel.steer - 0.1)
                if self.wheel.steer < 0:
                    self.wheel.steer = 0
            elif self.wheel.steer < 0:
                self.wheel.left(self.wheel.steer + 0.1)
                if self.wheel.steer > 0:
                    self.wheel.steer = 0

        self.steer_back_interval = Interval(func=steer_back, ms=100)

        self.adjust_multiply = 1

        for _ in range(4):
            threading.Thread(target=self._process_images).start()
        threading.Thread(target=self._share_jobs).start()
        threading.Thread(target=self.adjust).start()

    """
    Carla client functions
    1. execute
    2. _initialize_game
    3. _on_new_episode
    4. _on_loop
    5. _get_keyboard_control
    6. _on_render
    """

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        """Display pygame window"""
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int(
                    (WINDOW_HEIGHT / float(self._map.map_image.shape[0])) * self._map.map_image.shape[1]),
                 WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        """episode init"""
        self._carla_settings.randomize_seeds()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):
        """Main loop for carla client"""
        self._timer.tick()
        measurements, sensor_data = None, None
        try:
            measurements, sensor_data = self.client.read_data()
            self.player_measurements = measurements.player_measurements
            self._main_image = sensor_data.get('CameraRGB', None)
            self._side_image = sensor_data.get('side', None)
            self._lidar_measurement = sensor_data.get('Lidar32', None)
        except RuntimeError:
            # workarounds for carla 8.0.2, as there's no support for python 3.7
            pass
        if measurements is not None and sensor_data is not None:
            control = self._get_keyboard_control(pygame.key.get_pressed())
            # if self.on_capture:
            #   control = VehicleControl()
            #  control.throttle = 0.25
            if self._city_name is not None:
                self._position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                self._agent_positions = measurements.non_player_agents
            if control is None:
                self._on_new_episode()
            else:
                self.client.send_control(control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            if not self.released:
                print("Export")
                self.released = True
                self.export.release()
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            self.wheel.left(.7)
        elif keys[K_RIGHT] or keys[K_d]:
            self.wheel.right(.7)
        elif not self.on_auto:
            self.wheel.steer = 0
            # self.wheel.release()
        if keys[K_UP] or keys[K_w]:
            self.wheel.throttle = 1
        elif keys[K_DOWN] or keys[K_s]:
            self.wheel.brake(1.0)
        elif not self.on_auto:
            self.wheel.throttle = 0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = True
        if keys[K_e]:
            self._is_on_reverse = False
        if keys[K_p]:
            '''
            self._enable_autopilot = not self._enable_autopilot'''
            self.wheel.release()
        if keys[K_k]:
            self.on_capture = True
            print("[ON] capture mode")
        if keys[K_l]:
            self.on_capture = False
            print("[OFF] capture mode")
        if keys[K_c]:
            print("ADJUST")
            self.adjust_direction = "left"
            self.trigger_adjust = True
            '''
            if not self.on_capture:
                pass
                #self.im_buff.save_to_disk("data")
            else:
                print("Cannot save data while on capture mode.")'''
        if keys[K_t]:
            self.on_auto = True
            self.max_speed = 30
            print("[ON] auto pilot")
        if keys[K_y]:
            self.max_speed = 70
            self.on_auto = False
            self.control = [0, 0, 0]
            print("[OFF] auto pilot")

        if True:
            values = self.wheel.get_values()
            control.steer = values.steer
            control.throttle = self.wheel.throttle
            control.brake = values.brake
        # self.wheel.throttle = 0
        self.wheel.brake_val = 0

        '''
        if self.control[0] > 0:
            if self._get_speed() < self.max_speed:
                control.throttle = self.control[0]
            else:
                control.throttle = 0.2

        if self.control[1] != 0:
            control.steer = self.control[1]

        if self.control[2] != 0:
            control.brake = self.control[2]'''

        control.reverse = self._is_on_reverse
        # control.throttle = 0.7
        return control

    """
    Self-driving functions
    (Ori David - orid2004@gmail.com)
    """

    def _get_speed(self):
        """
        :return: Current speed
        """
        if self.player_measurements is None:
            return 0
        return int(self.player_measurements.forward_speed * 3.6)

    def _share_jobs(self):
        """
        Shares jobs with the server.
        This thread uses admin's put_jobs(jobs).
        :return: None
        """
        while True:
            try:
                while len(self.jobs_queue) > 0:
                    self.admin.put_jobs(self.jobs_queue)
                    self.jobs_queue.clear()
            except Exception as e:
                print(e)
            time.sleep(0.1)

    def _process_images(self):
        """
        Calls detection function for each frame.
        As carla's mainloop loop must work smoothly, no detection can
        be done their. Therefore, a queue is shared between the threads.
        :return: None
        """
        while True:
            while self.image_queue.qsize() > 0:
                self._handle_speedlimit_detection(self.image_queue.get())
            time.sleep(0.05)

    def _handle_speedlimit_detection(self, image_np):
        """
        Receives potential frames.
        :param image_np: frame as numpy array
        :return: None
        """
        if not self.is_online:
            return
        '''
        self.editor.img = image_np  # loads image into editor
        self.editor.implement_filter(selfdrive.image.filters.CROP_SPEEDLIMIT_SCREEN)  # crops the image
        # crop_search_areas returns potential arrays that may contain a speedlimit sign.
        dataset = self.editor.crop_search_areas(
            size=area_filter.size,
            cords=area_filter.cords,
            function=self.editor.crop_circle_box
        )
        # jobs are added to the jobs_queue for each array
        # Limit is set to 50 - This is required to prevent Carla from freezing.
        for (im, _) in dataset:
            if self.sl_jobs < 50:
                args_key = str(uuid.uuid1())  # job's unique id
                self.admin.mc_writer.set(args_key, pickle.dumps((im,)))  # load image to memcache
                self.sl_jobs += 1  # increase count
                self.jobs_queue.append(
                    Job(
                        type="speedlimit",
                        id=f"sl{args_key}",
                        args=args_key
                    )
                )
        # Resets the timer
        if self.timer.sec_loop():
            self.sl_jobs = 0
            self.timer.start()'''

    def adjust(self):
        while True:
            if self.trigger_adjust:
                mul = self.adjust_multiply
                self.last_turn = time.time()
                max_index = 5
                if self.adjust_index < max_index:
                    if self.adjust_direction == "left":
                        self.wheel.left(0.04 * mul)
                    else:
                        self.wheel.right(0.04 * mul)
                    self.adjust_index += 1
                elif max_index <= self.adjust_index < max_index * 2:
                    self.adjust_index += 1
                    if self.adjust_direction == "left":
                        self.wheel.right(0.015 * mul)
                    else:
                        self.wheel.left(0.015 * mul)
                else:
                    self.wheel.release()
                    self.adjust_index = 0
                    self.trigger_adjust = False
            time.sleep(0.2)

    def _steer_forward(self, value):
        if time.time() - self.last_turn < 0.5:
            return
        if abs(value) > 0.01:
            steer_max = 0.4 if self._get_speed() < 20 else 0.2
            if abs(value) < 0.025:
                steer_max *= 0.75
            if value > 0:
                score = value / 0.20
                func = self.wheel.right
                print('[right]', round(steer_max * score, 4))
            else:
                value = abs(value)
                score = value / 0.12
                func = self.wheel.left
                print('[left]', round(steer_max * score, 4))
            turn_value = steer_max * score
            if self.on_turn:
                turn_value = 0.4
            if turn_value > 0.5:
                return
            func(turn_value)

        else:
            self.wheel.right(0)

    def check_for_turns(self, non_zero_pixels):
        if non_zero_pixels > 750:
            #self.wheel.brake_mul = 2
            if not self.on_turn:
                self.on_turn = True
                self.wheel.steer_back_slowly = True
                print("Start Turn")
            self.max_speed = 8
        elif non_zero_pixels > 200:
            self.max_speed = 12
        elif non_zero_pixels > 100:
            self.wheel.brake_mul = 1
            self.max_speed = 20
        if 0 <= non_zero_pixels < 75:
            if self.on_turn:
                self.on_turn = False
                self.wheel.steer_back_slowly = False
                print("End RTuen")
            self.wheel.brake_target = 0
            self.max_speed = 30

        if non_zero_pixels > 100:
            #print("brake target to", self.max_speed)
            self.wheel.set_brake_target(self.max_speed)

    def auto_pilot_set_wheel(self, main_img, side_img):
        # An array is passed every 5 frames to prevent overflow
        if self.is_online:
            self.image_queue.put(np.array(main_img))

        blank_image = main_img.copy()
        blank_image = cv2.resize(blank_image, (1600, 900))
        image, lines = lane_detection.detect_lanes(
            image=cv2.cvtColor(np.asarray(side_img), cv2.COLOR_RGB2BGR),
            color=lane_detection.Yellow,
            reg=lane_detection.RegionsOfInterest.steer_adjust
        )
        side_img, steer_score = lane_detection.draw_lines_simple(
            img=image,
            lines=lines
        )
        cv2.imshow('side', side_img)

        image, lines = lane_detection.detect_lanes(
            cv2.cvtColor(np.asarray(main_img), cv2.COLOR_RGB2BGR), lane_detection.Yellow,
            lane_detection.RegionsOfInterest.turn_detection)
        img, good, left, right, score_left, score_right, count_zero = lane_detection.draw_lines(image,
                                                                                                lines)
        cv2.imshow('lanes', img)

        blank_image[0:600, 800:1600] = img
        blank_image[400:900, 0:800] = side_img[0:500, :]

        blank_image[0:600, 0:800] = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        blank_image = cv2.resize(blank_image, (1600, 900))
        self.export.write(blank_image)
        #cv2.imshow('blank', blank_image)
        if self.on_auto:
            self.check_for_turns(non_zero_pixels=count_zero)
            if good == -1:
                self.wheel.steer = 0
                self.control_wait = True
                return
            elif self.control_wait and (good > 2 or left > 2 or right > 2):
                self.control_wait = False
            if (right < 3 and left < 3) and (good == -1 or good >= 5):
                if self._get_speed() < self.max_speed - 5 and not self.on_turn:
                    if self.max_speed < 15:
                        value = 0.6
                    elif self.max_speed <= 20:
                        value = 0.65
                    else:
                        value = 0.7
                    #print("GAS!")
                    self.wheel.accelerate(value)
                if not self.on_turn:
                    if steer_score is not None:
                        self._steer_forward(steer_score)
                    else:
                        self.steer_back_interval.try_calling()
                self.maintain_on_turn = False
            elif not self.control_wait:
                if left > right:
                    func = self.wheel.left
                    direction = "left"
                    score = score_left
                else:
                    func = self.wheel.right
                    direction = "right"
                    score = score_right
                if score < 0.21:
                    if self._get_speed() < 10:
                        self.wheel.accelerate(0.6)
                    else:
                        self.wheel.accelerate(0.75)
                    if steer_score is not None and not self.control_wait:
                        self._steer_forward(steer_score)
                    self.maintain_on_turn = False
                else:
                    if score < 0.4:
                        steer_max = 0.15 if self._get_speed() < 20 else 0.075
                        #if self.on_turn:
                         #   steer_max *= 3
                        self.wheel.accelerate(0.6)
                        self.maintain_on_turn = False
                    elif score < 0.6:
                        steer_max = 0.2 if self._get_speed() < 20 else 0.1
                      #  if self.on_turn:
                       #     steer_max *= 3
                        if not self.maintain_on_turn:
                            self.wheel.accelerate(0.5)
                    else:
                        if score < 0.8:
                            steer_max = 0.6 if self._get_speed() < 20 else 0.3
                        else:
                            steer_max = 0.7 if self._get_speed() < 20 else 0.5

                        if self._get_speed() - self.max_speed > 2:
                            diff = self._get_speed() - self.max_speed
                            print("brake main")
                            self.wheel.brake(diff / 10)
                        else:
                            self.maintain_on_turn = True
                            self.wheel.accelerate(0.48)
                        #if self.on_turn and count_zero < 750:
                         #   steer_max *= 0.5
                    current_speed = self._get_speed()
                    if current_speed > 50:
                        current_speed = 50
                    if self._get_speed() <= 15:
                        steer_max *= 1.2
                    elif self._get_speed() <= 25:
                        steer_max *= 0.5
                    elif self._get_speed() <= 35:
                        steer_max *= 0.3
                    elif self._get_speed() <= 45:
                        steer_max *= 0.15
                    else:
                        steer_max *= 0.1
                    value = steer_max * score
                    if value <= 0:
                        value = 0.05
                    elif value > 0.65:
                        value = 0.65
                    #if value > 0.6:
                     #   self.wheel.steer_back_slowly = True
                    func(value)
                    print("turn", direction, round(value, 2), "score is", round(score, 2))
            if self._get_speed() - self.max_speed > 10:
                diff = self._get_speed() - self.max_speed
                print("BREAK!", diff / 20)
                self.wheel.brake(diff / 20)
            if self._get_speed() >= self.max_speed:
                self.wheel.set_maintain_settings()
                self.wheel.maintain_speed()
            if self.on_turn and self.wheel.steer < 0.4 and steer_score and steer_score > 0.4:
                self.wheel.steer = 0.45
                print("Turn assist ASSIST 0.4")

    def _on_render(self):
        """Process and display main image"""
        if self._main_image is not None and self._side_image is not None:
            """
            Pass main image to processing functions
            Features
            -----------------------------
            [1] speedlimit + ocr detection
            """

            self.wheel.forward_speed = self.player_measurements.forward_speed
            self.wheel.v = self._get_speed()
            self.wheel.max_speed = self.max_speed
            array = image_converter.to_rgb_array(self._main_image)
            side_array = image_converter.to_rgb_array(self._side_image)
            self.auto_pilot_set_wheel(
                main_img=array,
                side_img=side_array
            )
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        """
        Carla client view and measurements
        """
        if self._lidar_measurement is not None:
            lidar_data = np.array(self._lidar_measurement.data[:, :2])
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            # draw lidar
            lidar_img_size = (200, 200, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            surface = pygame.surfarray.make_surface(lidar_img)
            self._display.blit(surface, (10, 10))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]
            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            w_pos = int(self._position[0] * (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
            h_pos = int(self._position[1] * (new_window_width / float(self._map_shape[1])))
            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])
                    w_pos = int(agent_position[0] * (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
                    h_pos = int(agent_position[1] * (new_window_width / float(self._map_shape[1])))
                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)
            self._display.blit(surface, (WINDOW_WIDTH, 0))
        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    parser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    parser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    parser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    parser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    parser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = parser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)
    while True:
        try:
            with make_carla_client(args.host, args.port) as client:
                print("Connecting to:", args.host)
                game = CarlaGame(client, args)
                game.execute()
                break
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
