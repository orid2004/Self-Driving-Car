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

import selfdrive
from selfdrive.image.filters import CROP_SPEEDLIMIT_AREA
from selfdrive.image.editor import ImageEditor
from selfdrive import navigation
from selfdrive import visualization_utils as vis_utils
from selfdrive import vehicle_control

from disnet.client import Admin
from disnet.job import Job

from collections import namedtuple

# Create and configure logger
logging.basicConfig(filename=f"CarlaLog.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

NavigationData = namedtuple("NavigationData", [
    "forward", "right_count", "right_score", "left_count", "left_score", "non_zero", "mean_slope"
])

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
IS_ONLINE = True
if IS_ONLINE:
    ADMIN = Admin(host="172.16.20.211")

area_filter = CROP_SPEEDLIMIT_AREA
area_filter.cords = (
    (0, 0), (50, 0), (100, 0), (150, 0), (200, 0), (250, 0), (300, 0), (350, 0), (400, 0),
    (0, 50), (50, 50), (100, 50), (150, 50), (200, 50), (250, 50), (300, 50), (350, 50), (400, 50),
    (0, 100), (50, 100), (100, 100), (150, 100), (200, 100), (250, 100), (300, 100), (350, 100), (400, 100),
    (0, 150), (50, 150), (100, 150), (150, 150), (200, 150), (250, 150), (300, 150), (350, 150), (400, 150),
    (0, 200), (50, 200), (100, 200), (150, 200), (200, 200), (250, 200), (300, 200), (350, 200), (400, 200),
)


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
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.next_log = ""
        self.timer = SecInterval()
        self.timer.start()
        self.on_auto = False  # toggle auto-pilot
        self.editor = SpeedLimitEditor()  # speedlimit image editor for preprocessing
        self.player_measurements = None
        self.speed_limit = 30
        self.max_speed = 30
        self.is_online = IS_ONLINE
        self.required_jobs = ["speedlimit"]
        if self.is_online:
            self.admin = ADMIN
            self.admin.connect(supported_jobs=self.required_jobs)
        else:
            self.admin = None

        self.jobs_queue = []
        self.sl_jobs = 0
        self.speed_limit_queue = queue.Queue()
        self.frame = 0

        self.control = vehicle_control.PhysicsControl()
        self.export = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 50, (1600, 900))
        self.last_turn = 0
        self.masked_output = None
        self.side_masked_output = None

        # Flags
        self.control_wait = False
        self.on_turn = False
        self.maintain_on_turn = False
        self.video_released = False

        def _steer_forward():
            if self.control.steer > 0:
                self.control.right(self.control.steer - 0.1)
                if self.control.steer < 0:
                    self.control.steer = 0
            elif self.control.steer < 0:
                self.control.left(self.control.steer + 0.1)
                if self.control.steer > 0:
                    self.control.steer = 0

        self.steer_back_interval = Interval(func=_steer_forward, ms=150)

        for _ in range(4):
            threading.Thread(target=self._process_images).start()
        threading.Thread(target=self._share_jobs).start()

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
            if not self.video_released:
                print("Video export.")
                self.video_released = True
                self.export.release()
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            self.control.left(.7)
        elif keys[K_RIGHT] or keys[K_d]:
            self.control.right(.7)
        elif not self.on_auto:
            self.control._steer = 0
        if keys[K_UP] or keys[K_w]:
            self.control._throttle = 1
        elif keys[K_DOWN] or keys[K_s]:
            self.control.brake(1.0)
        elif not self.on_auto:
            self.control._throttle = 0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = True
        if keys[K_e]:
            self._is_on_reverse = False
        if keys[K_t]:
            self.on_auto = True
            self.max_speed = 30
            self.logger.info("On Auto Pilot. Good Luck!")
            print("[ON] auto pilot")
        if keys[K_y]:
            self.max_speed = 70
            self.on_auto = False
            print("[OFF] auto pilot")

        control.steer, control.throttle, control.brake = self.control.get_values()
        control.reverse = self._is_on_reverse
        self.control.reset_brake_value()

        return control

    """
    Self-driving functions
    (Ori David - orid2004@gmail.com)
    """

    def _add_log(self, text):
        self.next_log += text + ' ' * 2 + '|' + ' ' * 2
        if len(self.next_log) > 100:
            self.logger.info(self.next_log)
            self.next_log = ""

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
                    self.jobs_queue = self.jobs_queue + self.jobs_queue.copy()
                    print("Sending", len(self.jobs_queue), "jobs to server...")
                    self.admin.put_jobs(self.jobs_queue)
                    self.jobs_queue.clear()
            except Exception as e:
                self._add_log(f'Exception-{e}')
            time.sleep(0.1)

    def _process_images(self):
        """
        Calls detection function for each frame.
        As carla's mainloop loop must work smoothly, no detection can
        be done their. Therefore, a queue is shared between the threads.
        :return: None
        """
        while True:
            while self.speed_limit_queue.qsize() > 0:
                self._handle_speedlimit_detection(self.speed_limit_queue.get())
            time.sleep(0.05)

    def _handle_speedlimit_detection(self, image_np):
        """
        Receives potential frames.
        :param image_np: frame as numpy array
        :return: None
        """
        if not self.is_online:
            return
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
            self.timer.start()

    def auto_steer_forward(self, value):
        """
        This function control the wheels as turn scores are very low.
        It uses steer values that are very close to 0, this keeps the car
        as parallel as possible.
        :param value: turn score
        """
        if time.time() - self.last_turn < 0.5 or self.on_turn:
            return
        if abs(value) > 0.01:
            # setting the right steer value
            steer_max = 0.4 if self._get_speed() < 20 else 0.2
            if abs(value) < 0.025:
                steer_max *= 0.75
            if value > 0:
                score = value / 0.20
                func = self.control.right
            else:
                value = abs(value)
                score = value / 0.12
                func = self.control.left

            # calling the matching function (either right ot left)
            turn_value = steer_max * score
            self._add_log(f'{"R" if value > 0 else "L"} adj ({round(steer_max * score, 3)})')

            # skipping turns, as this function handles minor adjustments
            if turn_value > 0.5:
                return
            func(turn_value)
        else:
            # We wish for a value lower or equal to 0.01
            self.control.right(0)

    def auto_accelerate(self, mean_slope):
        """
        This functions is mostly called after the adjustments, or
        in case the car faces forward and can accelerate.
        :param mean_slope:
        """
        if self._get_speed() < self.max_speed - 5 and not self.on_turn:
            # matching a relevant throttle value
            if self.max_speed < 15:
                value = 0.6
            elif self.max_speed <= 20:
                value = 0.65
            else:
                value = 0.7
            self.control.accelerate(value)
        if not self.on_turn:
            if mean_slope is not None:
                self.auto_steer_forward(mean_slope)
            else:
                self.steer_back_interval.try_calling()
        self.maintain_on_turn = False

    def auto_get_relevant_max_speed(self, non_zero_pixels):
        """
        Non zero pixels represent yellow lanes that are rotating either
        left or right. For example, a 90 degree turn would return 1000-2500 pixels.
        This function return a rational maximum speed according to the non zero value.
        :param non_zero_pixels: number of non zero pixels found in predefined search areas.
        :return : A rational speed limit
        """
        if non_zero_pixels > 100:
            # breaking to a rational speed
            self.control.set_brake_target(self.max_speed)

        if non_zero_pixels >= 750:
            if not self.on_turn:
                self.on_turn = True
                self.control.steer_back_slowly = True
            return 10
        elif non_zero_pixels >= 200:
            return 12
        elif non_zero_pixels >= 100:
            return 20
        elif 75 <= non_zero_pixels < 100:
            return 25
        elif 0 <= non_zero_pixels < 75:
            if self.on_turn:
                self.on_turn = False
                self.control.steer_back_slowly = False
            self.control.brake_target = 0
            return self.speed_limit

    def _multiply_max_steer(self, max_value):
        """
        This function simply lowers the steer value as the car is faster, as
        we normally do in real life.
        :param max_value: maximum steer value
        """
        if self._get_speed() <= 15:
            return max_value * 1.2
        elif self._get_speed() <= 25:
            return max_value * 0.5
        elif self._get_speed() <= 35:
            return max_value * 0.3
        elif self._get_speed() <= 45:
            return max_value * 0.15
        else:
            return max_value * 0.1

    def auto_handle_turns(self, data: NavigationData):
        """
        This function handles the auto-pilot turns. It sets a steer value
        according to the navigation data.
        :param data: navigation data
        """
        _, right_count, right_score, left_count, left_score, _, mean_slop = data
        if left_count > right_count:
            func = self.control.left
            direction = "L"
            score = left_score
        else:
            func = self.control.right
            direction = "R"
            score = right_score
        if score < 0.21:
            if self._get_speed() < 10:
                self.control.accelerate(0.6)
            else:
                self.control.accelerate(0.75)
            if mean_slop is not None and not self.control_wait:
                self.auto_steer_forward(mean_slop)
            self.maintain_on_turn = False
        else:
            if score < 0.4:
                max_steer = 0.15 if self._get_speed() < 20 else 0.075
                self.control.accelerate(0.6)
                self.maintain_on_turn = False
            elif score < 0.6:
                max_steer = 0.2 if self._get_speed() < 20 else 0.1
                if not self.maintain_on_turn:
                    self.control.accelerate(0.5)
            else:
                if score < 0.8:
                    max_steer = 0.6 if self._get_speed() < 20 else 0.3
                else:
                    max_steer = 0.7 if self._get_speed() < 20 else 0.5

                if self._get_speed() - self.max_speed > 2:
                    diff = self._get_speed() - self.max_speed
                    self.control.brake(diff / 10)
                else:
                    self.maintain_on_turn = True
                    self.control.accelerate(0.48)

            max_steer = self._multiply_max_steer(max_steer)
            value = max_steer * score
            if value <= 0:
                value = 0.05
            elif value > 0.65:
                value = 0.65

            func(value)
            self._add_log(f"turn-{direction}-{round(value, 2)}-score-{round(score, 2)}")

    def auto_process_input(self, main_img, side_view):
        """
        This function processes the images and detect lanes.
        :param main_img: The main camera faces forward (and 15 degrees to the left).
        The main image returns detections that are used to place the car inside its lane,
        but not necessary facing forward.
        :param side_view: The side camera records the lanes, 90 degrees to the left.
        The side image returns detections that are used to straighten the car, and place it
        as parallel as possible to the yellow lines.
        """
        main_img = cv2.cvtColor(np.asarray(main_img), cv2.COLOR_RGB2BGR)
        side_view = cv2.cvtColor(np.asarray(side_view), cv2.COLOR_RGB2BGR)

        height, width = main_img.shape[:2]

        hough_lines, masked = navigation.detect_lanes(
            img=main_img,
            mask=navigation.Masks.yellow,
            vertices=navigation.get_turn_detection_region(
                width=width,
                height=height
            )
        )
        if hough_lines is None:
            return None

        lines, _ = navigation.filter_lines(
            lines=hough_lines,
            min_slope=0.1
        )

        if lines is None:
            print("lines are None")
            return None

        navigation_scores: navigation.NavScores = navigation.process_lines(
            image=main_img,
            filtered_lines=lines
        )

        if navigation_scores is None:
            return None

        forward, right_count, right_score, left_count, left_score = navigation_scores

        masked_output = vis_utils.visualize_lane_detection(
            img=masked,
            filtered_lines=lines,
            scores=navigation_scores
        )

        non_zero_rectangle = navigation.get_non_zero_rectangle(
            width=width,
            height=height,
            direction=navigation.Directions.right
        )

        non_zero = navigation.get_non_zero_pixels(
            masked=masked,
            rectangle=non_zero_rectangle,
        )

        vis_utils.visualize_non_zero_pixels(
            img=masked_output,
            rectangle=non_zero_rectangle,
            non_zero_pixels=non_zero
        )

        side_hough_lines, side_masked = navigation.detect_lanes(
            img=side_view,
            mask=navigation.Masks.yellow,
            vertices=navigation.get_steer_adjust_region(
                width=width,
                height=height
            )
        )

        mean_slope = None
        if side_hough_lines is not None:
            side_lines, mean_slope = navigation.filter_lines(
                lines=side_hough_lines
            )

            if mean_slope is not None:
                side_masked_output = vis_utils.visualize_lane_detection_side_view(
                    img=side_masked,
                    filtered_lines=side_lines,
                )
                self.side_masked_output = side_masked_output
                cv2.imshow('side-view', side_masked_output)

        self.masked_output = masked_output
        cv2.imshow('masked', masked_output)

        return NavigationData(forward=forward, right_count=right_count, right_score=right_score,
                              left_count=left_count, left_score=left_score, non_zero=non_zero,
                              mean_slope=mean_slope)

    def auto_process_detections(self, data: NavigationData):
        """
        This main function that process the detections. It mostly calls
        the above functions if necessary:
        * auto_get_relevant_max_speed()
        * auto_accelerate()
        * auto_handle_turns()
        It also makes last minor adjustments, according to the current speed.
        """
        if not data:
            # Drive forward as there are no other instructions
            self.control.steer = 0
            self.control_wait = True
        else:
            # Unpack the data tuple
            forward, right_count, right_score, left_count, left_score, \
            non_zero, mean_slope = data

            # As detections are back, the wait flag is canceled
            if self.control_wait and (forward > 2 or left_count > 2 or right_count > 2):
                self.control_wait = False

            # I set an appropriate speed, as cars must slow down before turns
            max_speed = self.auto_get_relevant_max_speed(
                non_zero_pixels=non_zero
            )
            self.max_speed = max_speed

            # Settings control values according to data
            if right_count < 3 and left_count < 3 and forward >= 5:
                self.auto_accelerate(mean_slope=mean_slope)
            elif not self.control_wait:
                self.auto_handle_turns(data)

            # Last adjustments
            if self._get_speed() - self.max_speed > 10:
                diff = self._get_speed() - self.max_speed
                self.control.brake(diff / 20)
            if self._get_speed() >= self.max_speed:
                self.control.set_maintain_settings()
                self.control.maintain_speed()
            if self.on_turn and self.control.steer < 0.4 and mean_slope and mean_slope > 0.4:
                self.control.steer = 0.42

    def auto_pilot(self, main_img, side_view):
        """
        Auto Pilot!
        This function is called from the Carla loop (Every frame).
        :param main_img: Main camera output
        "param side_view: Side camera output
        """
        nav_data: NavigationData = self.auto_process_input(
            main_img=main_img,
            side_view=side_view
        )
        self.auto_process_detections(data=nav_data)

    def _on_render(self):
        """Process and display main image"""

        if self._main_image is not None:
            """
            Pass main image to processing functions
            Features
            -----------------------------
            [1] speedlimit + ocr detection
            """
            array = image_converter.to_rgb_array(self._main_image)
            if self.frame == 4:
                # An array is passed every 5 frames to prevent overflow
                self.speed_limit_queue.put(np.array(array))
            else:
                self.frame += 1
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        if self._main_image is not None and self._side_image is not None:
            """
            Pass main image to processing functions
            Features
            -----------------------------
            [1] speedlimit + ocr detection
            """
            array = image_converter.to_rgb_array(self._main_image)
            side_array = image_converter.to_rgb_array(self._side_image)

            if self.on_auto:
                self.control.set_speed_values(
                    forward_speed=abs(int(self.player_measurements.forward_speed)),
                    max_speed=self.max_speed
                )

                if self.is_online and self.frame == 5:
                    """Detect speed-limit signs"""
                    self.speed_limit_queue.put(np.array(array))
                    self.frame = 0

                self.auto_pilot(
                    main_img=array,
                    side_view=side_array
                )

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