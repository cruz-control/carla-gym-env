# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# This file utilizes, with modification, LIDAR code from the CARLA Python examples library:
# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import sys
import numpy as np
import random
import time
from queue import PriorityQueue
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from skimage.transform import resize
import carla
from enum import Enum


class Turn(Enum):
    LEFT = 1
    STRAIGHT = 2
    RIGHT = 3


def get_closest_waypoint(route, curr_location):
    closest_waypoint = None
    previous_dist = None
    i = None

    for i, waypoint in enumerate(route):
        if closest_waypoint is None:
            closest_waypoint = waypoint
            previous_dist = waypoint.transform.location.distance(curr_location)
            i = i
            continue

        dist = waypoint.transform.location.distance(curr_location)

        if dist < previous_dist:
            closest_waypoint = waypoint
            previous_dist = dist

        return i, closest_waypoint, previous_dist


def euclidean_heuristic(waypoint, end_waypoint):
    return waypoint.transform.location.distance(end_waypoint.transform.location)


def manhattan_heuristic(waypoint, end_waypoint):
    dx = abs(waypoint.transform.location.x - end_waypoint.transform.location.x)
    dy = abs(waypoint.transform.location.y - end_waypoint.transform.location.y)
    dz = abs(waypoint.transform.location.z - end_waypoint.transform.location.z)
    return dx + dy + dz


class AStarNode:
    def __init__(self, waypoint, g_cost, h_cost, parent=None):
        self.waypoint = waypoint
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent


def get_legal_neighbors(waypoint):
    neighbors = []
    # Forward neighbor
    forward = waypoint.next(2.0)
    if forward:
        neighbors.extend(forward)

    # Legal left lane change
    if waypoint.lane_change & carla.LaneChange.Left:
        left_lane = waypoint.get_left_lane()
        if left_lane and left_lane.lane_type == carla.LaneType.Driving:
            neighbors.append(left_lane)

    # Legal right lane change
    if waypoint.lane_change & carla.LaneChange.Right:
        right_lane = waypoint.get_right_lane()
        if right_lane and right_lane.lane_type == carla.LaneType.Driving:
            neighbors.append(right_lane)

    return neighbors


def a_star(
    world,
    start_waypoint,
    end_waypoint,
    heuristic_func=euclidean_heuristic,
    max_distance=5000,
):
    start_node = AStarNode(
        start_waypoint, 0, heuristic_func(start_waypoint, end_waypoint)
    )
    open_set = PriorityQueue()
    open_set.put((start_node.f_cost, id(start_node), start_node))
    came_from = {}
    g_score = {start_waypoint.id: 0}
    f_score = {start_waypoint.id: start_node.f_cost}

    while not open_set.empty():
        current_node = open_set.get()[2]

        # Early exit if we have reached near the goal
        if (
            current_node.waypoint.transform.location.distance(
                end_waypoint.transform.location
            )
            < 10.0
        ):
            path = []

            while current_node:
                path.append(current_node.waypoint)
                current_node = came_from.get(current_node.waypoint.id)
            return list(reversed(path))

        for next_waypoint in get_legal_neighbors(current_node.waypoint):
            lane_change_cost = (
                5 if next_waypoint.lane_id != current_node.waypoint.lane_id else 0
            )
            tentative_g_score = (
                g_score[current_node.waypoint.id]
                + euclidean_heuristic(current_node.waypoint, next_waypoint)
                + lane_change_cost
            )
            if (
                next_waypoint.id not in g_score
                or tentative_g_score < g_score[next_waypoint.id]
            ):
                came_from[next_waypoint.id] = current_node
                g_score[next_waypoint.id] = tentative_g_score
                f_score[next_waypoint.id] = tentative_g_score + heuristic_func(
                    next_waypoint, end_waypoint
                )
                new_node = AStarNode(
                    next_waypoint,
                    tentative_g_score,
                    heuristic_func(next_waypoint, end_waypoint),
                    current_node,
                )
                open_set.put((f_score[next_waypoint.id], id(new_node), new_node))

    print("A* search failed to find a path")
    return None


class NewCarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self):
        # parameters
        self.number_of_vehicles = 1
        self.number_of_walkers = 0
        self.max_time_episode = 100

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )  # throttle, braking, steering
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(786532,), dtype=np.float32
        )

        # Connect to carla server and get world object
        print("connecting to Carla server...")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4000.0)
        self.world = self.client.get_world()

        print("Carla server connected!")

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc != None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(
            "vehicle.lincoln*", color="49,8,8"
        )

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )

        # Lidar sensor
        self.lidar_data = None
        self.lidar_height = 1.8
        self.lidar_trans = carla.Transform(carla.Location(x=-0.5, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        self.lidar_bp.set_attribute("channels", "64.0")
        self.lidar_bp.set_attribute("range", "100.0")
        self.lidar_bp.set_attribute("upper_fov", "15")
        self.lidar_bp.set_attribute("lower_fov", "-25")
        self.lidar_bp.set_attribute("rotation_frequency", str(1.0 / 0.05))
        self.lidar_bp.set_attribute("points_per_second", "500000")

        # Camera sensor
        self.img_size = 256
        self.camera_img = np.zeros(
            (4, self.img_size, self.img_size, 3), dtype=np.dtype("uint8")
        )
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute("image_size_x", str(self.img_size))
        self.camera_bp.set_attribute("image_size_y", str(self.img_size))
        self.camera_bp.set_attribute("fov", "110")

        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute("sensor_tick", "0.02")
        self.camera_trans = carla.Transform(carla.Location(x=1.5, z=1.5))
        self.camera_trans2 = carla.Transform(
            carla.Location(x=0.7, y=0.9, z=1), carla.Rotation(pitch=-35.0, yaw=134.0)
        )
        self.camera_trans3 = carla.Transform(
            carla.Location(x=0.7, y=-0.9, z=1), carla.Rotation(pitch=-35.0, yaw=-134.0)
        )
        self.camera_trans4 = carla.Transform(
            carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180.0)
        )
        
        self.time_step = 0
        
        # Set fixed simulation step for synchronous mode
        self._set_synchronous_mode()
        self.things = []

    def reset(self, seed=None, options={}):
        print("HERE")
        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.camera2_sensor = None
        self.camera3_sensor = None
        self.camera4_sensor = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors()

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles

        while count > 0:
          v = self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4])
          if v != False:
            self.things.append(v)
            count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        
        while count > 0:
          v = self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points))
          if v != False:
            self.things.append(v)
            count -= 1
        
        # Spawn Ego
        while True:
          carla_map = self.world.get_map()
          spawn_points = carla_map.get_spawn_points()

          # Choose a random starting location (point A)
          point_a = random.choice(spawn_points)

          # Choose a random destination (point B)
          point_b = random.choice(spawn_points)
          while point_b.location == point_a.location:
              point_b = random.choice(spawn_points)

          start_waypoint = carla_map.get_waypoint(point_a.location)
          end_waypoint = carla_map.get_waypoint(point_b.location)

          self.route = a_star(self.world, start_waypoint, end_waypoint)
          v = self.world.try_spawn_actor(self.ego_bp, start_waypoint.transform)

          if v is not None:
            self.ego = v
            self.things.append(v)
            break

        # Add collision sensor
        self.collision_hist = []

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego
        )
        self.things.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        # Add lidar sensor
        self.lidar_data = None

        def get_lidar_data(point_cloud):
            data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype("f4")))
            data = np.reshape(data, (int(data.shape[0] / 4), 4))
            self.lidar_data = data

        self.lidar_sensor = self.world.spawn_actor(
            self.lidar_bp, self.lidar_trans, attach_to=self.ego
        )
        self.things.append(self.lidar_sensor)
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        # Add Cameras
        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img[0] = array

        def get_camera_img2(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img[1] = array

        def get_camera_img3(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img[2] = array

        def get_camera_img4(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img[3] = array

        # Add camera sensors
        self.camera_sensor = self.world.spawn_actor(
            self.camera_bp, self.camera_trans, attach_to=self.ego
        )
        self.things.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: get_camera_img(data))
        
        self.camera_sensor2 = self.world.spawn_actor(
            self.camera_bp, self.camera_trans2, attach_to=self.ego
        )
        self.things.append(self.camera_sensor2)
        self.camera_sensor2.listen(lambda data: get_camera_img2(data))
        
        self.camera_sensor3 = self.world.spawn_actor(
            self.camera_bp, self.camera_trans3, attach_to=self.ego
        )
        self.things.append(self.camera_sensor3)
        self.camera_sensor3.listen(lambda data: get_camera_img3(data))
        
        self.camera_sensor4 = self.world.spawn_actor(
            self.camera_bp, self.camera_trans4, attach_to=self.ego
        )
        self.things.append(self.camera_sensor4)
        self.camera_sensor4.listen(lambda data: get_camera_img4(data))

        # Set waypoint calc for reward
        _, waypoint, dist = get_closest_waypoint(self.route, self.ego.get_location())
        self.prev_waypoint = waypoint
        self.prev_w_dist = dist

        goal = self.route[-1].transform.location
        self.prev_g_dist = goal.distance(self.ego.get_location())
        
        self.time_step = 0

        return self._get_obs(), {}

    def step(self, action):
        def map_value(value, from_min, from_max, to_min, to_max):
            """Maps a value from one range to another."""
            from_range = from_max - from_min
            to_range = to_max - to_min
            scaled_value = (value - from_min) / from_range
            return to_min + (scaled_value * to_range)

        throttle = map_value(action[0], -1, 1, 0, 1)
        brake = map_value(action[1], -1, 1, 0, 1)
        steer = action[2]

        # Apply control
        act = carla.VehicleControl(
            throttle=float(throttle), steer=float(-steer), brake=float(brake)
        )
        self.ego.apply_control(act)

        self.world.tick()

        _, _, dist = get_closest_waypoint(self.route, self.ego.get_location())

        reward = self._get_reward()

        sys.stdout.write(
            "\r"
            + " Step: "
            + str(self.time_step)
            + " Dist: "
            + str(round(dist))
            + " Reward: "
            + str(round(reward))
            + " Throttle: "
            + str(round(throttle, 2))
            + " Braking: "
            + str(round(brake, 2))
            + " Steer: "
            + str(round(steer, 2))
            + "\n"
        )
        sys.stdout.flush()

        terminated, truncated = self._terminal()
        
        self.time_step += 1
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Get the observations."""

        ## Display camera image
        camera = resize(self.camera_img, (4, self.img_size, self.img_size, 3)) * 255
        camera = camera.astype(np.float32)
        cameras = camera.flatten()

        # Distance to closest waypoint
        _, _, dist = get_closest_waypoint(self.route, self.ego.get_location())
        dist_to_waypoint = np.array([dist])
        dist_to_waypoint = np.pad(
            dist_to_waypoint,
            (0, 49),
            "constant",
            constant_values=(0, dist_to_waypoint[0]),
        )

        # Turn to take
        next_turn = self.get_turn()
        turn = np.array([next_turn.value])
        turn = np.pad(turn, (0, 99), "constant", constant_values=(0, turn[0]))

        obs = np.concatenate((cameras, turn))
        return np.float32(obs)

    def _get_reward(self):
        """Calculate the step reward."""
        # reward for collision
        r_collision = -1
        if len(self.collision_hist) > 0:
            r_collision = -3

        # reward for distance from A* path
        _, current_waypoint, curr_w_dist = get_closest_waypoint(
            self.route, self.ego.get_location()
        )
        prev_w_dist = self.prev_w_dist
        self.prev_w_dist = curr_w_dist

        r_w_dist = 1

        if self.prev_waypoint.id == current_waypoint.id:
            r_w_dist = abs(prev_w_dist) - abs(curr_w_dist)
        else:
            self.prev_waypoint = current_waypoint
            self.prev_w_dist = curr_w_dist

        # reward for distance from goal
        prev_g_dist = self.prev_g_dist
        goal = self.route[-1].transform.location
        current = self.ego.get_location()
        curr_g_dist = goal.distance(current)
        self.prev_g_dist = curr_g_dist
        r_g_dist = abs(prev_g_dist) - abs(curr_g_dist)

        r = (r_collision) + (r_w_dist * 20) + (r_g_dist * 10)
        r = r * 10

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        truncated = False
        terminated = False

        # If collides
        if len(self.collision_hist) > 0:
            terminated = True

        # If reach maximum time step
        if self.time_step > self.max_time_episode:
            truncated = True

        # If arrived at the destination
        destination = self.route[-1].transform.location
        dest_dist = self.ego.get_location().distance(destination)

        if dest_dist < 10:
            terminated = True

        return terminated, truncated

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _create_vehicle_bluepprint(
        self, actor_filter, color=None, number_of_wheels=[4]
    ):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints if int(x.get_attribute("number_of_wheels")) == nw
            ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute("color"):
            if not color:
                color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        return bp

    def _set_synchronous_mode(self):
        new_settings = self.world.get_settings()
        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(new_settings) 
        self.client.reload_world(False)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint(
            "vehicle.*", number_of_wheels=number_of_wheels
        )
        blueprint.set_attribute("role_name", "autopilot")
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot(enabled=True, tm_port=4050)
            return vehicle
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter("walker.*"))
        # set as not invencible
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find(
                "controller.ai.walker"
            )
            walker_controller_actor = self.world.spawn_actor(
                walker_controller_bp, carla.Transform(), walker_actor
            )
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(
                self.world.get_random_location_from_navigation()
            )
            # random max speed
            walker_controller_actor.set_max_speed(
                1 + random.random()
            )  # max speed between 1 and 2 (default is 1.4 m/s)
            return walker_controller_actor
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def get_turn(self):
        route = self.route

        n = 20

        i, waypoint, _ = get_closest_waypoint(route, self.ego.get_location())

        nth = None

        if i + n >= len(route):
            nth = route[-1]

        else:
            nth = route[i + n]

        self.world.debug.draw_string(
            nth.transform.location,
            "^",
            draw_shadow=False,
            color=carla.Color(r=0, g=0, b=255),
            life_time=25.0,
            persistent_lines=True,
        )

        n_r = nth.transform.rotation.yaw
        w_r = waypoint.transform.rotation.yaw
        ang = round(n_r - w_r)

        if ang == 0 or ang == 360 or ang == -360:
            self.next_turn = Turn.STRAIGHT
        elif ang > 0:
            self.next_turn = Turn.RIGHT
        elif ang < 0:
            self.next_turn = Turn.LEFT

        return self.next_turn

    def get_pos(vehicle):
        """
        Get the position of a vehicle
        :param vehicle: the vehicle whose position is to get
        :return: speed as a float in Kmh
        """
        trans = vehicle.get_transform()
        x = trans.location.x
        y = trans.location.y
        return x, y

    def _clear_all_actors(self):
        for thing in self.things:
          if type(thing) != carla.Vehicle:
            thing.stop()
          thing.destroy()
          
        self.things = []
