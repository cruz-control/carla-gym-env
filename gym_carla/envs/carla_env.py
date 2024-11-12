# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# This file utilizes, with modification, LIDAR code from the CARLA Python examples library:
# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import glob
import os
import sys
from datetime import datetime
from matplotlib import cm
import open3d as o3d
import copy
import numpy as np
# import pygame
import random
import time
import threading
from skimage.transform import resize
from PIL import Image
from queue import PriorityQueue

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import carla

# from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *
from enum import Enum

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

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
    
    dist =  waypoint.transform.location.distance(curr_location)
    
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

def a_star(world, start_waypoint, end_waypoint, heuristic_func=euclidean_heuristic, max_distance=5000):
    start_node = AStarNode(start_waypoint, 0, heuristic_func(start_waypoint, end_waypoint))
    open_set = PriorityQueue()
    open_set.put((start_node.f_cost, id(start_node), start_node))
    came_from = {}
    g_score = {start_waypoint.id: 0}
    f_score = {start_waypoint.id: start_node.f_cost}
    
    while not open_set.empty():
        current_node = open_set.get()[2]
        
        # Early exit if we have reached near the goal
        if current_node.waypoint.transform.location.distance(end_waypoint.transform.location) < 10.0:
            path = []

            while current_node:
                path.append(current_node.waypoint)
                current_node = came_from.get(current_node.waypoint.id)
            return list(reversed(path))
        
        for next_waypoint in get_legal_neighbors(current_node.waypoint):
            lane_change_cost = 5 if next_waypoint.lane_id != current_node.waypoint.lane_id else 0
            tentative_g_score = g_score[current_node.waypoint.id] + euclidean_heuristic(current_node.waypoint, next_waypoint) + lane_change_cost
            if next_waypoint.id not in g_score or tentative_g_score < g_score[next_waypoint.id]:
                came_from[next_waypoint.id] = current_node                
                g_score[next_waypoint.id] = tentative_g_score
                f_score[next_waypoint.id] = tentative_g_score + heuristic_func(next_waypoint, end_waypoint)
                new_node = AStarNode(next_waypoint, tentative_g_score, heuristic_func(next_waypoint, end_waypoint), current_node)
                open_set.put((f_score[next_waypoint.id], id(new_node), new_node))
                
    print("A* search failed to find a path")
    return None


params = {
  'number_of_vehicles': 1,
  'number_of_walkers': 0,
  'display_size': 256,  # screen size of bird-eye render
  'max_past_step': 1,  # the number of past steps to draw
  'dt': 0.1,  # time interval between two frames
  'discrete': True,  # whether to use discrete control space
  'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
  'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
  'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
  'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
  'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
  'port': 2000,  # connection port
  'town': 'Town03',  # which town to simulate
  'max_time_episode': 1000,  # maximum timesteps per episode
  'max_waypt': 12,  # maximum number of waypoints
  'obs_range': 32,  # observation range (meter)
  'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
  'd_behind': 12,  # distance behind the ego vehicle (meter)
  'out_lane_thres': 2.0,  # threshold for out of lane
  'desired_speed': 8,  # desired speed (m/s)
  'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
  'display_route': True,  # whether to render the desired route
}

class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params = params):
    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']

    # action and observation spaces
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer

    self.observation_space = spaces.Box(low=0, high=255, shape=(786532,), dtype=np.float32)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(4000.0)
    self.world = client.load_world(params['town'])
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # Lidar sensor
    self.lidar_data = None
    self.lidar_height = 1.8
    self.lidar_trans = carla.Transform(carla.Location(x=-0.5, z=self.lidar_height))
    self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    self.lidar_bp.set_attribute('channels', '64.0')
    self.lidar_bp.set_attribute('range', '100.0')
    self.lidar_bp.set_attribute('upper_fov', '15')
    self.lidar_bp.set_attribute('lower_fov', '-25')
    self.lidar_bp.set_attribute('rotation_frequency', str(1.0 / 0.05))
    self.lidar_bp.set_attribute('points_per_second', '500000')

    # Camera sensor
    self.camera_img = np.zeros((4, self.obs_size, self.obs_size, 3), dtype = np.dtype("uint8"))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    self.camera_trans = carla.Transform(carla.Location(x=1.5, z=1.5))

    self.camera_trans2 = carla.Transform(carla.Location(x=0.7, y=0.9, z=1), carla.Rotation(pitch=-35.0, yaw=134.0))

    self.camera_trans3 = carla.Transform(carla.Location(x=0.7, y=-0.9, z=1), carla.Rotation(pitch=-35.0, yaw=-134.0))

    self.camera_trans4 = carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180.0))

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0

    # Initialize the renderer
    # self._init_renderer()
    
    # Initialize next turn
    self.next_turn = Turn.STRAIGHT
    
    self.prev_g_dist = 0

  def reset(self, seed=None, options={}):
    # Clear sensor objects
    self.collision_sensor = None
    self.lidar_sensor = None
    self.camera_sensor = None
    self.camera2_sensor = None
    self.camera3_sensor = None
    self.camera4_sensor = None

    # Delete sensors, vehicles and walkers
    self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

    # Disable sync mode
    self._set_synchronous_mode(False)

    # Spawn surrounding vehicles
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
        count -= 1

    # Spawn pedestrians
    random.shuffle(self.walker_spawn_points)
    count = self.number_of_walkers
    if count > 0:
      for spawn_point in self.walker_spawn_points:
        if self._try_spawn_random_walker_at(spawn_point):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
        count -= 1

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()
      
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
      
      # for waypoint in self.route:
            # self.world.debug.draw_string(waypoint.transform.location, '^', draw_shadow=False, color=carla.Color(r=220, g=0, b=0), life_time=25.0, persistent_lines=True)

      if self._try_spawn_ego_vehicle_at(point_a):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: get_collision_hist(event))
    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist)>self.collision_hist_l:
        self.collision_hist.pop(0)
    self.collision_hist = []

    # Add lidar sensor
    self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
    self.point_list = o3d.geometry.PointCloud()
    self.lidar_sensor.listen(lambda data: get_lidar_data(data, self.point_list))
    def get_lidar_data(point_cloud, point_list):
      data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
      data = np.reshape(data, (int(data.shape[0] / 4), 4))

      # Isolate the intensity and compute a color for it
      intensity = data[:, -1]
      intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
      int_color = np.c_[
          np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
          np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
          np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

      # Isolate the 3D data
      points = data[:, :-1]

      points[:, :1] = -points[:, :1]

      point_list.points = o3d.utility.Vector3dVector(points)
      point_list.colors = o3d.utility.Vector3dVector(int_color)

    def run_open3d():
      # self.vis = o3d.visualization.Visualizer()
      # self.vis.create_window(
      #     window_name='Carla Lidar',
      #     width=540,
      #     height=540,
      #     left=480,
      #     top=270, visible=False)
      # self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
      # self.vis.get_render_option().point_size = 1
      # self.vis.get_render_option().show_coordinate_frame = True

      self.frame = 0
      self.dt0 = datetime.now()


    thread_open3d = threading.Thread(target=run_open3d)
    thread_open3d.start()
    
    def get_camera_img(data):
      array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      self.camera_img[0] = array

    def get_camera_img2(data):
      array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      self.camera_img[1] = array

    def get_camera_img3(data):
      array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      self.camera_img[2] = array

    def get_camera_img4(data):
      array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      self.camera_img[3] = array

    # Add camera sensors
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
    self.camera_sensor.listen(lambda data: get_camera_img(data))
    self.camera_sensor2 = self.world.spawn_actor(self.camera_bp, self.camera_trans2, attach_to=self.ego)
    self.camera_sensor2.listen(lambda data: get_camera_img2(data))
    self.camera_sensor3 = self.world.spawn_actor(self.camera_bp, self.camera_trans3, attach_to=self.ego)
    self.camera_sensor3.listen(lambda data: get_camera_img3(data))
    self.camera_sensor4 = self.world.spawn_actor(self.camera_bp, self.camera_trans4, attach_to=self.ego)
    self.camera_sensor4.listen(lambda data: get_camera_img4(data))
      
    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    # self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    # self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
    
    print(self.ego.get_location())
    
    _ , waypoint , dist = get_closest_waypoint(self.route, self.ego.get_location())
    self.prev_waypoint = waypoint
    self.prev_w_dist = dist
    
    goal = self.route[-1].transform.location
    self.prev_g_dist = goal.distance(self.ego.get_location())
    
    # Set ego information for render
    # self.birdeye_render.set_hero(self.ego, self.ego.id)
    
    # state information
    info = {
      #'waypoints': self.waypoints,
      #'vehicle_front': self.vehicle_front
    }
    
    return self._get_obs(), copy.deepcopy(info)

  def step(self, action):
    # Calculate acceleration and steering
    if self.discrete:
      acc = self.discrete_act[0][action//self.n_steer]
      steer = self.discrete_act[1][action%self.n_steer]
    else:
      acc = action[0]
      steer = action[1]

    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc/3,0,1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc/8,0,1)

    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    # def update_open3d():
    #   if self.frame == 2:
    #       self.vis.add_geometry(self.point_list)
    #   self.vis.update_geometry(self.point_list)

    #   self.vis.poll_events()
    #   self.vis.update_renderer()
    #   self.vis.capture_screen_image(filename="lidar_temp_img.png")


    # thread_update3d = threading.Thread(target=update_open3d)
    # thread_update3d.start()
         # This can fix Open3D jittering issues:
    # time.sleep(0.005)

    self.world.tick()
    
    _ , _ , dist = get_closest_waypoint(self.route, self.ego.get_location())
    
    reward = self._get_reward()

    process_time = datetime.now() - self.dt0
    sys.stdout.write('\r' + 'FPS: ' + str(round(1.0 / process_time.total_seconds())) + " Dist: " + str(round(dist)) + " Reward: " + str(round(reward)) + " Throttle: " + str(round(throttle, 2)) + " Braking: " + str(round(brake, 2)) + " Steer: " + str(round(steer, 2)) + " 0: " + str(action[0]))
    sys.stdout.flush()
    self.dt0 = datetime.now()
    self.frame += 1

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # route planner
    #self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # state information
    info = {
      #'waypoints': self.waypoints,
      #'vehicle_front': self.vehicle_front
    }

    # Update timesteps
    self.time_step += 1
    self.total_step += 1
    
    return (self._get_obs(), reward, self._terminal(), self._terminal(), copy.deepcopy(info))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode):
    pass

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  # def _init_renderer(self):
  #   """Initialize the birdeye view renderer.
  #   """
  #   pygame.init()
  #   self.display = pygame.display.set_mode(
  #   (self.display_size * 6, self.display_size),
  #   pygame.HWSURFACE | pygame.DOUBLEBUF)

  #   pixels_per_meter = self.display_size / self.obs_range
  #   pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
  #   birdeye_params = {
  #     'screen_size': [self.display_size, self.display_size],
  #     'pixels_per_meter': pixels_per_meter,
  #     'pixels_ahead_vehicle': pixels_ahead_vehicle
  #   }
  #   self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      vehicle.set_autopilot(enabled=True, tm_port=4050)
      return True
    return False

  def _try_spawn_random_walker_at(self, transform):
    """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    # set as not invencible
    if walker_bp.has_attribute('is_invincible'):
      walker_bp.set_attribute('is_invincible', 'false')
    walker_actor = self.world.try_spawn_actor(walker_bp, transform)

    if walker_actor is not None:
      walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
      walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
      # start walker
      walker_controller_actor.start()
      # set walk to random point
      walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
      # random max speed
      walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
      return True
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
      self.ego=vehicle
      return True

    return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering
    # self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    # self.birdeye_render.walker_polygons = self.walker_polygons
    # self.birdeye_render.waypoints = self.waypoints

    # birdeye view with roadmap and actors
    # birdeye_render_types = ['roadmap', 'actors']
    # if self.display_route:
      # birdeye_render_types.append('waypoints')
    # self.birdeye_render.render(self.display, birdeye_render_types)
    # birdeye = pygame.surfarray.array3d(self.display)
    # birdeye = birdeye[0:self.display_size, :, :]
    # birdeye = display_to_rgb(birdeye, self.obs_size)

    # Display birdeye image
    # birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    # self.display.blit(birdeye_surface, (0, 0))

    img = Image.open("lidar_temp_img.png")
    self.lidar_img = np.array(img)
    lidar_arr = np.zeros((1, self.obs_size, self.obs_size, 3))
    lidar_arr = lidar_arr.astype(np.float32)
    lidar_arr[0] = resize(self.lidar_img, (self.obs_size, self.obs_size, 3)) * 255
    # lidar_surface = rgb_to_display_surface(lidar_arr[0], self.display_size)
    # self.display.blit(lidar_surface, (self.display_size * 1, 0))

    ## Display camera image
    camera = resize(self.camera_img, (4, self.obs_size, self.obs_size, 3)) * 255
    camera = camera.astype(np.float32)

    # camera_surface = rgb_to_display_surface(camera[0], self.display_size)
    # self.display.blit(camera_surface, (self.display_size * 3, 0))

    # camera_surface2 = rgb_to_display_surface(camera[1], self.display_size)
    # self.display.blit(camera_surface2, (self.display_size * 2, 0))

    # camera_surface3 = rgb_to_display_surface(camera[2], self.display_size)
    # self.display.blit(camera_surface3, (self.display_size * 4, 0))

    # camera_surface4 = rgb_to_display_surface(camera[3], self.display_size)
    # self.display.blit(camera_surface4, (self.display_size * 5, 0))

    # Display on pygame
    # pygame.display.flip()

    obs = {
      'camera':camera,
      # 'lidar':lidar_arr,
      # 'birdeye':birdeye.astype(np.uint8),
    }
    
    next_turn = self.get_turn()
    
    # Distance to closest waypoint
    _ , _ , dist = get_closest_waypoint(self.route, self.ego.get_location())
    dist_to_waypoint =  np.array([dist])
    dist_to_waypoint = np.pad(dist_to_waypoint, (0, 49), 'constant', constant_values=(0, dist_to_waypoint[0]))
    
    # Turn to take
    turn = np.array([next_turn.value]) 
    turn = np.pad(turn, (0, 49), 'constant', constant_values=(0, turn[0]))
    
    cameras = obs['camera'].flatten()

    obs = np.concatenate((cameras, turn, dist_to_waypoint))
    return np.float32(obs)
  
  def get_turn(self):
    route = self.route
    
    n = 20
    
    i, waypoint, _ = get_closest_waypoint(route, self.ego.get_location())

    nth = None

    if i + n >= len(route):
        nth = route[-1]
        
    else:
        nth = route[i + n]
        
    self.world.debug.draw_string(nth.transform.location, '^', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=25.0, persistent_lines=True)
    
    n_r = (nth.transform.rotation.yaw)
    w_r = (waypoint.transform.rotation.yaw)
    ang = round(n_r - w_r)
    
    if ang == 0 or ang == 360 or ang == -360:
        self.next_turn = Turn.STRAIGHT
    elif ang > 0:
        self.next_turn = Turn.RIGHT
    elif ang < 0:
        self.next_turn = Turn.LEFT
    
    return self.next_turn

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    # v = self.ego.get_velocity()
    # speed = np.sqrt(v.x**2 + v.y**2)
    # r_speed = -abs(speed - self.desired_speed)

    # reward for collision
    r_collision = -1
    if len(self.collision_hist) > 0:
      r_collision = -3

    # reward for steering:
    # r_steer = -self.ego.get_control().steer**2
    
    # reward for distance from A* path
    _ , current_waypoint , curr_w_dist = get_closest_waypoint(self.route, self.ego.get_location())
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

    # reward for out of lane
    # ego_x, ego_y = get_pos(self.ego)
    # dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    # r_out = 0
    # if abs(dis) > self.out_lane_thres:
    #   r_out = -1

    # longitudinal speed
    # lspeed = np.array([v.x, v.y])
    # lspeed_lon = np.dot(lspeed, w)

    # cost for too fast
    # r_fast = 0
    # if lspeed_lon > self.desired_speed:
    #   r_fast = -1

    # cost for lateral acceleration
    # r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

    r = (r_collision) + (r_w_dist * 20) + (r_g_dist * 10)
    r = r * 10
    
    return r

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides
    if len(self.collision_hist)>0:
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True

    destination = self.route[-1].transform.location
    dest_dist = self.ego.get_location().distance(destination)
    
    if dest_dist < 10:
      return True

    return False

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()
