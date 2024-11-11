# Based off gym_carla with several modifications

from __future__ import division

import glob
import os
import sys
from datetime import datetime
from matplotlib import cm
import open3d as o3d
import copy
import numpy as np
import pygame
import random
import time
import threading
from skimage.transform import resize
from PIL import Image

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner, RoadOption
from gym_carla.envs.misc import *

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
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

    self.observation_space = spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 4), dtype=np.uint8)

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
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='242,250,12')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    self.past_img = []



    # Camera sensor
    self.camera_img = np.zeros((4, self.obs_size, self.obs_size), dtype=np.uint8)

    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    self.camera_trans = carla.Transform(carla.Location(x=1.5, z=1.5))

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0

    # Initialize the renderer
    self._init_renderer()

  def reset(self):
    # Clear sensor objects
    self.collision_sensor = None
    self.camera_sensor = None


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

      transform = random.choice(self.vehicle_spawn_points)

      if self._try_spawn_ego_vehicle_at(transform):
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


    # Add camera sensors
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
    self.camera_sensor.listen(lambda data: get_camera_img(data))

    def get_camera_img(data):
      array = np.frombuffer(data.raw_data, dtype=np.uint8)
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      # Grayscale
      array = np.mean(array, axis=2)
      array = array.astype(np.uint8)
      self.camera_img[0] = array
      while len(self.past_img) < 3:
        self.past_img.append(array)
      for i in range(3):
        self.camera_img[i+1] = self.past_img[i]
      self.past_img.pop()
      self.past_img.insert(0, array)


    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    self.last_waypoint = self.waypoints[0]
    ego_x, ego_y = get_pos(self.ego)
    self.last_distance = (ego_x-self.waypoints[1][0])**2 + (ego_y-self.waypoints[1][1])**2
    self.last_lane_distance, w = get_lane_dis(self.waypoints, ego_x, ego_y)

    for _,direction in self.waypoints:
      if direction != RoadOption.LANEFOLLOW:
        print(direction)
        break

    # Set ego information for render
    self.birdeye_render.set_hero(self.ego, self.ego.id)
    return self._get_obs()

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

    carPos = self.ego.get_transform()
    self.world.get_spectator().set_transform(carla.Transform(carla.Location(carPos.location.x, carPos.location.y, 50), carla.Rotation(300,carPos.rotation.yaw,0)))


    self.world.tick()

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
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # state information
    info = {
      'waypoints': self.waypoints,
      'vehicle_front': self.vehicle_front
    }

    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

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

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    pygame.init()
    self.display = pygame.display.set_mode(
    (self.display_size * 5, self.display_size),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

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
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    # birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)


    # Display birdeye image
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    # Direction
    intersect_option = 0
    for location,direction in self.waypoints:
      self.world.debug.draw_point(carla.Location(location[0],location[1],1),life_time=1)
      if intersect_option == 0 and direction != RoadOption.LANEFOLLOW:
        intersect_option = direction

    img = None
    if intersect_option == 0:
      img = Image.open("lane.jpg")
    elif intersect_option == 1:
      img = Image.open("left.jpg")
    elif intersect_option == 2:
      img = Image.open("right.png")
    else:
      img = Image.open("straight.png")
    self.sign_img = np.array(img)
    self.sign_img = np.mean(self.sign_img, axis=2)
    self.sign_img = self.sign_img.astype(np.uint8)
    sign_arr = resize(self.sign_img, (self.obs_size//8, self.obs_size//8), preserve_range=True)
    sign_arr = sign_arr.astype(np.uint8)
    self.sign_img = Image.fromarray(sign_arr)

    ## Display camera image
    camera = resize(self.camera_img, (4, self.obs_size, self.obs_size), preserve_range=True)
    camera = camera.astype(np.uint8)

    temp = Image.fromarray(camera[0])
    temp.paste(self.sign_img, (0,0), mask=self.sign_img)
    camera[0] = resize(np.array(temp), (self.obs_size, self.obs_size), preserve_range=True)

    temp = Image.fromarray(camera[1])
    temp.paste(self.sign_img, (0,0), mask=self.sign_img)
    camera[1] = resize(np.array(temp), (self.obs_size, self.obs_size), preserve_range=True)

    temp = Image.fromarray(camera[2])
    temp.paste(self.sign_img, (0,0), mask=self.sign_img)
    camera[2] = resize(np.array(temp), (self.obs_size, self.obs_size), preserve_range=True)

    temp = Image.fromarray(camera[3])
    temp.paste(self.sign_img, (0,0), mask=self.sign_img)
    camera[3] = resize(np.array(temp), (self.obs_size, self.obs_size), preserve_range=True)

    camera_surface = grayscale_to_display_surface(camera[0], self.display_size)
    self.display.blit(camera_surface, (self.display_size * 1, 0))

    camera_surface = grayscale_to_display_surface(camera[1], self.display_size)
    self.display.blit(camera_surface, (self.display_size * 2, 0))

    camera_surface = grayscale_to_display_surface(camera[2], self.display_size)
    self.display.blit(camera_surface, (self.display_size * 3, 0))

    camera_surface = grayscale_to_display_surface(camera[3], self.display_size)
    self.display.blit(camera_surface, (self.display_size * 4, 0))


    #direction_arr = np.zeros((1, self.obs_size, self.obs_size), dtype=np.uint8)
    #direction_arr[0][0 if intersect_option % 2 == 0 else -1][0 if intersect_option // 2 == 0 else -1] = 255
    #direction_arr = direction_arr.astype(np.uint8)



    # Display on pygame
    pygame.display.flip()


    obs = {
      'camera':camera,
      'birdeye':birdeye.astype(np.uint8),
    }

    return np.transpose(obs['camera'], (1,2,0))

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    #r_speed = -300 * (abs(speed - self.desired_speed)**2)
    #r_speed = 0

    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -100000

    # reward for steering:
    #r_steer = -self.ego.get_control().steer**2

    # reward for out of lane
    ego_x, ego_y = get_pos(self.ego)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    #r_lane_dis = -100 * (abs(dis)**2)
    if abs(dis) < 0.5:
      r_lane_dis = 10000
    elif abs(dis) < abs(self.last_lane_distance):
      r_lane_dis = 5000
    else:
      r_lane_dis = -5000
    self.last_lane_distance = dis

    r_out = 0
    if abs(dis) > self.out_lane_thres:
      r_out = -100000

    # longitudinal speed
    #lspeed = np.array([v.x, v.y])
    #lspeed_lon = np.dot(lspeed, w)


    # cost for stopped
    r_speed = 0
    if speed < 3:
      r_speed = -6000

    r_waypoint = 0
    if (self.waypoints[0] != self.last_waypoint):
      r_waypoint = 10000
      self.last_waypoint = self.waypoints[0]
      self.last_distance = (ego_x-self.waypoints[1][0][0])**2 + (ego_y-self.waypoints[1][0][1])**2
    else:
      if ((ego_x-self.waypoints[1][0][0])**2 + (ego_y-self.waypoints[1][0][1])**2) < self.last_distance:
        self.last_distance = (ego_x-self.waypoints[1][0][0])**2 + (ego_y-self.waypoints[1][0][1])**2
        r_waypoint = 5000
      else:
        r_waypoint = -1000

    print("Collision: " + str(r_collision) + "\n")
    print("Speed: " + str(r_speed) + "\n")
    print("Out: " + str(r_out) + "\n")
    print("Waypoint: " + str(r_waypoint) + "\n")
    print("Lane Dis: " + str(r_lane_dis) + "-----True Value: " + str(dis))
    print("\n-----------------------------------------------")
    r = r_collision + r_speed + r_out - 200 + r_waypoint + r_lane_dis

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


    # If out of lane
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
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
