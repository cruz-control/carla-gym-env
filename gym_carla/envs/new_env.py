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
import os
from scipy.ndimage import label
from PIL import Image
import torch
import torch.nn.functional as torchfunc
import math

SEMANTIC_TAGS = {
    'unlabeled':    0,
    'road':         1,
    'sidewalk':     2,
    'building':     3,
    'wall':         4,
    'fence':        5,
    'pole':         6,
    'traffic_light': 7,
    'traffic_sign': 8,
    'vegetation':   9,
    'terrain':      10,
    'sky':          11,
    'pedestrian':   12,
    'rider':        13,
    'car':          14,
    'truck':        15,
    'bus':          16,
    'train':        17,
    'motorcycle':   18,
    'bicycle':      19,
    'static':       20,
    'dynamic':      21,
    'other':        22,
    'water':        23,
    'road_line':    24,
    'ground':       25,
    'bridge':       26,
    'rail_track':   27,
    'guard_rail':   28,
    'ego': 29, #custom tag
    'route':30 #custom tag
}

def save_tensor_visualization(tensor, path):
    # tensor shape: (NUM_CLASSES, H, W)
    # Convert one-hot back to tag indices by taking argmax across class dimension
    tag_indices = tensor.argmax(dim=0)  # (H, W)
    
    # Convert to numpy
    tag_indices = tag_indices.numpy().astype(np.uint8)
    
    # Remap indices back to colors
    COLORS = {
        0:  (0,   0,   0),    # unlabeled
        1:  (128, 64,  128),  # road
        2:  (244, 35,  232),  # sidewalk
        3:  (70,  70,  70),   # building
        4:  (102, 102, 156),  # wall
        5:  (190, 153, 153),  # fence
        6:  (153, 153, 153),  # pole
        7:  (250, 170, 30),   # traffic light
        8:  (220, 220, 0),    # traffic sign
        9:  (107, 142, 35),   # vegetation
        10: (152, 251, 152),  # terrain
        11: (70,  130, 180),  # sky
        12: (220, 20,  60),   # pedestrian
        13: (200, 0,   55),    # rider
        14: (0,   0,   142),  # car
        15: (0,   0,   70),   # truck
        16: (0,   60,  100),  # bus
        17: (0,   80,  100),  # train
        18: (0,   0,   230),  # motorcycle
        19: (119, 11,  32),   # bicycle
        20: (110, 190, 160),  # static
        21: (170, 120, 50),   # dynamic
        22: (55,  90,  80),   # other
        23: (45,  60,  150),  # water
        24: (157, 234, 50),   # road line
        25: (81,  0,   81),   # ground
        26: (150, 100, 100),  # bridge
        27: (230, 150, 140),  # rail track
        28: (180, 165, 180),  # guard rail
        29: (255, 0,   0),    # ego — red
        30: (255, 255, 255),  # route — white
    }
    
    # Build RGB image from index map
    rgb = np.zeros((tag_indices.shape[0], tag_indices.shape[1], 3), dtype=np.uint8)
    for idx, color in COLORS.items():
        rgb[tag_indices == idx] = color
    

    route_channel = tensor[SEMANTIC_TAGS['route']].numpy()
    rgb[route_channel == 1] = COLORS[SEMANTIC_TAGS['route']]

    Image.fromarray(rgb).save(path)


def lerp(x1, y1, x2, y2, f):
    x = x1 * (1-f) + x2*f
    y = y1 * (1-f) +y2*f

    return (x, y)

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



params = {
    'number_of_vehicles': 1,
    'number_of_walkers': 0,
    'max_time_episode': 100,
    'port': 4000,
    'connection_timeout': 100,
    'town': 'Town03',
    'weather': carla.WeatherParameters.ClearNoon,
    'ego_vehicle_filter': "vehicle.mini.cooper_s_2021",
    'ego_vehicle_color': '0,255,115',
    'spectator_height': 50,

    'bev_params': {
        'dim_x': '520',
        'dim_y': '720',
        'ego_bev_rgb':  [0,0,255], # Depreciated
        'height': 200,
        'fov': '20'
    }
}




class NewCarlaEnv(gym.Env):

    def disable_unneeded_layers(self):
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.world.unload_map_layer(carla.MapLayer.Walls)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        

    def __init__(self, params=params):
        # parameters
        self.number_of_vehicles = params["number_of_vehicles"]
        self.number_of_walkers  = params["number_of_walkers"]
        self.max_time_episode   = params["max_time_episode"]

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )  # throttle, braking, steering
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(786532,), dtype=np.float32
        )

        # Connect to carla server and get world object
        print("connecting to Carla server...")
        self.client = carla.Client("localhost", params["port"])
        self.client.set_timeout(params["connection_timeout"])
        #self.world = self.client.load_world(params['town'])
        self.world = self.client.get_world()


        print("Carla server connected!")

        # Set weather
        self.world.set_weather(params["weather"])
        self.spectator_height = params["spectator_height"]


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
            params["ego_vehicle_filter"], params["ego_vehicle_color"]
        )



        #BEV

        self.make_bev_camera_bp(params['bev_params'])

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




        self.bev_output_folder = '/home/ubuntu/bev_output/'

        print("making out dir...")
        os.makedirs(self.bev_output_folder, exist_ok=True)
        print("made out dir!")

        
        # Set fixed simulation step for synchronous mode
        self._set_synchronous_mode()
        self.things = []


        self.disable_unneeded_layers()

    def draw_a_start_path_in_simulation(self, path, lifetime = 1):
        for i in range(len(path)-1):
            begin=carla.Location(
                x=path[i].transform.location.x,
                y=path[i].transform.location.y,
                z=path[i].transform.location.z + 3  # lift off ground
            )
            end=carla.Location(
                x=path[i+1].transform.location.x,
                y=path[i+1].transform.location.y,
                z=path[i+1].transform.location.z + 3  # lift off ground
            )
            self.world.debug.draw_line(begin, end =end, thickness=0.1, color = carla.Color(255,0,0), life_time = lifetime)
            self.world.debug.draw_point(begin, size=0.05, color=carla.Color(255,0,0), life_time = lifetime)



    def set_up_inspector_camera(self, desired_transform = carla.Transform(carla.Location(0,0,100), carla.Rotation(pitch=-90, yaw=0, roll=0))):
        spectator = self.world.get_spectator()
        spectator.set_transform(desired_transform)

    def attach_spectator_above_ego(self, height):
        cam_transform = self.ego.get_transform()
        cam_transform.location.z = height
        cam_transform.rotation.pitch = -90
        self.set_up_inspector_camera(cam_transform)


    # does not fully work. Does not fully cover blue parts of the ego vehicle
    def approximate_ego_mask(self):
        mask = np.zeros((self.bev_cam_y_dim, self.bev_cam_x_dim), dtype=np.uint8)

        bb = self.ego.bounding_box.extent
        vehicle_length = bb.x * 2
        vehicle_width  = bb.y * 2

        fov = float(self.bev_cam.attributes['fov'])
        scale = (2 * self.bev_cam_height * math.tan(math.radians(fov / 2))) / self.bev_cam_x_dim


        length_px = int(1.15 * vehicle_length / scale)
        width_px  = int(1.0 *vehicle_width  / scale)

        cx = self.bev_cam_x_dim // 2
        cy = self.bev_cam_y_dim // 2

        x1 = cx - width_px  // 2
        x2 = cx + width_px  // 2
        y1 = cy - length_px // 2 
        y2 = cy + length_px // 2

        x1 = max(0, x1)
        x2 = min(self.bev_cam_x_dim, x2)
        y1 = max(0, y1)
        y2 = min(self.bev_cam_y_dim, y2)

        # Draw main body rectangle (rear 3/4 of vehicle)
        front_start = y1 - length_px //7
        body_start  = y1 + length_px //6  # front quarter is the pointed part
        mask[body_start:y2, x1:x2] = 1

        # Draw pointed front — each row gets narrower toward the tip
        for row in range(front_start, body_start):
            t = (row - front_start) / (body_start - front_start)
            row_width = int(width_px * t)
            row_cx = (x1 + x2) // 2
            row_x1 = max(0, row_cx - row_width // 2)
            row_x2 = min(self.bev_cam_x_dim, row_cx + row_width // 2)
            mask[row, row_x1:row_x2] = 1

        return mask.astype(bool)

    def get_ego_mask(self, image, search_ahead_pixels=6):

        pixel_array = np.frombuffer(image.raw_data, dtype=np.uint8).copy() # convert image to numpy array 1D
        pixel_array = pixel_array.reshape((image.height, image.width, 4)) # convert image to numpy array 2D where each value has BGRA values
        tags = pixel_array[:, :, 2]  # We only need red channel because it stores semantic tag id, rest is useless

        vehicle_mask = (tags == SEMANTIC_TAGS["car"])  # TAG_VEHICLE = 14, only find the ones with tag == TAG_VEHICLE
        labeled, _ = label(vehicle_mask) # Find each vehicle on the mask and give it a unique ID. 0 = no vehicle, 1+ means yes vehicle

        center_y, center_x = image.height // 2, image.width // 2 # Find center of image, our bev is in the center
        ego_label = labeled[center_y, center_x] # get the id of vehicle in center. that is ego
        

        #if exact center is obstructed we search forward a bit
        i = 0
        while(ego_label == 0):
            i += 1
            ego_label = labeled[center_y+i, center_x]
            if(i > search_ahead_pixels):
                break

        

         # return mask where ego pixels are true, rest are false. if none we return none, may happen on first frame when vehicle just spawns for some reason!
        if ego_label > 0:
            return (labeled == ego_label) 
        else:
            return None #self.approximate_ego_mask()
    

    def world_to_bev_pixel(self, world_location):
        """
        Convert a world coordinate to BEV camera pixel coordinates.
        Accounts for FOV, camera height, camera yaw rotation, and image dimensions.
        
        Returns (px, py) or None if the point is outside the camera's view.
        """
        ego_transform = self.ego.get_transform()
        ego_loc = ego_transform.location
        cam_yaw = ego_transform.rotation.yaw  # camera rotates with ego

        # Vector from camera to point in world space
        dx = world_location.x - ego_loc.x
        dy = world_location.y - ego_loc.y

        # Rotate vector into camera space (account for camera yaw)
        yaw_rad = math.radians(cam_yaw)
        cam_x =  dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
        cam_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)

        # Scale from world space to pixel space using FOV and height
        fov = float(self.bev_cam.attributes['fov'])
        scale = (2 * self.bev_cam_height * math.tan(math.radians(fov / 2))) / self.bev_cam_x_dim

        px = int(self.bev_cam_x_dim / 2 + cam_y / scale)
        py = int(self.bev_cam_y_dim / 2 - cam_x / scale)

    
        # Return None if outside image bounds
        if not (0 <= px < self.bev_cam_x_dim and 0 <= py < self.bev_cam_y_dim):
            return None

        return (px, py)
                


    def draw_circle_for_bev(self, x, y, radius, target):
        for ox in range(-radius, radius+1):
            for oy in range(-radius, radius+1):
                if ox**2 + oy**2 <= radius**2:  # circle with radius 3
                    nx, ny = x + ox, y + oy
                    if 0 <= nx < self.bev_cam_x_dim and 0 <= ny <  self.bev_cam_y_dim :
                        target[ny, nx] = 1


    def get_astar_route_mask(self, route, point_frequency):
        """Creates a binary 2D mask with 1s where waypoints are projected"""
        mask = np.zeros(( self.bev_cam_y_dim ,  self.bev_cam_x_dim ), dtype=np.uint8)

        for i in range(len(route)):
            wp = route[i]
            next_wp = wp
            if i < len(route)-1:
                next_wp = route[i+1]



            result = self.world_to_bev_pixel(wp.transform.location)

            if(result is None):
                continue

            px, py = result


            result = self.world_to_bev_pixel(next_wp.transform.location)


            if(result is None):
                next_px, next_py = px,py
            else:
                next_px, next_py = result


            for j in range(point_frequency):
                ix, iy = lerp(px, py, next_px, next_py, j/point_frequency)
                self.draw_circle_for_bev(int(ix), int(iy), 3, mask)

        return mask

    
    #depreciated, do not use. use save_tensor_visualization instead
    def save_humanized_image(self, image, ego_mask, percent_to_save=0.2):
        if(random.random() > percent_to_save):
            return
        image.convert(carla.ColorConverter.CityScapesPalette)

        pixel_array = np.frombuffer(image.raw_data, dtype=np.uint8).copy() 
        pixel_array = pixel_array.reshape((image.height, image.width, 4))[:, :, :3]
        if ego_mask is None:
            print("ego mask is none!")
        if ego_mask is not None:
            pixel_array[ego_mask] = self.ego_bev_rgb
        
        humanized_image = pixel_array[:, :, ::-1]
        Image.fromarray(humanized_image).save(f'{self.bev_output_folder}/frame_{image.frame:06d}.png')
    
    def update_bev_onehot_tensor(self, image, ego_mask, route_mask, save_for_debug_percent = 0.2):
        pixel_array = np.frombuffer(image.raw_data, dtype=np.uint8).copy() #get raw data from image as 1D array
        pixel_array = pixel_array.reshape((image.height, image.width, 4)) #turn it into 2D array

        tags_array = pixel_array[:, :, 2].copy() # only keep red channel. discard the rest. thats where tags are

        # set ego vehicle to ego tag
        if ego_mask is not None:
            tags_array[ego_mask] = SEMANTIC_TAGS['ego']
    
        
        tensor = torch.from_numpy(tags_array).long() # create a tensor from our tags_array
        one_hot = torchfunc.one_hot(tensor, num_classes=len(SEMANTIC_TAGS)) # one hot encode it, every value will just be a true at its own layer. This will improve learning accuracy
        one_hot = one_hot.permute(2, 0, 1).float() # format the tensor to be ready for learning

        # add A* layer
        if route_mask is not None:
            route_tensor = torch.from_numpy(route_mask).float().unsqueeze(0)
            one_hot[SEMANTIC_TAGS['route']] = torch.from_numpy(route_mask).float()



        # save results
        self.bev_onehot_tensor = one_hot
        if(random.random() < save_for_debug_percent):
            save_tensor_visualization(self.bev_onehot_tensor, f'{self.bev_output_folder}/tensor_{image.frame:06d}.png')

    def bev_cam_callback(self, image):
        ego_mask = self.get_ego_mask(image)
        route_mask = self.get_astar_route_mask(self.route, point_frequency = 4)
        self.update_bev_onehot_tensor(image, ego_mask, route_mask)
        #self.save_humanized_image(image, ego_mask)

    

    def make_bev_camera_bp(self, bev_params):
        bev_cam_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bev_cam_bp.set_attribute('image_size_x', bev_params['dim_x'])
        bev_cam_bp.set_attribute('image_size_y', bev_params['dim_y'])
        bev_cam_bp.set_attribute('fov', bev_params['fov'])

        self.ego_bev_rgb = bev_params["ego_bev_rgb"]
        self.bev_cam_bp = bev_cam_bp
        self.bev_cam_height = bev_params["height"]
        self.bev_cam_transform = carla.Transform(carla.Location(x=0, y=0, z=self.bev_cam_height), carla.Rotation(pitch = -90, yaw = 0, roll= 0))

        self.bev_onehot_tensor = None

    def spawn_bev_cam(self):
        self.bev_cam = self.world.spawn_actor(self.bev_cam_bp, self.bev_cam_transform , attach_to=self.ego)
        self.bev_cam.listen(self.bev_cam_callback)
        self.bev_cam_x_dim = int(self.bev_cam.attributes['image_size_x'])
        self.bev_cam_y_dim = int(self.bev_cam.attributes['image_size_y'])



    

    def reset(self, seed=None, options={}):

        # Clear sensor objects

        


        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.camera2_sensor = None
        self.camera3_sensor = None
        self.camera4_sensor = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors()


        # Spawn Ego
        while True:
          carla_map = self.world.get_map()


          if(carla_map == None):
            print("ERROR, map could not be retrieved")

          if(len(self.vehicle_spawn_points) == 0):
            print("ERROR, no spawn points found") 

          # Choose a random starting location (point A)
          point_a = random.choice(self.vehicle_spawn_points)

          # Choose a random destination (point B)
          point_b = random.choice(self.vehicle_spawn_points)
          while point_b.location == point_a.location:
              point_b = random.choice(spawn_points)

          start_waypoint = carla_map.get_waypoint(point_a.location)
          end_waypoint = carla_map.get_waypoint(point_b.location)

          self.route = a_star(self.world, start_waypoint, end_waypoint)
          
          v = self.world.try_spawn_actor(self.ego_bp, point_a)

          if v is not None:
            self.ego = v
            self.things.append(v)
            break
          print("Spawn Ego has failed")
        



        # Spawn BEV

        self.spawn_bev_cam()


        
        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles

        assert(len(self.vehicle_spawn_points) >= self.number_of_vehicles +1)


        

        while count > 0:
          v = self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4])
          if v != False and v != None:
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
        print("___ lidar spawned")
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

        self.disable_unneeded_layers()




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

        self.attach_spectator_above_ego(self.spectator_height)
        self.draw_a_start_path_in_simulation(self.route)

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
