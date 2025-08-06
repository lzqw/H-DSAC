#TODO: self.NAVI_POINT_DIST=50

import math
import sys
import os
import time
import copy
import json
from typing import List
import numpy as np
import rclpy
from rclpy.node import Node
from car_interfaces.msg import *
import pygame
import math
ros_node_name = "surrounding_info"
sys.path.append(os.getcwd() + "/src/utils/")
sys.path.append(os.getcwd() + "/src/%s/%s/"%(ros_node_name, ros_node_name))
from surrounding_utils import Obstacle, compute_radar_distances, parse_obstacle_data, update_obstacles
VISUALIZATION = True
NUM_RAYS=240
MAX_DISTANCE=30.0
LOCAL_LEN=100
MAX_SPEED=30
PI = 3.1415926535
import tjitools

class SurroundingInfo(Node):
    def __init__(self):
        super().__init__(ros_node_name)

        # define publishers
        self.pubSurroundingInfo = self.create_publisher(SurroundingInfoInterface, "surrounding_info_data", 10)
        self.timerSurroundingInfo= self.create_timer(0.1, self.pub_callback_surrounding_info)


        # define subscribers
        self.subFusion = self.create_subscription(FusionInterface, "fusion_data", self.sub_callback_fusion, 1)
        self.subGlobalPathPlanning = self.create_subscription(GlobalPathPlanningInterface, "global_path_planning_data", self.sub_callback_global_path_planning, 10)

        self.rcvMsgFusion = None
        self.glbPath = None
        self.is_turning = False         
        self.speed_max = 0                                                    

        if VISUALIZATION:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Vehicle Radar Simulation - Moving Obstacles")
            self.clock = pygame.time.Clock()
            self.running = True

        tjitools.ros_log(self.get_name(), 'Start Node:%s'%self.get_name())

        

    def pub_callback_surrounding_info(self):
        """
        Callback of publisher (timer), publish the topic surrounding_info_data.
        :param None.
        """
        msgSurroundingInfo= SurroundingInfoInterface()
        now_ts = time.time()
        msgSurroundingInfo.timestamp = now_ts

        radar_distances = [MAX_DISTANCE] * NUM_RAYS
        turn_signals=0.0
        path_rfu_=[]
        if self.rcvMsgFusion is not None:

            msgSurroundingInfo.carlength=self.rcvMsgFusion.carlength
            msgSurroundingInfo.carwidth=self.rcvMsgFusion.carwidth
            msgSurroundingInfo.carheight=self.rcvMsgFusion.carheight
            msgSurroundingInfo.carspeed=self.rcvMsgFusion.carspeed
            msgSurroundingInfo.steerangle=self.rcvMsgFusion.steerangle
            msgSurroundingInfo.throttle_percentage=self.rcvMsgFusion.throttle_percentage
            msgSurroundingInfo.braking_percentage=self.rcvMsgFusion.braking_percentage
            msgSurroundingInfo.braketq=self.rcvMsgFusion.braketq
            msgSurroundingInfo.gearpos=self.rcvMsgFusion.gearpos
            msgSurroundingInfo.car_run_mode=self.rcvMsgFusion.car_run_mode
            print(self.rcvMsgFusion.centeroffset)
            if VISUALIZATION:
                delta_time = self.clock.get_time() / 1000.0
                self.clock.tick(FPS)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            if len(self.rcvMsgFusion.obstacledata)>0:
                obstacles = parse_obstacle_data(self.rcvMsgFusion.obstacledata)

                # Compute radar distances based on updated obstacle positions
                radar_distances = compute_radar_distances(obstacles,num_rays=NUM_RAYS,max_distance=MAX_DISTANCE)        

            radar_distances=[d/MAX_DISTANCE for d in radar_distances]
            # radar_distances/=MAX_DISTANCE
            # print(radar_distances)

            msgSurroundingInfo.surroundinginfo=radar_distances


            path_rfu_, turn_signals, errorYaw,errorDistance = self._get_local_path()
            msgSurroundingInfo.turn_signals=turn_signals
            msgSurroundingInfo.error_yaw=errorYaw
            msgSurroundingInfo.error_distance=errorDistance
            path_rfu_send= path_rfu_[:,0:2]
            msgSurroundingInfo.path_rfu = path_rfu_send.flatten().tolist()
            if VISUALIZATION:
                self.screen.fill(COLOR_BACKGROUND)
                draw_radar_lines(self.screen)
                draw_vehicle(self.screen)
                draw_radar(self.screen, radar_distances)
                if len(self.rcvMsgFusion.obstacledata)>0:                   
                    draw_obstacles(self.screen, obstacles)
                draw_reference_path(self.screen, path_rfu_, turn_signals)
                pygame.display.flip()

        msgSurroundingInfo.process_time = time.time() - now_ts

        self.pubSurroundingInfo.publish(msgSurroundingInfo)
        
        tjitools.ros_log(self.get_name(), 'Publish surrounding_info msg !!!')

    def _get_local_path(self):
        lon_ = self.rcvMsgFusion.longitude
        lat_ = self.rcvMsgFusion.latitude
        vel_ = self.rcvMsgFusion.carspeed
        yaw_ = self.rcvMsgFusion.yaw
        nowYaw = self.rcvMsgFusion.yaw/180*PI
        err_lon_ = self.glbPath[:,0] - lon_
        err_lat_ = self.glbPath[:,1] - lat_
        err_ = err_lon_**2 + err_lat_**2
        now_idx = err_.argmin()

        tjitools.set_gps_org(lon_,lat_,0)
        
        if now_idx + LOCAL_LEN < len(self.glbPath):
            loc_path_end = self.glbPath[now_idx+LOCAL_LEN, 0:2]
            loc_path_lon = self.glbPath[now_idx:now_idx+LOCAL_LEN, 0]
            loc_path_lat = self.glbPath[now_idx:now_idx+LOCAL_LEN, 1]
            loc_path_vel = self.glbPath[now_idx:now_idx+LOCAL_LEN, 2]
            loc_path_ang = self.glbPath[now_idx:now_idx+LOCAL_LEN, 3]
        else:
            loc_path_end = self.glbPath[-1, 0:2]
            loc_path_lon = self.glbPath[now_idx:, 0]
            loc_path_lat = self.glbPath[now_idx:, 1]
            loc_path_vel = self.glbPath[now_idx:, 2]
            loc_path_ang = self.glbPath[now_idx:, 3]

            pad_len = LOCAL_LEN - len(loc_path_lon)
            loc_path_lon = np.pad(loc_path_lon, pad_width=(0, pad_len), mode='edge')
            loc_path_lat = np.pad(loc_path_lat, pad_width=(0, pad_len), mode='edge')
            loc_path_vel = np.pad(loc_path_vel, pad_width=(0, pad_len), mode='edge')
            loc_path_ang = np.pad(loc_path_ang, pad_width=(0, pad_len), mode='edge')

            loc_path_vel[-pad_len:] = 0
        self.speed_max = np.mean(loc_path_vel[1:31])
        error_ang = loc_path_ang[0] - loc_path_ang[80]
        if error_ang > 180:
            error_ang = error_ang -360
        if error_ang <-180:
            error_ang = error_ang +360

        if error_ang > 30:
            turn_signals = 1.0
        elif error_ang < -30:
            turn_signals = -1.0
        else:
            turn_signals = 0.0

        gps_ = np.array([loc_path_lon, loc_path_lat, np.zeros_like(loc_path_lat)]).swapaxes(0,1)
        car_gps_ = np.array([lon_, lat_, 0])
        car_rot_ = np.array([0, 0, yaw_])

        path_rfu_ = tjitools.gps_to_rfu(gps_, car_gps_, car_rot_)
        # path_rfu_= path_rfu_[0:2,]
        # print(path_rfu_)

        nowPosX, nowPosY, nowPosZ = self.conversion_of_coordinates(lat_, lon_, 0)
        advanceDistance = 10
        refLatitude = loc_path_lat[advanceDistance]
        refLongitude = loc_path_lon[advanceDistance]
        refYaw = loc_path_ang[advanceDistance]/180*PI
        refPosX, refPosY, refPosz= self.conversion_of_coordinates(refLatitude, refLongitude, 0)
        errorYaw=-nowYaw+refYaw

        if(errorYaw > PI):
            errorYaw=errorYaw - 2*PI
        if(errorYaw < -PI):
            errorYaw=errorYaw + 2*PI
        print("ephi=", errorYaw)
        errorDistance = ((-nowPosY+refPosY)*math.cos(refYaw)-(-nowPosX+refPosX)*math.sin(refYaw))
        
        print("ed=", errorDistance)
        return path_rfu_, turn_signals, errorYaw, errorDistance
    
    def conversion_of_coordinates(self, conversionLatitude, conversionLongitude, conversionAltitude):
        
        #WGS84 参数定义
        wgs84_a  = 6378137.0
        wgs84_f  = 1/298.257223565
        wgs84_e2 = wgs84_f * (2-wgs84_f)
        D2R      = PI / 180.0

        #局部坐标原点经纬度

        #天南街原点
        Latitude0 = 36.65538784
        Longitude0 = 114.58287107
        # Latitude0 = 39.1052154
        # Longitude0 = 117.1641687


        lat0 = Latitude0  *D2R
        lon0 = Longitude0 *D2R
        alt0 = 0
        N0 = wgs84_a / math.sqrt(1 - wgs84_e2 * math.sin(lat0)*math.sin(lat0))
        x0 = (N0 + alt0) * math.cos(lat0) * math.cos(lon0)
        y0 = (N0 + alt0) * math.cos(lat0) * math.sin(lon0)
        z0 = (N0*(1-wgs84_e2) + alt0) * math.sin(lat0)

        lat = conversionLatitude  *D2R
        lon = conversionLongitude *D2R
        alt = conversionAltitude
        N = wgs84_a / math.sqrt(1 - wgs84_e2 * math.sin(lat)*math.sin(lat))
        x = (N + alt) * math.cos(lat) * math.cos(lon)
        y = (N + alt) * math.cos(lat) * math.sin(lon)
        z = (N*(1-wgs84_e2) + alt) * math.sin(lat)

        dx = x - x0
        dy = y - y0
        dz = z - z0

        s11 = -math.sin(lon0)
        s12 = math.cos(lon0)
        s13 = 0
        s21 = -math.sin(lat0) * math.cos(lon0)
        s22 = -math.sin(lat0) * math.sin(lon0)
        s23 = math.cos(lat0)
        s31 = math.cos(lat0) * math.cos(lon0)
        s32 = math.cos(lat0) * math.sin(lon0)
        s33 = math.sin(lat0)

        posX = s11 * dx + s12 * dy + s13 * dz
        posY = s21 * dx + s22 * dy + s23 * dz
        posZ = s31 * dx + s32 * dy + s33 * dz

        return posX, posY, posZ

    def sub_callback_fusion(self, msgFusion:FusionInterface):
        """
        Callback of subscriber, subscribe the topic fusion_data.
        :param msgFusion: The message heard by the subscriber.
        """
        self.rcvMsgFusion = msgFusion
        pass

    def sub_callback_global_path_planning(self, msgGlobalPathPlanning:GlobalPathPlanningInterface):
        """
        Callback of subscriber, subscribe the topic global_path_planning_data.
        :param msgGlobalPathPlanning: The message heard by the subscriber.
        """
        msgGlbPath = msgGlobalPathPlanning
        self.glbPath = np.array(msgGlbPath.routedata)
        self.glbPath = self.glbPath.reshape(-1,4)
        pass


# Configuration parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 30

# Coordinate transformation parameters
SCALE = 8  # 1 meter = 8 pixels
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2

# Color definitions (RGB)
COLOR_BACKGROUND = (30, 30, 30)  # Dark gray background
COLOR_VEHICLE = (0, 255, 0)  # Green vehicle
COLOR_OBSTACLE = (255, 0, 0)  # Red obstacles (default)
COLOR_RADAR_POINT = (0, 0, 255)  # Blue radar points
COLOR_RADAR_LINE = (50, 50, 50)  # Gray radar lines
COLOR_PATH_LEFT = (255, 0, 0)  # Red for left turn
COLOR_PATH_STRAIGHT = (0, 255, 0)  # Green for straight
COLOR_PATH_RIGHT = (0, 0, 255)  # Blue for right turn

def world_to_screen(x: float, y: float) -> (int, int):
    """
    Convert world coordinates (meters) to screen coordinates (pixels).
    :param x: X coordinate in meters
    :param y: Y coordinate in meters
    :return: Tuple of (screen_x, screen_y) in pixels
    """
    screen_x = CENTER_X + int(x * SCALE)
    screen_y = CENTER_Y - int(y * SCALE)  # Invert Y-axis for screen coordinates
    return screen_x, screen_y


def draw_vehicle(screen):
    """
    Draw the vehicle at the center of the screen.
    :param screen: Pygame screen surface
    """
    vehicle_radius = 10
    pygame.draw.circle(screen, COLOR_VEHICLE, (CENTER_X, CENTER_Y), vehicle_radius)


def draw_obstacles(screen, obstacles: List[Obstacle]):
    """
    Draw all obstacles on the screen.
    :param screen: Pygame screen surface
    :param obstacles: List of Obstacle instances
    """
    for obs in obstacles:
        screen_x, screen_y = world_to_screen(obs.x, obs.y)

        # Choose color based on obstacle category
        if obs.category == 1:
            color = (255, 0, 0)  # Car - Red
        elif obs.category == 3:
            color = (0, 255, 255)  # Pedestrian - Cyan
        elif obs.category == 2:
            color = (255, 165, 0)  # Truck - Orange
        elif obs.category == 8:
            color = (255, 255, 0)  # Traffic Light - Yellow
        else:
            color = COLOR_OBSTACLE  # Default - Red

        # Calculate obstacle size in pixels
        obstacle_length = int(obs.length * SCALE)
        obstacle_width = int(obs.width * SCALE)

        # Ensure minimum size for visibility
        obstacle_length = max(obstacle_length, 5)
        obstacle_width = max(obstacle_width, 5)

        # Create a rectangle centered at (screen_x, screen_y)
        rect = pygame.Rect(0, 0, obstacle_length, obstacle_width)
        rect.center = (screen_x, screen_y)
        pygame.draw.rect(screen, color, rect)


def draw_radar(screen, radar_distances: List[float]):
    """
    Draw radar points based on radar distances.
    :param screen: Pygame screen surface
    :param radar_distances: List of radar distances for each ray
    """
    num_rays = len(radar_distances)
    angle_per_ray = 360.0 / num_rays
    for i in range(num_rays):
        distance = radar_distances[i]
        distance*=30
        if distance > 30.0:
            distance = 30.0  # Cap the maximum distance to 50 meters
        # Calculate the angle in degrees and radians
        angle_deg = i * angle_per_ray
        angle_rad = math.radians(angle_deg)
        # Calculate the point coordinates in meters
        x = distance * math.cos(angle_rad)
        y = distance * math.sin(angle_rad)
        # Convert to screen coordinates
        screen_x, screen_y = world_to_screen(x, y)

        # Draw the radar point (small circle with radius 2)
        pygame.draw.circle(screen, COLOR_RADAR_POINT, (screen_x, screen_y), 2)


def draw_radar_lines(screen):
    """
    Draw radar scanning circles and lines for visual reference.
    :param screen: Pygame screen surface
    """
    # Draw the maximum detection range circle
    pygame.draw.circle(screen, COLOR_RADAR_LINE, (CENTER_X, CENTER_Y), int(30 * SCALE), 1)

    # Draw each radar scanning line
    num_rays = 240
    angle_per_ray = 360.0 / num_rays
    for i in range(num_rays):
        angle_deg = i * angle_per_ray
        angle_rad = math.radians(angle_deg)
        x = 30 * math.cos(angle_rad) * SCALE
        y = 30 * math.sin(angle_rad) * SCALE
        end_x = CENTER_X + int(x)
        end_y = CENTER_Y - int(y)
        pygame.draw.line(screen, COLOR_RADAR_LINE, (CENTER_X, CENTER_Y), (end_x, end_y), 1)

def draw_reference_path(screen, path: List[List[float]], turn_signals: List[int]):
    """
    Draw the reference path with different colors based on turn signals.
    :param screen: Pygame screen surface
    :param path: List of [x, y, z] points representing the reference path
    :param turn_signals: List of turn signals corresponding to each path segment
                         -1: Left Turn, 0: Straight, 1: Right Turn
    """
    for i in range(len(path) - 1):
        # Get current and next point
        current_point = path[i]
        next_point = path[i + 1]

        # Get the turn signal for the current segment
        turn_signal = turn_signals

        # Choose color based on turn signal
        if turn_signal == -1:
            color = COLOR_PATH_LEFT  # Red for left turn
        elif turn_signal == 0:
            color = COLOR_PATH_STRAIGHT  # Green for straight
        elif turn_signal == 1:
            color = COLOR_PATH_RIGHT  # Blue for right turn
        else:
            color = (255, 255, 255)  # White for undefined signals

        # Convert world coordinates to screen coordinates
        start_pos = world_to_screen(current_point[0], current_point[1])
        end_pos = world_to_screen(next_point[0], next_point[1])

        # Draw the line segment
        pygame.draw.line(screen, color, start_pos, end_pos, 3)  # Thickness of 3 pixels


def main():
    rclpy.init()
    rosNode = SurroundingInfo()
    rclpy.spin(rosNode)
    rclpy.shutdown()

