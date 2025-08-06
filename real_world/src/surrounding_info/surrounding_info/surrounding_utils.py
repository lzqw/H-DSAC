
import math
import random
from typing import List
import pygame

class Obstacle:
    def __init__(self, category: int, length: float, width: float, height: float,
                 x: float, y: float, z: float, v: float, latv: float):
        self.category = category
        self.length = length
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.z = z
        self.v = v
        self.latv = latv

    @staticmethod
    def from_list(data: List[float]):
        """
        Create an Obstacle instance from a list of data.
        :param data: List containing obstacle data in the order:
                     [category, length, width, height, x, y, z, v, latv]
        :return: Obstacle instance
        """
        category = int(data[0])
        length = data[1]
        width = data[2]
        height = data[3]
        x = data[4]
        y = data[5]
        z = data[6]
        v = data[7]
        latv = data[8]
        return Obstacle(category, length, width, height, x, y, z, v, latv)

    def update_position(self, delta_time: float):
        """
        Update the obstacle's position based on its velocity.
        :param delta_time: Time increment in seconds
        """
        self.x += self.v * delta_time
        self.y += self.latv * delta_time
        # Optional: Update z-axis position or other attributes if needed


def parse_obstacle_data(obstacledata: List[float]) -> List[Obstacle]:
    """
    Parse the one-dimensional obstacle data array into a list of Obstacle instances.
    :param obstacledata: List containing obstacle data
    :return: List of Obstacle instances
    """
    obstacles = []
    num_obstacles = len(obstacledata) // 9
    for i in range(num_obstacles):
        data_slice = obstacledata[i * 9: (i + 1) * 9]
        obstacle = Obstacle.from_list(data_slice)
        obstacles.append(obstacle)
    return obstacles


def compute_radar_distances(obstacles: List[Obstacle],
                            num_rays: int = 240,
                            max_distance: float = 50.0,
                            height_min: float = -5.0,
                            height_max: float = 10.0) -> List[float]:
    """
    Compute the nearest distance detected by each radar ray based on the obstacles.
    :param obstacles: List of Obstacle instances
    :param num_rays: Number of radar rays (default: 240)
    :param max_distance: Maximum detection distance in meters (default: 50.0)
    :param height_min: Minimum valid height of obstacles in meters (default: -5.0)
    :param height_max: Maximum valid height of obstacles in meters (default: 10.0)
    :return: List containing the nearest distance for each radar ray
    """
    # Initialize radar distances with the maximum distance
    radar_distances = [max_distance for _ in range(num_rays)]

    # Calculate the angle covered by each radar ray in degrees
    angle_per_ray = 360.0 / num_rays  # 1.5 degrees per ray

    for obstacle in obstacles:
        # Check if the obstacle's height is within the valid range
        if not (height_min <= obstacle.z <= height_max):
            continue  # Skip obstacles with invalid height

        # Calculate the Euclidean distance from the vehicle to the obstacle in the XY plane
        distance = math.hypot(obstacle.x, obstacle.y)
        if distance == 0:
            continue  # Avoid division by zero

        # Calculate the angle of the obstacle relative to the vehicle in degrees
        angle = math.degrees(math.atan2(obstacle.y, obstacle.x))  # Range: [-180, 180]
        if angle < 0:
            angle += 360  # Convert to [0, 360] range

        # Calculate the angular width of the obstacle based on its width and distance
        angular_width = math.degrees(math.atan2(obstacle.width / 2, distance))
        min_angle = angle - angular_width
        max_angle = angle + angular_width

        # Ensure angles are within [0, 360] degrees
        min_angle = max(min_angle, 0)
        max_angle = min(max_angle, 360)

        # Determine the range of radar rays affected by the obstacle
        start_ray = int(min_angle / angle_per_ray)
        end_ray = int(max_angle / angle_per_ray)

        # Update the radar distances for affected rays
        for ray in range(start_ray, end_ray + 1):
            if ray >= num_rays:
                ray -= num_rays  # Handle wrap-around
            if distance < radar_distances[ray]:
                radar_distances[ray] = min(distance, max_distance)

    return radar_distances


def simulate_radar(obstacledata: List[float]) -> List[float]:
    """
    Simulate radar by parsing obstacle data and computing radar distances.
    :param obstacledata: List containing obstacle data
    :return: List of radar distances for each ray
    """
    # Parse obstacle data into Obstacle instances
    obstacles = parse_obstacle_data(obstacledata)

    # Compute radar distances based on current obstacles
    radar_distances = compute_radar_distances(obstacles)

    return radar_distances

def update_obstacles(obstacles: List[Obstacle], delta_time: float):
    """
    Update the positions of all obstacles and handle those that move out of range.
    :param obstacles: List of Obstacle instances
    :param delta_time: Time increment in seconds
    """
    for obs in obstacles:
        obs.update_position(delta_time)

        # Check if the obstacle is out of the radar detection range
        distance = math.hypot(obs.x, obs.y)
        if distance > 60.0:  # 50 meters detection range + 10 meters buffer
            # Reset obstacle to a random position within 5 to 50 meters
            angle = random.uniform(0, 360)
            angle_rad = math.radians(angle)
            new_distance = random.uniform(5, 50)  # Between 5m and 50m
            obs.x = new_distance * math.cos(angle_rad)
            obs.y = new_distance * math.sin(angle_rad)
            # Assign random velocities
            obs.v = random.uniform(-2.0, 2.0)  # Radial velocity between -2 and 2 m/s
            obs.latv = random.uniform(-2.0, 2.0)  # Lateral velocity between -2 and 2 m/s



