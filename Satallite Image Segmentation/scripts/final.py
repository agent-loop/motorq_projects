import cv2
import numpy as np
from heapq import heappush, heappop
import os
from math import radians, cos, sin, asin, sqrt

class Node:
    def __init__(self, position, g_cost, h_cost):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = None

    def __lt__(self, other):
        return self.f_cost < other.f_cost

def get_color_weight(pixel):
    color_weights = {
        'green': {'color': [0, 255, 0], 'weight': 3},  # Forest
        'red': {'color': [0, 0, 255], 'weight': 5},    # Buildings
        'yellow': {'color': [0, 255, 255], 'weight': 7}, # Road
        'blue': {'color': [255, 0, 0], 'weight': 2},   # River
        'brown': {'color': [42, 42, 165], 'weight': 4}, # Land
        'black': {'color': [0, 0, 0], 'weight': 7}     # Rest
    }

    min_dist = float('inf')
    weight = 6  # Default weight (black)
    
    for color_info in color_weights.values():
        dist = np.sum(np.abs(pixel - color_info['color']))
        if dist < min_dist:
            min_dist = dist
            weight = color_info['weight']
    
    return weight

def heuristic(a, b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def get_neighbors(pos, image_shape):
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                 (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    for dx, dy in directions:
        new_pos = (pos[0] + dx, pos[1] + dy)
        if 0 <= new_pos[0] < image_shape[0] and 0 <= new_pos[1] < image_shape[1]:
            neighbors.append(new_pos)
    return neighbors

def find_path(image, start, end):
    open_set = []
    closed_set = set()
    
    start_node = Node(start, 0, heuristic(start, end))
    heappush(open_set, start_node)
    
    nodes = {start: start_node}
    
    while open_set:
        current = heappop(open_set)
        
        if current.position == end:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        
        closed_set.add(current.position)
        
        for neighbor_pos in get_neighbors(current.position, image.shape):
            if neighbor_pos in closed_set:
                continue
                
            weight = get_color_weight(image[neighbor_pos])
            g_cost = current.g_cost + weight * heuristic(current.position, neighbor_pos)
            h_cost = heuristic(neighbor_pos, end)
            
            neighbor_node = nodes.get(neighbor_pos)
            
            if neighbor_node is None:
                neighbor_node = Node(neighbor_pos, g_cost, h_cost)
                neighbor_node.parent = current
                nodes[neighbor_pos] = neighbor_node
                heappush(open_set, neighbor_node)
            elif g_cost < neighbor_node.g_cost:
                neighbor_node.g_cost = g_cost
                neighbor_node.f_cost = g_cost + h_cost
                neighbor_node.parent = current
    
    return None

def convert_pixel_to_global(pixel_coord, start_pixel, start_global, end_pixel, end_global):
    x_pixel, y_pixel = pixel_coord
    x_start_pixel, y_start_pixel = start_pixel
    x_end_pixel, y_end_pixel = end_pixel
    x_start_global, y_start_global = start_global
    x_end_global, y_end_global = end_global
    
    if (x_end_pixel - x_start_pixel) != 0:
        x_ratio = (x_pixel - x_start_pixel) / (x_end_pixel - x_start_pixel)
    else:
        x_ratio = 0
    
    if (y_end_pixel - y_start_pixel) != 0:
        y_ratio = (y_pixel - y_start_pixel) / (y_end_pixel - y_start_pixel)
    else:
        y_ratio = 0
    
    x_global = x_start_global + x_ratio * (x_end_global - x_start_global)
    y_global = y_start_global + y_ratio * (y_end_global - y_start_global)
    
    return (x_global, y_global)

def calculate_global_distance(coord1, coord2):
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of Earth in meters
    
    return c * r

def find_point_coordinates(image, target_color, tolerance=5):
    lower_bound = target_color - tolerance
    upper_bound = target_color + tolerance
    
    mask = cv2.inRange(image, lower_bound, upper_bound)
    coordinates = np.where(mask == 255)
    
    if len(coordinates[0]) > 0:
        return (coordinates[0][0], coordinates[1][0])
    return None

def draw_path_with_coordinates(image, path, start_pixel, start_global, end_pixel, end_global):
    result = image.copy()
    vertex_coords = []
    total_distance = 0
    last_global = None
    
    for i, point in enumerate(path):
        global_coord = convert_pixel_to_global(
            (point[1], point[0]),
            (start_pixel[1], start_pixel[0]),
            start_global,
            (end_pixel[1], end_pixel[0]),
            end_global
        )
        
        if last_global is not None:
            distance = calculate_global_distance(last_global, global_coord)
            total_distance += distance
        
        last_global = global_coord
        vertex_coords.append((point, global_coord))
        
        if i < len(path) - 1:
            next_point = path[i + 1]
            cv2.line(result, (point[1], point[0]), (next_point[1], next_point[0]), (255, 255, 255), 2)
        
        cv2.circle(result, (point[1], point[0]), 3, (0, 255, 255), -1)
        # label = f"({global_coord[0]:.6f}, {global_coord[1]:.6f})"
        # cv2.putText(result, label, (point[1]+5, point[0]-5), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 1)
    
    return result, vertex_coords, total_distance

def main():
    # Define global coordinates (replace with actual coordinates)
    start_global = (longitude1, latitude1)
    end_global = (longitude2, latitude2)
    
    # Load image
    image_path = "C:\\loopster\\Path_Plan\\scripts\\Untitled.png"  # Update with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    print(f"Image loaded successfully. Shape: {image.shape}")
    
    # Find start and end points
    start_color = np.array([85, 131, 150])  # BGR format
    end_color = np.array([150, 121, 25])
    
    start = find_point_coordinates(image, start_color, tolerance=5)
    end = find_point_coordinates(image, end_color, tolerance=5)
    
    if start is None or end is None:
        print("Could not find start or end points")
        return
    
    print(f"Start point found at: {start}")
    print(f"End point found at: {end}")
    
    # Find path
    path = find_path(image, start, end)
    
    if path:
        # Draw path with coordinates and get vertex information
        result_image, vertex_coords, total_distance = draw_path_with_coordinates(
            image.copy(), path, start, start_global, end, end_global)
        
        # Save result image
        output_path = os.path.join("output", 'final_path_image.png')
        cv2.imwrite(output_path, result_image)
        
        # Save coordinates to text file
        text_output = os.path.join("output", 'path_coordinates.txt')
        with open(text_output, 'w') as f:
            f.write(f"Total path distance: {total_distance:.2f} meters\n\n")
            f.write("Vertex Coordinates (Pixel, Global):\n")
            for pixel_coord, global_coord in vertex_coords:
                f.write(f"Pixel: ({pixel_coord[0]}, {pixel_coord[1]}) -> ")
                f.write(f"Global: ({global_coord[0]:.6f}, {global_coord[1]:.6f})\n")
        
        print(f"Path found and saved to {output_path}")
        print(f"Coordinates saved to {text_output}")
        print(f"Total distance: {total_distance:.2f} meters")
    else:
        print("No path found")

if __name__ == "__main__":
    # Replace these coordinates with your actual global coordinates
    longitude1, latitude1 = 73.8567, 18.5204  # Start point
    longitude2, latitude2 = 51.8587, 23.5224  # End point
    main()