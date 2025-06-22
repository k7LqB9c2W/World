#this is a simulation game and KEEP IN MIND IT MUST BE MADE AND OPTIMZIED FOR SIMULATING MILLIONS OF HUANS!
#PLEASE READ THIS. ENSURE THAT THE CODE IS OPTIMIZED FOR SIMULATING MILLIONS OF HUANS!
# ALSO READ THIS. PLEASE ENSURE THAT THE START IS ALWAYS A BLANK SLATE, JUST OCEAN WATER! DO NOT SPAWN IN ANY GRASS OR SUCH.
import pygame
import numpy as np
import random
import time
import math
import numba
from numba import jit, prange
import heapq # For priority queue in A* and resource finding
import os # For CPU count
from typing import List, Optional

# --- Core Game Configuration ---
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
WORLD_WIDTH_TILES = 512
WORLD_HEIGHT_TILES = 512
TILE_SIZE = 16
MIN_TILE_SIZE = 2
MAX_TILE_SIZE = 32
FPS = 60
MAX_HUMANS_ESTIMATE = 1_000_000 # Target
MAX_TASKS = 200_000 # Max concurrent tasks
MAX_PATH_NODES = MAX_HUMANS_ESTIMATE * 50 # Pre-allocate memory for paths

# --- UI Configuration ---
UI_FONT_SIZE = 16
UI_PANEL_HEIGHT = 120
UI_INFO_PANEL_WIDTH = 300
UI_BG_COLOR = (25, 35, 45)
UI_BORDER_COLOR = (120, 140, 160)
UI_TEXT_COLOR = (230, 230, 255)
UI_HIGHLIGHT_COLOR = (255, 180, 90)

# --- Definitions ---
# Terrain Types
TERRAIN_WATER, TERRAIN_GRASS, TERRAIN_SAND, TERRAIN_FOREST, TERRAIN_MOUNTAIN, TERRAIN_FARM, TERRAIN_ROAD = 0, 1, 2, 3, 4, 5, 6
TERRAIN_COLORS = {
    TERRAIN_WATER: (64, 100, 225), TERRAIN_GRASS: (80, 160, 50), TERRAIN_SAND: (210, 200, 120),
    TERRAIN_FOREST: (30, 110, 40), TERRAIN_MOUNTAIN: (140, 140, 140), TERRAIN_FARM: (160, 130, 70),
    TERRAIN_ROAD: (100, 100, 100)
}
TERRAIN_NAMES = {
    TERRAIN_WATER: "Water", TERRAIN_GRASS: "Grass", TERRAIN_SAND: "Sand",
    TERRAIN_FOREST: "Forest", TERRAIN_MOUNTAIN: "Mountain", TERRAIN_FARM: "Farm", TERRAIN_ROAD: "Road"
}
TERRAIN_PASSABLE = np.array([False, True, True, True, False, True, True], dtype=np.bool_)
TERRAIN_BUILDABLE = np.array([False, True, True, False, False, False, False], dtype=np.bool_)

# Resource Types
RESOURCE_WOOD, RESOURCE_STONE, RESOURCE_FOOD = 0, 1, 2
NUM_RESOURCE_TYPES = 3
RESOURCE_NAMES = {RESOURCE_WOOD: "Wood", RESOURCE_STONE: "Stone", RESOURCE_FOOD: "Food"}

# Building Types
BUILDING_NONE, BUILDING_HOUSE, BUILDING_TOWNHALL, BUILDING_GRANARY, BUILDING_LUMBERMILL, BUILDING_STONEMASON = 0, 1, 2, 3, 4, 5
BUILDING_COLORS = {
    BUILDING_HOUSE: (150, 80, 40), BUILDING_TOWNHALL: (200, 180, 50),
    BUILDING_GRANARY: (180, 150, 90), BUILDING_LUMBERMILL: (100, 70, 30),
    BUILDING_STONEMASON: (130, 130, 130)
}
BUILDING_NAMES = {
    BUILDING_HOUSE: "House", BUILDING_TOWNHALL: "Town Hall", BUILDING_GRANARY: "Granary",
    BUILDING_LUMBERMILL: "Lumber Mill", BUILDING_STONEMASON: "Stone Mason"
}
# Which buildings serve as drop-off points for which resources
max_b_type_dropoff = max(BUILDING_GRANARY, BUILDING_LUMBERMILL, BUILDING_STONEMASON)
building_dropoff_map = np.zeros((max_b_type_dropoff + 1, NUM_RESOURCE_TYPES), dtype=np.int8)
building_dropoff_map[BUILDING_GRANARY, RESOURCE_FOOD] = 1
building_dropoff_map[BUILDING_LUMBERMILL, RESOURCE_WOOD] = 1
building_dropoff_map[BUILDING_STONEMASON, RESOURCE_STONE] = 1

# Building Costs
BUILDING_COSTS = {
    BUILDING_HOUSE: {RESOURCE_WOOD: 10},
    BUILDING_TOWNHALL: {RESOURCE_WOOD: 50, RESOURCE_STONE: 20},
    BUILDING_GRANARY: {RESOURCE_WOOD: 20, RESOURCE_STONE: 5},
    BUILDING_LUMBERMILL: {RESOURCE_WOOD: 30},
    BUILDING_STONEMASON: {RESOURCE_STONE: 30},
}
max_b_type_cost = max(BUILDING_COSTS.keys())
building_costs_array = np.zeros((max_b_type_cost + 1, NUM_RESOURCE_TYPES), dtype=np.int16)
for b_type, costs in BUILDING_COSTS.items():
    for res_type, amount in costs.items():
        building_costs_array[b_type, res_type] = amount

# Tools
TOOL_TERRAIN, TOOL_SPAWN_HUMAN, TOOL_SPAWN_TREE, TOOL_SPAWN_STONE, TOOL_FIRE, TOOL_ERASE, TOOL_INSPECT, TOOL_BUILD = 0, 1, 2, 3, 4, 5, 6, 7
TOOL_NAMES = {
    TOOL_TERRAIN: "Terrain Paint", TOOL_SPAWN_HUMAN: "Spawn Human", TOOL_SPAWN_TREE: "Spawn Tree",
    TOOL_SPAWN_STONE: "Spawn Stone", TOOL_FIRE: "Start Fire", TOOL_ERASE: "Eraser",
    TOOL_INSPECT: "Inspector", TOOL_BUILD: "Place Building"
}

# Human Data Array Indices (SOA - Structure of Arrays)
_h_idx = 0
def _human_attr(): global _h_idx; val = _h_idx; _h_idx += 1; return val
H_ID = _human_attr()
H_X, H_Y = _human_attr(), _human_attr()
H_SUB_X, H_SUB_Y = _human_attr(), _human_attr()
H_STATE = _human_attr()
H_HUNGER, H_ENERGY = _human_attr(), _human_attr()
H_AGE = _human_attr()
H_TARGET_X, H_TARGET_Y = _human_attr(), _human_attr()
H_ACTION_TIMER = _human_attr()
H_PATH_HEAD_IDX = _human_attr()
H_TASK_ID = _human_attr()
H_HOME_BUILDING_ID = _human_attr()
H_TARGET_BUILDING_ID = _human_attr()
NUM_HUMAN_ATTRIBUTES = _h_idx

# Human States
S_IDLE, S_MOVING, S_GATHERING, S_BUILDING, S_EATING, S_HAULING, S_DEAD = range(7)
STATE_NAMES = {
    S_IDLE: "Idle", S_MOVING: "Moving", S_GATHERING: "Gathering",
    S_BUILDING: "Building", S_EATING: "Eating", S_HAULING: "Hauling", S_DEAD: "Dead"
}

# --- Task System Definitions ---
# Task Array Indices
T_TYPE, T_X, T_Y, T_STATUS, T_CLAIMANT_ID, T_PARAM_1 = range(6)
NUM_TASK_ATTRIBUTES = 6
# Task Types
TASK_NONE, TASK_WANDER, TASK_GATHER_WOOD, TASK_GATHER_STONE, TASK_GATHER_FOOD, TASK_BUILD_HOUSE = range(6)
# Task Status
TASK_STATUS_INACTIVE, TASK_STATUS_OPEN, TASK_STATUS_CLAIMED = 0, 1, 2

# --- Path Node Definitions ---
# Path Node Array Indices (Singly Linked List in a NumPy array)
P_NEXT_IDX, P_X, P_Y = 0, 1, 2
NUM_PATH_NODE_ATTRIBUTES = 3

# =================================================================================
#
# NEW NUMBA-JITTED CORE FUNCTIONS
#
# =================================================================================

@jit(nopython=True)
def _numba_pathfinding_astar(start_x, start_y, target_x, target_y, pathfinding_cost_map, width, height):
    """
    A Numba-jitted A* pathfinding implementation.
    Returns a NumPy array of path coordinates (or an empty one if no path is found).
    """
    open_set = [(0.0, start_x, start_y)]
    
    came_from = np.full((width, height, 2), -1, dtype=np.int32)
    g_score = np.full((width, height), np.inf, dtype=np.float32)
    g_score[start_x, start_y] = 0
    
    f_score = np.full((width, height), np.inf, dtype=np.float32)
    f_score[start_x, start_y] = np.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)

    max_steps = 1000
    steps = 0
    
    while len(open_set) > 0 and steps < max_steps:
        steps += 1
        current_f, current_x, current_y = open_set.pop(0)

        if current_x == target_x and current_y == target_y:
            path = []
            curr_x, curr_y = target_x, target_y
            while curr_x != -1:
                path.append((curr_x, curr_y))
                curr_x, curr_y = came_from[curr_x, curr_y, 0], came_from[curr_x, curr_y, 1]
            return np.array(path[::-1], dtype=np.int32)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                
                neighbor_x, neighbor_y = current_x + dx, current_y + dy

                if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                    cost_multiplier = 1.414 if dx != 0 and dy != 0 else 1.0
                    path_cost = pathfinding_cost_map[neighbor_x, neighbor_y]
                    
                    if path_cost == np.inf:
                        continue

                    tentative_g_score = g_score[current_x, current_y] + path_cost * cost_multiplier
                    
                    if tentative_g_score < g_score[neighbor_x, neighbor_y]:
                        came_from[neighbor_x, neighbor_y] = (current_x, current_y)
                        g_score[neighbor_x, neighbor_y] = tentative_g_score
                        heuristic = np.sqrt((target_x - neighbor_x)**2 + (target_y - neighbor_y)**2)
                        new_f_score = tentative_g_score + heuristic
                        f_score[neighbor_x, neighbor_y] = new_f_score
                        
                        for i in range(len(open_set)):
                            if new_f_score < open_set[i][0]:
                                open_set.insert(i, (new_f_score, neighbor_x, neighbor_y))
                                break
                        else:
                            open_set.append((new_f_score, neighbor_x, neighbor_y))
                            
    return np.empty((0, 2), dtype=np.int32)

@jit(nopython=True)
def _numba_add_path_to_storage(path_coords, path_nodes, next_free_path_node_idx):
    """Adds a path to the shared path_nodes array and returns the head index."""
    if len(path_coords) == 0:
        return -1
    
    num_new_nodes = len(path_coords)
    if next_free_path_node_idx + num_new_nodes >= len(path_nodes):
        return -1

    path_head_idx = next_free_path_node_idx
    
    for i in range(num_new_nodes):
        current_node_idx = next_free_path_node_idx + i
        next_node_idx = current_node_idx + 1 if i < num_new_nodes - 1 else -1
        
        path_nodes[current_node_idx, P_NEXT_IDX] = next_node_idx
        path_nodes[current_node_idx, P_X] = path_coords[i, 0]
        path_nodes[current_node_idx, P_Y] = path_coords[i, 1]
        
    return path_head_idx

@jit(nopython=True, parallel=True)
def _numba_update_movement(human_data, path_nodes, dt):
    """A hyper-optimized movement function that traverses the linked-list path structure."""
    ticks_per_tile = 20.0
    move_speed_subtile = (1.0 / ticks_per_tile) * dt * 60.0

    for i in prange(len(human_data)):
        path_head = int(human_data[i, H_PATH_HEAD_IDX])
        if path_head == -1:
            continue

        target_step_x = path_nodes[path_head, P_X]
        target_step_y = path_nodes[path_head, P_Y]
        
        current_tile_x = int(human_data[i, H_X])
        current_tile_y = int(human_data[i, H_Y])
        
        if current_tile_x == target_step_x and current_tile_y == target_step_y:
            human_data[i, H_PATH_HEAD_IDX] = path_nodes[path_head, P_NEXT_IDX]
            human_data[i, H_SUB_X] = 0.5
            human_data[i, H_SUB_Y] = 0.5
            continue

        sub_x = human_data[i, H_SUB_X]
        sub_y = human_data[i, H_SUB_Y]
        
        dx = (target_step_x + 0.5) - (current_tile_x + sub_x)
        dy = (target_step_y + 0.5) - (current_tile_y + sub_y)

        dist = math.sqrt(dx*dx + dy*dy)

        if dist > 0.01:
            move_x = (dx / dist) * move_speed_subtile
            move_y = (dy / dist) * move_speed_subtile

            new_sub_x = sub_x + move_x
            new_sub_y = sub_y + move_y

            if new_sub_x >= 1.0:
                human_data[i, H_X] += 1
                new_sub_x -= 1.0
            elif new_sub_x < 0.0:
                human_data[i, H_X] -= 1
                new_sub_x += 1.0
            
            if new_sub_y >= 1.0:
                human_data[i, H_Y] += 1
                new_sub_y -= 1.0
            elif new_sub_y < 0.0:
                human_data[i, H_Y] -= 1
                new_sub_y += 1.0
            
            human_data[i, H_SUB_X] = new_sub_x
            human_data[i, H_SUB_Y] = new_sub_y
            
@jit(nopython=True, parallel=True)
def _numba_update_ai_and_tasks(
    human_data, human_inventories, tasks, world_terrain, world_resources, 
    world_buildings, world_building_types, building_costs, building_dropoffs, 
    pathfinding_cost_map, dt, tick
):
    """The master AI function. Replaces _ai_update_chunk."""
    num_humans = len(human_data)
    
    if tick % 5 == 0:
        for i in range(len(tasks)):
            if tasks[i, T_STATUS] == TASK_STATUS_OPEN:
                tasks[i, T_STATUS] = TASK_STATUS_INACTIVE

        next_task_idx = 0
        for x in range(world_terrain.shape[0]):
            for y in range(world_terrain.shape[1]):
                if next_task_idx >= len(tasks): break
                
                if world_terrain[x, y] == TERRAIN_FOREST and world_resources[x, y, RESOURCE_WOOD] > 0:
                    tasks[next_task_idx, T_TYPE] = TASK_GATHER_WOOD
                    tasks[next_task_idx, T_X], tasks[next_task_idx, T_Y] = x, y
                    tasks[next_task_idx, T_STATUS] = TASK_STATUS_OPEN
                    tasks[next_task_idx, T_CLAIMANT_ID] = -1
                    next_task_idx += 1
                
                if world_terrain[x, y] == TERRAIN_MOUNTAIN and world_resources[x, y, RESOURCE_STONE] > 0:
                    if next_task_idx >= len(tasks): break
                    tasks[next_task_idx, T_TYPE] = TASK_GATHER_STONE
                    tasks[next_task_idx, T_X], tasks[next_task_idx, T_Y] = x, y
                    tasks[next_task_idx, T_STATUS] = TASK_STATUS_OPEN
                    tasks[next_task_idx, T_CLAIMANT_ID] = -1
                    next_task_idx += 1
                
                if world_terrain[x, y] == TERRAIN_FARM:
                    if next_task_idx >= len(tasks): break
                    tasks[next_task_idx, T_TYPE] = TASK_GATHER_FOOD
                    tasks[next_task_idx, T_X], tasks[next_task_idx, T_Y] = x, y
                    tasks[next_task_idx, T_STATUS] = TASK_STATUS_OPEN
                    tasks[next_task_idx, T_CLAIMANT_ID] = -1
                    next_task_idx += 1

    for i in prange(num_humans):
        state = int(human_data[i, H_STATE])
        
        human_data[i, H_HUNGER] = min(100.0, human_data[i, H_HUNGER] + 0.05 * dt)
        human_data[i, H_ENERGY] = max(0.0, human_data[i, H_ENERGY] - 0.02 * dt)
        human_data[i, H_ACTION_TIMER] = max(0, human_data[i, H_ACTION_TIMER] - 1)
        
        if human_data[i, H_ACTION_TIMER] > 0:
            continue
            
        if human_data[i, H_HUNGER] > 80 and state != S_EATING:
            if human_inventories[i, RESOURCE_FOOD] > 0:
                human_data[i, H_STATE] = S_EATING
                human_data[i, H_ACTION_TIMER] = 50
                human_inventories[i, RESOURCE_FOOD] -= 1
                human_data[i, H_HUNGER] -= 40
            else:
                human_data[i, H_STATE] = S_IDLE
                human_data[i, H_TASK_ID] = -1
            continue

        is_moving = human_data[i, H_PATH_HEAD_IDX] != -1
        
        if state == S_IDLE:
            best_task_id = -1
            min_dist_sq = 1e9
            hx, hy = int(human_data[i, H_X]), int(human_data[i, H_Y])

            task_priority = TASK_GATHER_FOOD
            if human_inventories[i, RESOURCE_FOOD] >= 5 and human_inventories[i, RESOURCE_WOOD] < 20:
                task_priority = TASK_GATHER_WOOD
            elif human_inventories[i, RESOURCE_FOOD] >= 5 and human_inventories[i, RESOURCE_STONE] < 10:
                task_priority = TASK_GATHER_STONE
            
            for j in range(len(tasks)):
                if tasks[j, T_STATUS] == TASK_STATUS_OPEN and tasks[j, T_TYPE] == task_priority:
                    dist_sq = (tasks[j, T_X] - hx)**2 + (tasks[j, T_Y] - hy)**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_task_id = j
            
            if best_task_id != -1:
                tasks[best_task_id, T_STATUS] = TASK_STATUS_CLAIMED
                tasks[best_task_id, T_CLAIMANT_ID] = human_data[i, H_ID]
                human_data[i, H_TASK_ID] = best_task_id
                human_data[i, H_STATE] = S_MOVING
                human_data[i, H_TARGET_X] = tasks[best_task_id, T_X]
                human_data[i, H_TARGET_Y] = tasks[best_task_id, T_Y]
        
        elif state == S_MOVING:
            if not is_moving:
                task_id = int(human_data[i, H_TASK_ID])
                if task_id != -1:
                    task_type = int(tasks[task_id, T_TYPE])
                    if task_type in (TASK_GATHER_FOOD, TASK_GATHER_WOOD, TASK_GATHER_STONE):
                        human_data[i, H_STATE] = S_GATHERING
                        human_data[i, H_ACTION_TIMER] = 100
                else: # Arrived at drop-off or other non-task target
                    # Check if at a drop-off point
                    tx, ty = int(human_data[i, H_X]), int(human_data[i, H_Y])
                    b_type = world_building_types[tx, ty]
                    if b_type > 0 and b_type < len(building_dropoffs):
                        # Drop off all resources
                        for res_type in range(NUM_RESOURCE_TYPES):
                            if building_dropoffs[b_type, res_type] == 1:
                                # A real sim would add to building storage. We just make it disappear.
                                human_inventories[i, res_type] = 0
                    human_data[i, H_STATE] = S_IDLE

        elif state == S_GATHERING:
            task_id = int(human_data[i, H_TASK_ID])
            if task_id != -1:
                task_type = int(tasks[task_id, T_TYPE])
                res_type = -1
                if task_type == TASK_GATHER_FOOD: res_type = RESOURCE_FOOD
                if task_type == TASK_GATHER_WOOD: res_type = RESOURCE_WOOD
                if task_type == TASK_GATHER_STONE: res_type = RESOURCE_STONE
                
                if res_type != -1:
                    human_inventories[i, res_type] += 5
                    tx, ty = int(tasks[task_id, T_X]), int(tasks[task_id, T_Y])
                    if res_type != RESOURCE_FOOD: # Farms are infinite for now
                        world_resources[tx, ty, res_type] = max(0, world_resources[tx, ty, res_type] - 5)

                human_data[i, H_STATE] = S_HAULING
                tasks[task_id, T_STATUS] = TASK_STATUS_INACTIVE
                human_data[i, H_TASK_ID] = -1

        elif state == S_HAULING:
            res_to_haul = -1
            if human_inventories[i, RESOURCE_WOOD] > 0: res_to_haul = RESOURCE_WOOD
            elif human_inventories[i, RESOURCE_STONE] > 0: res_to_haul = RESOURCE_STONE
            elif human_inventories[i, RESOURCE_FOOD] > 0: res_to_haul = RESOURCE_FOOD
            
            if res_to_haul != -1:
                hx, hy = int(human_data[i, H_X]), int(human_data[i, H_Y])
                best_building_loc = (-1, -1)
                min_dist_sq = 1e9
                
                for x in range(world_building_types.shape[0]):
                    for y in range(world_building_types.shape[1]):
                        b_type = world_building_types[x, y]
                        if b_type > 0 and b_type < len(building_dropoffs):
                            if building_dropoffs[b_type, res_to_haul] == 1:
                                dist_sq = (x - hx)**2 + (y - hy)**2
                                if dist_sq < min_dist_sq:
                                    min_dist_sq = dist_sq
                                    best_building_loc = (x, y)
                
                if best_building_loc[0] != -1:
                    human_data[i, H_TARGET_X] = best_building_loc[0]
                    human_data[i, H_TARGET_Y] = best_building_loc[1]
                    human_data[i, H_STATE] = S_MOVING
                else:
                    human_data[i, H_STATE] = S_IDLE
            else:
                 human_data[i, H_STATE] = S_IDLE

# --- Utility Functions ---
def get_distance_sq(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# --- QuadTree Class (Optimized for storing IDs/indices) ---
class QuadTree:
    def __init__(self, boundary_rect: pygame.Rect, capacity: int):
        self.boundary = boundary_rect # pygame.Rect
        self.capacity = capacity
        self.entities: list[tuple[int, int, int]] = []
        self.divided = False
        self.children: List[Optional['QuadTree']] = [None, None, None, None]

    def subdivide(self):
        x, y, w, h = self.boundary
        hw, hh = w / 2, h / 2
        self.children[0] = QuadTree(pygame.Rect(x, y, hw, hh), self.capacity)
        self.children[1] = QuadTree(pygame.Rect(x + hw, y, hw, hh), self.capacity)
        self.children[2] = QuadTree(pygame.Rect(x, y + hh, hw, hh), self.capacity)
        self.children[3] = QuadTree(pygame.Rect(x + hw, y + hh, hw, hh), self.capacity)
        self.divided = True
        for entity_tuple in self.entities:
            for child in self.children:
                if child and child.insert(entity_tuple):
                    break
        self.entities = []

    def insert(self, entity_tuple: tuple[int, int, int]) -> bool:
        ex, ey, _ = entity_tuple
        if not self.boundary.collidepoint(ex, ey):
            return False

        if len(self.entities) < self.capacity and not self.divided:
            self.entities.append(entity_tuple)
            return True
        
        if not self.divided:
            self.subdivide()

        for child in self.children:
            if child and child.insert(entity_tuple):
                return True
        return False

    def query_rect(self, query_rect: pygame.Rect, found_entities: list[int]):
        if not self.boundary.colliderect(query_rect):
            return

        for ex, ey, eid in self.entities:
            if query_rect.collidepoint(ex, ey):
                found_entities.append(eid)
        
        if self.divided:
            for child in self.children:
                if child:
                    child.query_rect(query_rect, found_entities)
    
    def clear(self):
        self.entities = []
        self.divided = False
        self.children = [None, None, None, None]

# --- ResourceManager (Largely a placeholder now, logic moved to Numba) ---
class ResourceManager:
    def __init__(self, world_data):
        self.world_data = world_data
        # The dynamic task system in Numba has replaced the need for these sets.

# --- World Data (SoA approach for terrain and resources) ---
class WorldData:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.terrain_map = np.full((width, height), TERRAIN_GRASS, dtype=np.uint8)
        self.resource_map = np.zeros((width, height, NUM_RESOURCE_TYPES), dtype=np.uint16)
        self.building_map = np.zeros((width, height), dtype=np.int32)
        self.building_type_map = np.zeros((width, height), dtype=np.uint8)
        self.fire_map = np.zeros((width, height), dtype=np.uint8)
        self.pathfinding_cost_map = np.ones((width, height), dtype=np.float32)
        self.update_pathfinding_cost_map()

    def update_pathfinding_cost_map(self):
        # This function is critical and remains unchanged.
        self.pathfinding_cost_map.fill(1.0)
        self.pathfinding_cost_map[self.terrain_map == TERRAIN_WATER] = np.inf
        self.pathfinding_cost_map[self.terrain_map == TERRAIN_MOUNTAIN] = np.inf
        self.pathfinding_cost_map[self.terrain_map == TERRAIN_ROAD] = 0.5
        self.pathfinding_cost_map[self.terrain_map == TERRAIN_FOREST] = 1.5
        self.pathfinding_cost_map[self.terrain_map == TERRAIN_SAND] = 1.2
        self.pathfinding_cost_map[self.building_map != 0] = np.inf

# --- Human Manager (Handles all human data and logic) ---
class HumanManager:
    def __init__(self, world_data):
        self.world_data = world_data
        self.max_humans = 0
        self.num_active_humans = 0
        self.tick_counter = 0

        self.human_data = np.empty((0, NUM_HUMAN_ATTRIBUTES), dtype=np.float32)
        self.human_inventories = np.empty((0, NUM_RESOURCE_TYPES), dtype=np.int16)
        
        self.path_nodes = np.full((MAX_PATH_NODES, NUM_PATH_NODE_ATTRIBUTES), -1, dtype=np.int32)
        self.next_free_path_node_idx = 0
        
        self.tasks = np.full((MAX_TASKS, NUM_TASK_ATTRIBUTES), -1, dtype=np.int32)
        self.tasks[:, T_STATUS] = TASK_STATUS_INACTIVE
        
        self.next_human_id = 1
        self.human_id_to_idx = {}
        self.idx_to_human_id = {}
        self.path_cache = {}

    def clear_path_cache(self):
        print("Pathfinding cache cleared due to world changes.")
        self.path_cache = {}

    def _resize_arrays(self, new_capacity):
        print(f"Resizing human arrays to {new_capacity}")
        new_data = np.zeros((new_capacity, NUM_HUMAN_ATTRIBUTES), dtype=np.float32)
        if self.num_active_humans > 0:
            new_data[:self.num_active_humans] = self.human_data
        self.human_data = new_data

        new_inv = np.zeros((new_capacity, NUM_RESOURCE_TYPES), dtype=np.int16)
        if self.num_active_humans > 0:
            new_inv[:self.num_active_humans] = self.human_inventories
        self.human_inventories = new_inv
        
        self.max_humans = new_capacity

    def add_human(self, x, y):
        if self.num_active_humans == self.max_humans:
            new_cap = self.max_humans * 2 if self.max_humans > 0 else 128
            self._resize_arrays(new_cap)

        idx = self.num_active_humans
        human_id = self.next_human_id
        self.next_human_id += 1
        self.human_id_to_idx[human_id] = idx
        self.idx_to_human_id[idx] = human_id

        self.human_data[idx, H_ID] = human_id
        self.human_data[idx, H_X], self.human_data[idx, H_Y] = x, y
        self.human_data[idx, H_SUB_X], self.human_data[idx, H_SUB_Y] = 0.5, 0.5
        self.human_data[idx, H_STATE] = S_IDLE
        self.human_data[idx, H_HUNGER] = random.uniform(0, 30)
        self.human_data[idx, H_ENERGY] = 100
        self.human_data[idx, H_AGE] = 0
        self.human_data[idx, H_TARGET_X], self.human_data[idx, H_TARGET_Y] = -1, -1
        self.human_data[idx, H_ACTION_TIMER] = 0
        self.human_data[idx, H_PATH_HEAD_IDX] = -1
        self.human_data[idx, H_TASK_ID] = -1
        self.human_data[idx, H_HOME_BUILDING_ID] = -1
        self.human_data[idx, H_TARGET_BUILDING_ID] = -1
        
        self.human_inventories[idx, :] = 0
        self.num_active_humans += 1
        return human_id

    def remove_human_by_id(self, human_id_to_remove):
        if human_id_to_remove not in self.human_id_to_idx: return

        idx_to_remove = self.human_id_to_idx[human_id_to_remove]
        last_active_idx = self.num_active_humans - 1
        if idx_to_remove != last_active_idx:
            last_human_id = self.idx_to_human_id[last_active_idx]
            self.human_data[idx_to_remove] = self.human_data[last_active_idx]
            self.human_inventories[idx_to_remove] = self.human_inventories[last_active_idx]
            self.human_id_to_idx[last_human_id] = idx_to_remove
            self.idx_to_human_id[idx_to_remove] = last_human_id

        self.human_data[last_active_idx, :] = 0 
        self.human_inventories[last_active_idx, :] = 0
        del self.human_id_to_idx[human_id_to_remove]
        del self.idx_to_human_id[last_active_idx]
        self.num_active_humans -= 1

    def update_simulation(self, dt):
        if self.num_active_humans == 0:
            return
            
        active_humans_data = self.human_data[:self.num_active_humans]
        active_inventories = self.human_inventories[:self.num_active_humans]

        _numba_update_ai_and_tasks(
            active_humans_data, active_inventories, self.tasks,
            self.world_data.terrain_map, self.world_data.resource_map,
            self.world_data.building_map, self.world_data.building_type_map,
            building_costs_array, building_dropoff_map,
            self.world_data.pathfinding_cost_map, dt, self.tick_counter
        )
        
        for i in range(self.num_active_humans):
            if active_humans_data[i, H_STATE] == S_MOVING and active_humans_data[i, H_PATH_HEAD_IDX] == -1:
                start_x, start_y = int(active_humans_data[i, H_X]), int(active_humans_data[i, H_Y])
                target_x, target_y = int(active_humans_data[i, H_TARGET_X]), int(active_humans_data[i, H_TARGET_Y])

                if start_x == target_x and start_y == target_y: continue

                path_key = ((start_x, start_y), (target_x, target_y))
                if path_key in self.path_cache:
                    path_coords = self.path_cache[path_key]
                else:
                    path_coords = _numba_pathfinding_astar(
                        start_x, start_y, target_x, target_y,
                        self.world_data.pathfinding_cost_map,
                        self.world_data.width, self.world_data.height
                    )
                    self.path_cache[path_key] = path_coords

                if len(path_coords) > 0:
                    path_head = _numba_add_path_to_storage(path_coords, self.path_nodes, self.next_free_path_node_idx)
                    if path_head != -1:
                        active_humans_data[i, H_PATH_HEAD_IDX] = path_head
                        self.next_free_path_node_idx += len(path_coords)
                    else:
                        active_humans_data[i, H_STATE] = S_IDLE
                else:
                    active_humans_data[i, H_STATE] = S_IDLE
                    task_id = int(active_humans_data[i, H_TASK_ID])
                    if task_id != -1:
                        self.tasks[task_id, T_STATUS] = TASK_STATUS_OPEN

        _numba_update_movement(active_humans_data, self.path_nodes, dt)

        if self.next_free_path_node_idx > MAX_PATH_NODES * 0.9:
            print("Path node memory is high, performing garbage collection.")
            # This is a simple but potentially slow garbage collection.
            # A better approach would be a free-list.
            used_nodes = set(active_humans_data[:, H_PATH_HEAD_IDX])
            # A full GC would trace all linked lists, which is slow.
            # For now, we just reset if no paths are active.
            if len(used_nodes) <= 1 and -1 in used_nodes:
                 print("Resetting path node memory.")
                 self.next_free_path_node_idx = 0

        self.tick_counter += 1

    def get_human_info(self, human_id):
        if human_id not in self.human_id_to_idx: return None
        idx = self.human_id_to_idx[human_id]
        info = {
            "ID": int(self.human_data[idx, H_ID]),
            "Pos": (int(self.human_data[idx, H_X]), int(self.human_data[idx, H_Y])),
            "State": STATE_NAMES.get(int(self.human_data[idx, H_STATE]), "Unknown"),
            "Hunger": f"{self.human_data[idx, H_HUNGER]:.1f}",
            "Energy": f"{self.human_data[idx, H_ENERGY]:.1f}",
            "Wood": self.human_inventories[idx, RESOURCE_WOOD],
            "Food": self.human_inventories[idx, RESOURCE_FOOD],
            "Stone": self.human_inventories[idx, RESOURCE_STONE],
            "Target": (int(self.human_data[idx, H_TARGET_X]), int(self.human_data[idx, H_TARGET_Y])) if self.human_data[idx, H_TARGET_X] != -1 else "None",
            "Path Head": int(self.human_data[idx, H_PATH_HEAD_IDX]),
            "Task ID": int(self.human_data[idx, H_TASK_ID]),
        }
        return info

    def shutdown_threads(self):
        pass

# --- Building Manager (Placeholder) ---
class BuildingManager:
    def __init__(self, world_data):
        self.world_data = world_data
        self.buildings = {}
        self.next_building_id = 1

    def add_building(self, x, y, building_type, owner_human_id):
        if self.world_data.building_map[x,y] != 0: return None
        
        building_id = self.next_building_id
        self.next_building_id += 1
        
        self.buildings[building_id] = {
            'id': building_id, 'type': building_type, 
            'x': x, 'y': y, 'owner_human_id': owner_human_id,
        }
        self.world_data.building_map[x,y] = building_id
        self.world_data.building_type_map[x,y] = building_type
        self.world_data.update_pathfinding_cost_map()
        
        print(f"Building {BUILDING_NAMES[building_type]} (ID: {building_id}) placed at ({x},{y})")
        return building_id

    def get_building_info(self, building_id):
        if building_id not in self.buildings: return None
        b = self.buildings[building_id]
        return {
            "ID": b['id'],
            "Type": BUILDING_NAMES.get(b['type'], "Unknown"),
            "Pos": (b['x'], b['y']),
            "Owner": b['owner_human_id']
        }

# --- Village Manager (Basic Placeholder) ---
class VillageManager:
    def __init__(self, world_data):
        self.world_data = world_data
        self.villages = {}
        self.next_village_id = 1

    def get_village_info_at(self, x, y):
        building_id = self.world_data.building_map[x,y]
        if building_id != 0:
            for vid, vdata in self.villages.items():
                if building_id in vdata.get('building_ids', []):
                    return self.get_village_info(vid)
        return None

    def get_village_info(self, village_id):
        if village_id not in self.villages: return None
        v = self.villages[village_id]
        return {
            "ID": v['id'],
            "Name": v.get('name', f"Village {v['id']}"),
            "Population": len(v.get('members', [])),
            "Buildings": len(v.get('building_ids', []))
        }

# --- Game Class ---
class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("ProtoWorld - Optimized Simulation")
        self.clock = pygame.time.Clock()
        
        self.world_data = WorldData(WORLD_WIDTH_TILES, WORLD_HEIGHT_TILES)
        self.village_manager = VillageManager(self.world_data)
        self.human_manager = HumanManager(self.world_data)
        self.building_manager = BuildingManager(self.world_data)
        
        self.running = True
        self.camera_x_tile = WORLD_WIDTH_TILES // 2
        self.camera_y_tile = WORLD_HEIGHT_TILES // 2
        self.tile_size_render = TILE_SIZE

        self.painting = False
        self.current_tool = TOOL_TERRAIN
        self.paint_terrain_type = TERRAIN_GRASS
        self.paint_brush_size = 1
        self.build_tool_type = BUILDING_HOUSE

        self.font_small = pygame.font.SysFont("Consolas", UI_FONT_SIZE)
        self.font_medium = pygame.font.SysFont("Consolas", UI_FONT_SIZE + 4)
        self.font_large = pygame.font.SysFont("Consolas", UI_FONT_SIZE + 8)

        self.selected_entity_info = None
        self.selected_tile_info = None

        self.debug_mode = False
        self.paused = False
        
        self.world_surface = pygame.Surface((WORLD_WIDTH_TILES * self.tile_size_render, 
                                             WORLD_HEIGHT_TILES * self.tile_size_render)).convert()
        self.world_surface_needs_redraw = True

        self.generate_initial_world()

        self.human_quadtree = QuadTree(pygame.Rect(0, 0, WORLD_WIDTH_TILES, WORLD_HEIGHT_TILES), 4)

    def redraw_tile_on_world_surface(self, x, y):
        if not (0 <= x < self.world_data.width and 0 <= y < self.world_data.height):
            return
        if self.world_surface is None:
            return
        
        if self.tile_size_render * WORLD_WIDTH_TILES != self.world_surface.get_width() or \
           self.tile_size_render * WORLD_HEIGHT_TILES != self.world_surface.get_height():
            self.world_surface_needs_redraw = True
            return

        terrain_type = int(self.world_data.terrain_map[x,y])
        color = TERRAIN_COLORS[terrain_type]
        
        if terrain_type == TERRAIN_FOREST:
            wood_amount = self.world_data.resource_map[x,y,RESOURCE_WOOD]
            shade = max(0.3, min(1.0, wood_amount / 150.0))
            color = (int(color[0]*shade), int(color[1]*shade), int(color[2]*shade))
        
        if terrain_type == TERRAIN_MOUNTAIN:
            stone_amount = self.world_data.resource_map[x,y,RESOURCE_STONE]
            shade = max(0.4, min(1.0, stone_amount / 200.0))
            color = (int(color[0]*shade), int(color[1]*shade), int(color[2]*shade))

        rect = (x * self.tile_size_render, y * self.tile_size_render, 
                self.tile_size_render, self.tile_size_render)
        pygame.draw.rect(self.world_surface, color, rect)

        building_type_on_tile = int(self.world_data.building_type_map[x,y])
        if building_type_on_tile != BUILDING_NONE:
            b_color = BUILDING_COLORS.get(building_type_on_tile, (200,0,200))
            b_rect_inset = max(1, self.tile_size_render // 8)
            b_rect = (x * self.tile_size_render + b_rect_inset, 
                      y * self.tile_size_render + b_rect_inset,
                      self.tile_size_render - 2 * b_rect_inset,
                      self.tile_size_render - 2 * b_rect_inset)
            pygame.draw.rect(self.world_surface, b_color, b_rect)
            if self.tile_size_render > 8:
                 pygame.draw.rect(self.world_surface, (50,50,50), b_rect, 1)

    def generate_initial_world(self):
        print("Generating initial world...")
        self.world_data.terrain_map[:,:] = TERRAIN_WATER
        self.world_data.resource_map[:,:,:] = 0 
        
        self.world_data.update_pathfinding_cost_map()
        self.world_surface_needs_redraw = True

    def run(self):
        while self.running:
            start_frame_time = time.perf_counter()
            
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 1.0 / 20.0)

            self.handle_events()
            if not self.paused:
                self.update(dt)
            self.draw()

            frame_time_ms = (time.perf_counter() - start_frame_time) * 1000
            fps_actual = self.clock.get_fps()
            caption = (f"ProtoWorld | FPS: {fps_actual:.1f} ({frame_time_ms:.2f}ms) | "
                       f"Humans: {self.human_manager.num_active_humans} | "
                       f"Tool: {TOOL_NAMES[self.current_tool]} | Zoom: {self.tile_size_render}px")
            if self.paused: caption += " [PAUSED]"
            pygame.display.set_caption(caption)
            
        self.human_manager.shutdown_threads()
        pygame.quit()

    def handle_events(self):
        mouse_world_x, mouse_world_y = -1,-1
        mouse_pos = pygame.mouse.get_pos()
        current_screen_width, current_screen_height = self.screen.get_size()

        cam_px_x = self.camera_x_tile * self.tile_size_render
        cam_px_y = self.camera_y_tile * self.tile_size_render
        
        mouse_relative_x = mouse_pos[0] 
        mouse_relative_y = mouse_pos[1]

        if 0 <= mouse_relative_x < current_screen_width and \
           0 <= mouse_relative_y < (current_screen_height - UI_PANEL_HEIGHT):
            mouse_world_px_x = cam_px_x + mouse_relative_x
            mouse_world_px_y = cam_px_y + mouse_relative_y
            mouse_world_x = int(mouse_world_px_x // self.tile_size_render)
            mouse_world_y = int(mouse_world_px_y // self.tile_size_render)
        
        self.selected_tile_info = None
        if 0 <= mouse_world_x < self.world_data.width and 0 <= mouse_world_y < self.world_data.height:
             self.selected_tile_info = {
                 "Pos": (mouse_world_x, mouse_world_y),
                 "Terrain": TERRAIN_NAMES.get(self.world_data.terrain_map[mouse_world_x, mouse_world_y], "N/A"),
                 "Wood": self.world_data.resource_map[mouse_world_x, mouse_world_y, RESOURCE_WOOD],
                 "Stone": self.world_data.resource_map[mouse_world_x, mouse_world_y, RESOURCE_STONE],
                 "BuildingID": self.world_data.building_map[mouse_world_x, mouse_world_y]
             }

        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.WINDOWRESIZED:
                self.world_surface_needs_redraw = True
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: self.running = False
                if event.key == pygame.K_p: self.paused = not self.paused
                if event.key == pygame.K_F1: self.debug_mode = not self.debug_mode
                
                if event.key == pygame.K_g: self.current_tool = TOOL_TERRAIN
                if event.key == pygame.K_h: self.current_tool = TOOL_SPAWN_HUMAN
                if event.key == pygame.K_t: self.current_tool = TOOL_SPAWN_TREE
                if event.key == pygame.K_r: self.current_tool = TOOL_SPAWN_STONE
                if event.key == pygame.K_f: self.current_tool = TOOL_FIRE
                if event.key == pygame.K_e: self.current_tool = TOOL_ERASE
                if event.key == pygame.K_i: self.current_tool = TOOL_INSPECT
                if event.key == pygame.K_b: self.current_tool = TOOL_BUILD

                if self.current_tool == TOOL_TERRAIN:
                    if event.key == pygame.K_1: self.paint_terrain_type = TERRAIN_GRASS
                    if event.key == pygame.K_2: self.paint_terrain_type = TERRAIN_SAND
                    if event.key == pygame.K_3: self.paint_terrain_type = TERRAIN_FOREST
                    if event.key == pygame.K_4: self.paint_terrain_type = TERRAIN_MOUNTAIN
                    if event.key == pygame.K_5: self.paint_terrain_type = TERRAIN_FARM
                    if event.key == pygame.K_6: self.paint_terrain_type = TERRAIN_ROAD
                    if event.key == pygame.K_0: self.paint_terrain_type = TERRAIN_WATER
                
                if self.current_tool == TOOL_BUILD:
                    if event.key == pygame.K_1: self.build_tool_type = BUILDING_HOUSE
                    if event.key == pygame.K_2: self.build_tool_type = BUILDING_TOWNHALL
                    if event.key == pygame.K_3: self.build_tool_type = BUILDING_GRANARY
                    if event.key == pygame.K_4: self.build_tool_type = BUILDING_LUMBERMILL
                    if event.key == pygame.K_5: self.build_tool_type = BUILDING_STONEMASON

                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS: 
                    self.paint_brush_size = min(25, self.paint_brush_size + 1)
                if event.key == pygame.K_MINUS: 
                    self.paint_brush_size = max(0, self.paint_brush_size - 1)

                if event.key == pygame.K_PAGEUP:
                    self.tile_size_render = min(MAX_TILE_SIZE, self.tile_size_render * 2)
                    self.world_surface_needs_redraw = True
                if event.key == pygame.K_PAGEDOWN:
                    self.tile_size_render = max(MIN_TILE_SIZE, self.tile_size_render // 2)
                    self.world_surface_needs_redraw = True

            if event.type == pygame.MOUSEBUTTONDOWN:
                if mouse_pos[1] < current_screen_height - UI_PANEL_HEIGHT:
                    if event.button == 1 or event.button == 3:
                        self.painting = True
                        self.use_tool_at(mouse_world_x, mouse_world_y, event.button)
                    elif event.button == 4:
                        new_center_world_px_x = self.camera_x_tile * self.tile_size_render + mouse_pos[0]
                        new_center_world_px_y = self.camera_y_tile * self.tile_size_render + mouse_pos[1]
                        self.tile_size_render = min(MAX_TILE_SIZE, int(self.tile_size_render * 1.25) + 1)
                        self.camera_x_tile = (new_center_world_px_x / self.tile_size_render) - (mouse_pos[0] / self.tile_size_render)
                        self.camera_y_tile = (new_center_world_px_y / self.tile_size_render) - (mouse_pos[1] / self.tile_size_render)
                        self.world_surface_needs_redraw = True
                    elif event.button == 5:
                        new_center_world_px_x = self.camera_x_tile * self.tile_size_render + mouse_pos[0]
                        new_center_world_px_y = self.camera_y_tile * self.tile_size_render + mouse_pos[1]
                        self.tile_size_render = max(MIN_TILE_SIZE, int(self.tile_size_render * 0.8) - 1)
                        self.camera_x_tile = (new_center_world_px_x / self.tile_size_render) - (mouse_pos[0] / self.tile_size_render)
                        self.camera_y_tile = (new_center_world_px_y / self.tile_size_render) - (mouse_pos[1] / self.tile_size_render)
                        self.world_surface_needs_redraw = True

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 or event.button == 3:
                    self.painting = False
            
            if event.type == pygame.MOUSEMOTION:
                if self.painting and (mouse_pos[1] < current_screen_height - UI_PANEL_HEIGHT):
                    button_pressed = 1 if pygame.mouse.get_pressed()[0] else (3 if pygame.mouse.get_pressed()[2] else 0)
                    if button_pressed:
                        self.use_tool_at(mouse_world_x, mouse_world_y, button_pressed)

        keys = pygame.key.get_pressed()
        cam_speed = 500.0 / self.tile_size_render * (self.clock.get_time()/1000.0)
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: self.camera_x_tile -= cam_speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.camera_x_tile += cam_speed
        if keys[pygame.K_UP] or keys[pygame.K_w]: self.camera_y_tile -= cam_speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: self.camera_y_tile += cam_speed

        vis_tiles_w = current_screen_width / self.tile_size_render
        vis_tiles_h = (current_screen_height - UI_PANEL_HEIGHT) / self.tile_size_render
        
        self.camera_x_tile = max(0, min(self.camera_x_tile, self.world_data.width - vis_tiles_w))
        self.camera_y_tile = max(0, min(self.camera_y_tile, self.world_data.height - vis_tiles_h))

    def use_tool_at(self, world_x, world_y, mouse_button):
        if not (0 <= world_x < self.world_data.width and 0 <= world_y < self.world_data.height):
            return

        brush_r = self.paint_brush_size
        tiles_to_update_pathmap = False

        for dx in range(-brush_r, brush_r + 1):
            for dy in range(-brush_r, brush_r + 1):
                if dx*dx + dy*dy > brush_r*brush_r and brush_r > 0: continue
                
                tx, ty = world_x + dx, world_y + dy
                if not (0 <= tx < self.world_data.width and 0 <= ty < self.world_data.height):
                    continue

                tile_changed_visuals = False

                if self.current_tool == TOOL_TERRAIN:
                    terrain_to_paint = self.paint_terrain_type if mouse_button == 1 else TERRAIN_GRASS
                    if self.world_data.terrain_map[tx,ty] != terrain_to_paint:
                        self.world_data.terrain_map[tx,ty] = terrain_to_paint
                        tile_changed_visuals = True
                        tiles_to_update_pathmap = True
                        if terrain_to_paint == TERRAIN_FOREST:
                            self.world_data.resource_map[tx,ty,RESOURCE_WOOD] = random.randint(50, 200)
                        elif terrain_to_paint == TERRAIN_MOUNTAIN:
                            self.world_data.resource_map[tx,ty,RESOURCE_STONE] = random.randint(100, 300)

                elif self.current_tool == TOOL_SPAWN_TREE:
                    if self.world_data.terrain_map[tx,ty] not in [TERRAIN_WATER, TERRAIN_MOUNTAIN]:
                        self.world_data.terrain_map[tx,ty] = TERRAIN_FOREST
                        self.world_data.resource_map[tx,ty,RESOURCE_WOOD] = random.randint(100,250)
                        tile_changed_visuals = True
                        tiles_to_update_pathmap = True

                elif self.current_tool == TOOL_SPAWN_STONE:
                     if self.world_data.terrain_map[tx,ty] not in [TERRAIN_WATER]:
                        self.world_data.terrain_map[tx,ty] = TERRAIN_MOUNTAIN
                        self.world_data.resource_map[tx,ty,RESOURCE_STONE] = random.randint(100,300)
                        tile_changed_visuals = True
                        tiles_to_update_pathmap = True
                
                elif self.current_tool == TOOL_FIRE:
                    if self.world_data.fire_map[tx,ty] == 0:
                         self.world_data.fire_map[tx,ty] = 200

                elif self.current_tool == TOOL_ERASE:
                    if self.world_data.building_map[tx,ty] != 0 or np.any(self.world_data.resource_map[tx,ty,:] > 0):
                        self.world_data.building_map[tx,ty] = 0
                        self.world_data.building_type_map[tx,ty] = BUILDING_NONE
                        self.world_data.resource_map[tx,ty,:] = 0
                        tile_changed_visuals = True
                        tiles_to_update_pathmap = True
                    if self.world_data.fire_map[tx,ty] != 0:
                        self.world_data.fire_map[tx,ty] = 0
                
                if tile_changed_visuals:
                    self.redraw_tile_on_world_surface(tx, ty)
        
        if tiles_to_update_pathmap:
            self.world_data.update_pathfinding_cost_map()
            self.human_manager.clear_path_cache()

        if self.current_tool == TOOL_SPAWN_HUMAN and mouse_button == 1:
            if TERRAIN_PASSABLE[self.world_data.terrain_map[world_x, world_y]]:
                self.human_manager.add_human(world_x, world_y)
        
        elif self.current_tool == TOOL_INSPECT and mouse_button == 1:
            self.selected_entity_info = None 
            query_rect = pygame.Rect(world_x - 0.5, world_y - 0.5, 1, 1)
            nearby_human_ids = []
            self.human_quadtree.query_rect(query_rect, nearby_human_ids)
            if nearby_human_ids:
                for h_id in nearby_human_ids:
                    if h_id in self.human_manager.human_id_to_idx:
                        h_idx = self.human_manager.human_id_to_idx[h_id]
                        hx_tile = int(self.human_manager.human_data[h_idx, H_X])
                        hy_tile = int(self.human_manager.human_data[h_idx, H_Y])
                        if hx_tile == world_x and hy_tile == world_y:
                            self.selected_entity_info = self.human_manager.get_human_info(h_id)
                            break 
                if self.selected_entity_info: return

            building_id = self.world_data.building_map[world_x, world_y]
            if building_id != 0:
                self.selected_entity_info = self.building_manager.get_building_info(building_id)
                if self.selected_entity_info: return
            
            village_info = self.village_manager.get_village_info_at(world_x, world_y)
            if village_info:
                self.selected_entity_info = village_info
                return

        elif self.current_tool == TOOL_BUILD and mouse_button == 1:
            if TERRAIN_BUILDABLE[self.world_data.terrain_map[world_x, world_y]] and \
               self.world_data.building_map[world_x, world_y] == 0:
                self.add_building_to_world(world_x, world_y, self.build_tool_type, owner_id=-1)
                self.redraw_tile_on_world_surface(world_x, world_y)

    def add_building_to_world(self, x, y, building_type, owner_id):
        new_b_id = self.building_manager.add_building(x,y,building_type, owner_id)
        if new_b_id:
            pass
        return new_b_id

    def update(self, dt):
        current_burning_tiles = list(np.argwhere(self.world_data.fire_map > 0))
        for x, y in current_burning_tiles:
            if self.world_data.fire_map[x, y] == 0: continue
            self.world_data.fire_map[x, y] = max(0, int(self.world_data.fire_map[x, y]) - 1)
            if self.world_data.terrain_map[x,y] == TERRAIN_FOREST and self.world_data.resource_map[x,y,RESOURCE_WOOD] > 0:
                old_wood_amount = self.world_data.resource_map[x,y,RESOURCE_WOOD]
                self.world_data.resource_map[x,y,RESOURCE_WOOD] = max(0, old_wood_amount - 5)
                if self.world_data.resource_map[x,y,RESOURCE_WOOD] != old_wood_amount:
                    self.redraw_tile_on_world_surface(x,y)
            if self.world_data.fire_map[x,y] > 0 and random.random() < 0.05:
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0: continue
                        nx, ny = x + i, y + j
                        if 0 <= nx < self.world_data.width and 0 <= ny < self.world_data.height:
                            if self.world_data.terrain_map[nx,ny] in [TERRAIN_FOREST, TERRAIN_GRASS, TERRAIN_FARM] \
                               and self.world_data.fire_map[nx,ny] == 0:
                                self.world_data.fire_map[nx,ny] = random.randint(150, 250)
        
        self.human_manager.update_simulation(dt)

        self.human_quadtree.clear()
        if self.human_manager.num_active_humans > 0:
            active_human_data = self.human_manager.human_data[:self.human_manager.num_active_humans]
            for i in range(self.human_manager.num_active_humans):
                h_id = int(active_human_data[i, H_ID])
                hx_tile = int(active_human_data[i, H_X])
                hy_tile = int(active_human_data[i, H_Y])
                self.human_quadtree.insert((hx_tile, hy_tile, h_id))

    def draw(self):
        current_screen_width, current_screen_height = self.screen.get_size()
        
        perform_full_world_redraw = False
        if self.world_surface is None or \
           self.tile_size_render * WORLD_WIDTH_TILES != self.world_surface.get_width() or \
           self.tile_size_render * WORLD_HEIGHT_TILES != self.world_surface.get_height() or \
           self.world_surface_needs_redraw:
            perform_full_world_redraw = True

        if perform_full_world_redraw:
            new_width = WORLD_WIDTH_TILES * self.tile_size_render
            new_height = WORLD_HEIGHT_TILES * self.tile_size_render
            try:
                self.world_surface = pygame.Surface((new_width, new_height)).convert()
                print(f"Full world surface redraw executing ({new_width}x{new_height})...")
                for x in range(self.world_data.width):
                    for y in range(self.world_data.height):
                        self.redraw_tile_on_world_surface(x, y)
                self.world_surface_needs_redraw = False
            except pygame.error as e:
                print(f"Error during full world_surface redraw (likely out of memory): {e}")
                if self.tile_size_render > MIN_TILE_SIZE:
                    self.tile_size_render = max(MIN_TILE_SIZE, self.tile_size_render // 2)
                    print(f"Reducing tile_size_render to {self.tile_size_render}. Redraw will be re-attempted.")
                    self.world_surface_needs_redraw = True 
                else:
                    self.world_surface = None 
        
        if self.world_surface:
            src_rect = pygame.Rect(self.camera_x_tile * self.tile_size_render, 
                                   self.camera_y_tile * self.tile_size_render,
                                   current_screen_width, 
                                   current_screen_height)
            self.screen.blit(self.world_surface, (0,0), area=src_rect)
        else:
            self.screen.fill((20,20,20))

        cam_view_x_start = int(self.camera_x_tile)
        cam_view_y_start = int(self.camera_y_tile)
        cam_view_x_end = cam_view_x_start + (current_screen_width // self.tile_size_render) + 2
        cam_view_y_end = cam_view_y_start + (current_screen_height // self.tile_size_render) + 2
        
        fire_surf = pygame.Surface((self.tile_size_render, self.tile_size_render), pygame.SRCALPHA)
        for x_tile in range(cam_view_x_start, cam_view_x_end):
            for y_tile in range(cam_view_y_start, cam_view_y_end):
                if 0 <= x_tile < self.world_data.width and 0 <= y_tile < self.world_data.height:
                    if self.world_data.fire_map[x_tile, y_tile] > 0:
                        screen_x = (x_tile - self.camera_x_tile) * self.tile_size_render
                        screen_y = (y_tile - self.camera_y_tile) * self.tile_size_render
                        intensity = self.world_data.fire_map[x_tile, y_tile] / 255.0
                        r, g, b = 255, random.randint(50, 150), 0
                        alpha = random.randint(100, int(200 * intensity))
                        fire_surf.fill((r,g,b,alpha))
                        self.screen.blit(fire_surf, (screen_x, screen_y), special_flags=pygame.BLEND_RGBA_ADD)

        query_rect_tiles = pygame.Rect(cam_view_x_start, cam_view_y_start, 
                                       cam_view_x_end - cam_view_x_start, 
                                       cam_view_y_end - cam_view_y_start)
        visible_human_ids = []
        self.human_quadtree.query_rect(query_rect_tiles, visible_human_ids)

        human_draw_radius = max(1, self.tile_size_render // 3)
        
        for h_id in visible_human_ids:
            if h_id not in self.human_manager.human_id_to_idx: continue
            idx = self.human_manager.human_id_to_idx[h_id]
            
            h_tile_x = self.human_manager.human_data[idx, H_X]
            h_tile_y = self.human_manager.human_data[idx, H_Y]
            h_sub_x = self.human_manager.human_data[idx, H_SUB_X]
            h_sub_y = self.human_manager.human_data[idx, H_SUB_Y]

            screen_tile_x = (h_tile_x - self.camera_x_tile) * self.tile_size_render
            screen_tile_y = (h_tile_y - self.camera_y_tile) * self.tile_size_render
            final_screen_x = screen_tile_x + h_sub_x * self.tile_size_render
            final_screen_y = screen_tile_y + h_sub_y * self.tile_size_render

            human_color = (230, 200, 200)
            pygame.draw.circle(self.screen, human_color, (int(final_screen_x), int(final_screen_y)), human_draw_radius)
            if self.tile_size_render > 10:
                 pygame.draw.circle(self.screen, (50,50,50), (int(final_screen_x), int(final_screen_y)), human_draw_radius, 1)

        self.draw_ui_panel(current_screen_width, current_screen_height)
        if self.selected_entity_info or self.selected_tile_info:
            self.draw_info_panel(current_screen_width, current_screen_height)

        pygame.display.flip()

    def draw_ui_panel(self, screen_w, screen_h):
        panel_rect = pygame.Rect(0, screen_h - UI_PANEL_HEIGHT, screen_w, UI_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, UI_BG_COLOR, panel_rect)
        pygame.draw.rect(self.screen, UI_BORDER_COLOR, panel_rect, 2)

        y_offset = screen_h - UI_PANEL_HEIGHT + 10
        
        tool_text = f"Tool: {TOOL_NAMES[self.current_tool]} (Brush: {self.paint_brush_size}) [-/+]"
        if self.current_tool == TOOL_TERRAIN:
            tool_text += f" | Painting: {TERRAIN_NAMES[self.paint_terrain_type]} [0-6]"
        elif self.current_tool == TOOL_BUILD:
            tool_text += f" | Building: {BUILDING_NAMES.get(self.build_tool_type, 'N/A')} [1-5]"
        
        self.render_text(tool_text, 15, y_offset, self.font_small, UI_TEXT_COLOR)
        y_offset += UI_FONT_SIZE + 5

        controls_text = "Controls: [G]eo [H]uman [T]ree [R]ock [F]ire [E]rase [I]nspect [B]uild | [P]ause | [PgUp/Dn] Zoom"
        self.render_text(controls_text, 15, y_offset, self.font_small, UI_TEXT_COLOR)
        y_offset += UI_FONT_SIZE + 15

        stats_text = (f"Humans: {self.human_manager.num_active_humans} | FPS: {self.clock.get_fps():.1f}")
        self.render_text(stats_text, 15, y_offset, self.font_medium, UI_HIGHLIGHT_COLOR)

    def draw_info_panel(self, screen_w, screen_h):
        info_panel_rect = pygame.Rect(screen_w - UI_INFO_PANEL_WIDTH - 10, 10, UI_INFO_PANEL_WIDTH, screen_h - UI_PANEL_HEIGHT - 20)
        
        info_surface = pygame.Surface((info_panel_rect.width, info_panel_rect.height), pygame.SRCALPHA)
        info_surface.fill((*UI_BG_COLOR, 200))
        pygame.draw.rect(info_surface, (*UI_BORDER_COLOR, 220), info_surface.get_rect(), 2)

        y_offset = 10
        
        if self.selected_entity_info:
            self.render_text_on_surface("--- Selected Entity ---", info_surface, 15, y_offset, self.font_medium, UI_HIGHLIGHT_COLOR)
            y_offset += UI_FONT_SIZE + 8
            for key, value in self.selected_entity_info.items():
                self.render_text_on_surface(f"{key}: {value}", info_surface, 20, y_offset, self.font_small, UI_TEXT_COLOR)
                y_offset += UI_FONT_SIZE + 3
            y_offset += 10

        if self.selected_tile_info:
            self.render_text_on_surface("--- Hovered Tile ---", info_surface, 15, y_offset, self.font_medium, UI_HIGHLIGHT_COLOR)
            y_offset += UI_FONT_SIZE + 8
            for key, value in self.selected_tile_info.items():
                self.render_text_on_surface(f"{key}: {value}", info_surface, 20, y_offset, self.font_small, UI_TEXT_COLOR)
                y_offset += UI_FONT_SIZE + 3
        
        self.screen.blit(info_surface, info_panel_rect.topleft)

    def render_text(self, text, x, y, font, color, surface=None):
        if surface is None: surface = self.screen
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, (x, y))

    def render_text_on_surface(self, text, target_surface, x, y, font, color):
        text_surf = font.render(text, True, color)
        target_surface.blit(text_surf, (x,y))

if __name__ == "__main__":
    game = Game()
    try:
        game.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
