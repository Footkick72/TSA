# https://arxiv.org/pdf/1708.03080.pdf
from math import *
from typing import List, Sequence
import matplotlib.pyplot as plt
from random import *
import numpy as np
from PIL import Image
import heapq
import cv2
import os
import json
import cProfile

AGENT_RADIUS = 0.2  # size of agent (m)
AGENT_MAX_ANGLE_CHANGE = pi/2  # maximum agent direction change
AGENT_DISTANCE_WEIGHT = 0.4  # lower = more likely to slow down
AGENT_COMFORTABLE_SPACING = 1 # additional distance the agent likes to keep to others (m)
AGENT_SPACING_SHIFT = 0.8 # agent shift away from others (per step) (m) (NOT as a function of step size)
AGENT_STEP = 0.8  # step size of the agent (m)
AGENT_SEATING_PENALTY = 5 # scalar for the agent's dislike for seats (1 implies no avoidance of seats)
AGENT_PATH_LOOKAHEAD = 2 # number of steps ahead of current path location the agent tries to move to
AGENT_STUCK_THRESHOLD = 6 # how quickly the agent decides it is stuck and changes to a more direct pathing
AGENT_CROWD_THRESHOLD = 30 # how many people have to be present for the agent to decide "this is a crowd" and decrease movement checks
AGENT_THICK_CROWD_THRESHOLD = 52 # how many people have to be present for the agent to ignore shift calculations
EXIT_WIDTH = 2 # how close the people need to be to the exit to count as finished (path steps)
WORLD_FILE = "Stadium-map.png"
WORLD_EXTENTS = Image.open(WORLD_FILE).size

COLLISION_DIST_SQUARED = (2 * AGENT_RADIUS) * (2 * AGENT_RADIUS)
AVOIDANCE_DIST_SQUARED = (2 * AGENT_RADIUS + AGENT_COMFORTABLE_SPACING) * (2 * AGENT_RADIUS + AGENT_COMFORTABLE_SPACING)

values = {}
arcs = []
for theta in range(0, 5):
    arcs.append((0.5, (theta-2) / 2 * AGENT_MAX_ANGLE_CHANGE))
    for a in range(1, 5):
        direction = (theta-2) / 2 * AGENT_MAX_ANGLE_CHANGE
        distance = a / 4
        value = AGENT_DISTANCE_WEIGHT * a + (1 - AGENT_DISTANCE_WEIGHT) * (1 - abs(direction) / AGENT_MAX_ANGLE_CHANGE)
        values[(distance, direction)] = value

moves = [x for x in values]
moves.sort(key=lambda x: values[x], reverse=True)

class Agent:
    x_pos = 0
    y_pos = 0
    desired_step_size = 1.0
    desired_location = (0, 0)
    collision_grid_pos = (0, 0)
    desired_direction = 0
    path = None
    finished = False
    stuck_factor = 0

    def __init__(self, pos, speed, goal):
        self.x_pos, self.y_pos = pos
        self.desired_step_size = speed
        self.desired_location = goal
        self.compute_path()

    def compute_path(self):
        global world_collisions
        path = world_collisions.get_path((floor(self.x_pos), floor(self.y_pos)), (floor(self.desired_location[0]), floor(self.desired_location[1])))
        if path:
            if path == self.path:
                self.stuck_factor += 1
            else:
                self.path = path
                self.stuck_factor = 0

    def update(self):
        assert not self.finished

        global agent_collisions
        global world_collisions
        
        self.compute_path()

        if len(self.path) <= EXIT_WIDTH:
            agent_collisions.deregister_member(self)
            self.finished = True
            return
        
        lookahead = min(len(self.path)-1,AGENT_PATH_LOOKAHEAD)
        if self.stuck_factor >= AGENT_STUCK_THRESHOLD:
            lookahead = 1
        self.desired_direction = atan2(self.path[lookahead][1]+0.5 - self.y_pos, self.path[lookahead][0]+0.5 - self.x_pos)

        nearby_people = agent_collisions.get_all_collision_considerations(self.collision_grid_pos)
        
        if len(nearby_people) > AGENT_CROWD_THRESHOLD and self.stuck_factor >= 10*AGENT_STUCK_THRESHOLD:
            self.desired_location = world_collisions.random_goal()
            self.stuck_factor = 0
            self.update()
        
        shift = self.shift_avoid_people(nearby_people)
        original_length = self.desired_step_size
        original_direction = self.desired_direction
        self.desired_step_size = shift[0]
        self.desired_direction = shift[1]

        move = self.find_next_move(nearby_people)
        self.x_pos, self.y_pos = self.next_position(move)
        agent_collisions.check_for_updates(self)

        self.desired_step_size = original_length
        self.desired_direction = original_direction

    def shift_avoid_people(self, people):
        # if len(people) > AGENT_THICK_CROWD_THRESHOLD:
        #     return [self.desired_step_size, self.desired_direction]

        global agent_collisions
        people_pos_sum = [0, 0]
        n_people = 0
        for obstacle in people:
            if self != obstacle:
                dx = (self.x_pos - obstacle.x_pos)
                dy = (self.y_pos - obstacle.y_pos)
                if dx*dx + dy*dy < AVOIDANCE_DIST_SQUARED:
                    people_pos_sum[0] += obstacle.x_pos
                    people_pos_sum[1] += obstacle.y_pos
                    n_people += 1
        if n_people > 0:
            people_center = [people_pos_sum[0] / n_people, people_pos_sum[1] / n_people]
            avoidance_vector = [self.x_pos - people_center[0], self.y_pos - people_center[1]]
            avoidance_dir = atan2(avoidance_vector[1], avoidance_vector[0])
            shift = [
                cos(avoidance_dir) * AGENT_SPACING_SHIFT,
                sin(avoidance_dir) * AGENT_SPACING_SHIFT
                ]
            optimal_vector = [
                cos(self.desired_direction) * self.desired_step_size,
                sin(self.desired_direction) * self.desired_step_size
                ]
            shifted_vector = [
                optimal_vector[0] + shift[0],
                optimal_vector[1] + shift[1]
                ]
            shifted_angle = atan2(shifted_vector[1], shifted_vector[0])
            shifted_length = sqrt(shifted_vector[0]**2 + shifted_vector[1]**2)
            shifted_length = max(min(shifted_length, self.desired_step_size), 0)
            return [self.desired_step_size, shifted_angle]
        return [self.desired_step_size, self.desired_direction]

    def next_position(self, move):
        # according to the paper, sin and cos are flipped here, but that doesn't make sense, soooooooo...
        x = move[0] * self.desired_step_size * cos(self.desired_direction + move[1])
        y = move[0] * self.desired_step_size * sin(self.desired_direction + move[1])
        return (self.x_pos + x, self.y_pos + y)

    def find_next_move(self, people):
        global moves
        global arcs
        illegal_directions = set()
        if len(people) >= AGENT_CROWD_THRESHOLD:
            for arc in arcs:
                n = self.next_position(arc)
                if not self.is_legal_move(n, people):
                    illegal_directions.add(arc[1])
    
        best_selection = (0.0, 0.0)
        for move in moves:
            if move[1] in illegal_directions:
                continue
            n = self.next_position(move)
            if self.is_legal_move(n, people):
                best_selection = move
                break
        return best_selection

    def is_legal_move(self, n, people):
        global agent_collisions
        global world_collisions
        if not world_collisions.is_valid_location(n):
            return False
        for obstacle in people:
            if self != obstacle:
                dx = (n[0] - obstacle.x_pos)
                dy = (n[1] - obstacle.y_pos)
                if dx*dx + dy*dy < COLLISION_DIST_SQUARED:
                    return False
        return True

class AgentCollisionManager:
    square_side = 1
    squares = []

    def __init__(self, x, y):
        self.squares = [[[] for _ in range(ceil(y/self.square_side)+1)] for _ in range(ceil(x/self.square_side)+1)]
        # +1 because buffer (avoids having to out-of-bounds check, saving time in get_all_collision_considerations)

    def check_for_updates(self, member: Agent):
        coord = [floor(member.x_pos/self.square_side),floor(member.y_pos/self.square_side)]
        if coord[0] != member.collision_grid_pos[0] and coord[1] != member.collision_grid_pos[1]:
            self.squares[member.collision_grid_pos[0]][member.collision_grid_pos[1]].remove(member)
            self.squares[coord[0]][coord[1]].append(member)
            member.collision_grid_pos = coord

    def register_member(self, member: Agent):
        coord = [floor(member.x_pos/self.square_side),floor(member.y_pos/self.square_side)]
        self.squares[coord[0]][coord[1]].append(member)
        member.collision_grid_pos = coord
    
    def deregister_member(self, member: Agent):
        self.squares[member.collision_grid_pos[0]][member.collision_grid_pos[1]].remove(member)

    def point_in_square(self, pos, index):
        bounds = self.get_square_bounds(index)
        return pos[0] >= bounds[0] and pos[1] >= bounds[1] and pos[0] < bounds[2] and pos[1] < bounds[3]

    def get_square_bounds(self, index):
        min_x = index[0] * self.square_side
        min_y = index[1] * self.square_side
        max_x = (index[0] + 1) * self.square_side
        max_y = (index[1] + 1) * self.square_side
        return [min_x, min_y, max_x, max_y]

    def get_all_collision_considerations(self, index):
        objects = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                objects += self.squares[index[0]+dx][index[1]+dy]
        return objects

class WorldManager:
    illegal_pixels = set()
    seating_pixels = set()
    goals = []
    paths = {}

    def __init__(self, image):
        arr = np.asarray(image)
        for x in range(len(arr[0])):
            for y in range(len(arr)):
                px = arr[(WORLD_EXTENTS[1]-1)-y, x]
                if px[0] == 0 and px[1] == 0 and px[2] == 0:
                    self.illegal_pixels.add((float(x), float(y)))
                if px[0] == 255 and px[1] == 0 and px[2] == 0:
                    self.seating_pixels.add((float(x), float(y)))
                if px[0] == 0 and px[1] == 255 and px[2] == 0:
                    self.goals.append((x, y))
    
    def read_paths(self):
        if os.path.exists(WORLD_FILE[:-4] + "-pathdata.json"):
            f = open(WORLD_FILE[:-4] + "-pathdata.json", "r")
            self.paths = json.loads(f.read())
            f.close()
        else:
            self.precompute_paths()
    
    def precompute_paths(self):
        for goal in self.goals:
            print("computing paths to:", goal)
            self.compute_paths(goal)
        print("done computing paths")
        with open(WORLD_FILE[:-4] + "-pathdata.json", "w") as f:
            f.write(json.dumps(self.paths))
    
    def is_valid_location(self, point):
        x = floor(point[0])
        y = floor(point[1])
        return (x, y) not in self.illegal_pixels
    
    def is_seating_location(self, point):
        x = floor(point[0])
        y = floor(point[1])
        return (x, y) in self.seating_pixels

    def closest_goal(self, pos):
        best = 0
        distance = 1e100
        for i,g in enumerate(self.goals):
            if (g[0]-pos[0])**2 + (g[1]-pos[1])**2 < distance:
                distance = (g[0]-pos[0])**2 + (g[1]-pos[1])**2
                best = i
        return self.goals[best]
    
    def random_goal(self):
        return self.goals[randint(0,len(self.goals)-1)]
    
    def assign_goal(self, mode, pos):
        if mode == "random":
            return self.random_goal()
        elif mode == "closest":
            return self.closest_goal(pos)
    
    def get_path(self, pos, goal):
        if str((pos[0],pos[1],goal[0],goal[1])) in self.paths:
            path = [pos]
            for _ in range(AGENT_PATH_LOOKAHEAD):
                p = self.paths[str((path[-1][0],path[-1][1],goal[0],goal[1]))]
                if p == None:
                    break
                path.append(p)
            return path

    def compute_paths(self, goal):
        # Djikstra's algorithm
        # Computes every path to this goal
        frontier = [DijkstraNode(goal, None)]
        heapq.heapify(frontier)
        frontier_set = set(frontier)
        visited = set()

        while len(frontier) != 0:
            next_node = heapq.heappop(frontier)
            visited.add(next_node)
            frontier_set.remove(next_node)
            for neighbor in self.neighbors(next_node.pos):
                if not self.is_valid_location(neighbor):
                    continue
                n = DijkstraNode(neighbor, next_node)
                if n in visited:
                    continue
                if n not in frontier_set:
                    frontier_set.add(n)
                    heapq.heappush(frontier, n)
                else:
                    for e in frontier_set:
                        if e == n:
                            if e > n:
                                e.set_parent(next_node)
                            break
        
        for node in visited:
            if node.parent != None:
                self.paths[str((node.pos[0], node.pos[1], goal[0], goal[1]))] = node.parent.pos
            else:
                self.paths[str((node.pos[0], node.pos[1], goal[0], goal[1]))] = None

    def neighbors(self, pos):
        return [
            (pos[0] - 1, pos[1] - 1),
            (pos[0], pos[1] - 1),
            (pos[0] + 1, pos[1] - 1),
            (pos[0] - 1, pos[1]),
            (pos[0] + 1, pos[1]),
            (pos[0] - 1, pos[1] + 1),
            (pos[0], pos[1] + 1),
            (pos[0] + 1, pos[1] + 1)
        ]

class DijkstraNode:
    pos = (0, 0)
    parent = None
    cost = 0

    def __init__(self, pos, parent):
        self.pos = pos
        self.parent = parent
        if self.parent == None:
            self.cost = 0
        else:
            self.set_cost()

    def set_parent(self, new_parent):
        self.parent = new_parent
        self.set_cost()

    def set_cost(self):
        step_cost = sqrt((self.pos[0]-self.parent.pos[0])**2 + (self.pos[1]-self.parent.pos[1])**2)
        if world_collisions.is_seating_location(self.pos):
            step_cost *= AGENT_SEATING_PENALTY
        self.cost = self.parent.cost + step_cost

    def __eq__(self, other) -> bool:
        if type(other) == DijkstraNode:
            return self.pos == other.pos
        return False

    def __ne__(self, other) -> bool:
        if type(other) == DijkstraNode:
            return self.pos != other.pos
        return True

    def __lt__(self, other) -> bool:
        return self.cost < other.cost

    def __gt__(self, other) -> bool:
        return self.cost > other.cost

    def __le__(self, other) -> bool:
        return self.cost <= other.cost

    def __ge__(self, other) -> bool:
        return self.cost >= other.cost

    def __hash__(self):
        return hash(self.pos)

world_collisions = WorldManager(Image.open(WORLD_FILE))
world_collisions.read_paths()
agent_collisions = AgentCollisionManager(WORLD_EXTENTS[0], WORLD_EXTENTS[1])

def run_sim(percent_filled, time, draw_interval = 1, pathing = "random"):
    agents = []

    for gridpos in world_collisions.seating_pixels:
        for pos in [[gridpos[0]-0.25, gridpos[1]-0.25],[gridpos[0]-0.25, gridpos[1]+0.25],[gridpos[0]+0.25, gridpos[1]-0.25],[gridpos[0]+0.25, gridpos[1]+0.25]]:
            if random() < percent_filled:
                a = Agent((pos[0] + 0.5, pos[1] + 0.5), AGENT_STEP, world_collisions.assign_goal(pathing,pos))
                agents.append(a)
                agent_collisions.register_member(a)
    print(f"simulating {len(agents)} agents")

    agent_positions = [[] for _ in range(len(agents))]

    plt.imshow(plt.imread(WORLD_FILE), extent=[0, WORLD_EXTENTS[0], 0, WORLD_EXTENTS[1]])
    plt.savefig("temp.png")
    plt.clf()
    frame = cv2.imread("temp.png")
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("movement.mp4", fourcc, 30, (width, height))

    for t in range(time):
        print(str((t+1)/time * 100) + "%")
        done = True
        for i, a in enumerate(agents):
            if not a.finished:
                done = False
                a.update()
                if a.finished:
                    agent_positions[i].append([0,0])
                    continue
                if t%draw_interval == 0:
                    agent_positions[i].append((a.x_pos, a.y_pos))
        
        if done:
            break
        
        if t%draw_interval == 0:
            plt.scatter(list(map(lambda x: x[-1][0], agent_positions)), list(map(lambda x: x[-1][1], agent_positions)), s=4)
            plt.imshow(plt.imread(WORLD_FILE), extent=[0, WORLD_EXTENTS[0], 0, WORLD_EXTENTS[1]])
            # plt.show()
            plt.savefig("temp.png")
            plt.clf()
            video.write(cv2.imread("temp.png"))

    cv2.destroyAllWindows()
    video.release()
    
    for pos_list in agent_positions:
        plt.plot(list(map(lambda x: x[0], pos_list[:-1])), list(map(lambda x: x[1], pos_list[:-1])), 'o-', linewidth=1, markersize=3)
    plt.imshow(plt.imread(WORLD_FILE), extent=[0, WORLD_EXTENTS[0], 0, WORLD_EXTENTS[1]])
    plt.show()

    return (t+1) * AGENT_STEP/1.4

if __name__ == "__main__":
    max_time = 30*60*4 # ~half an hour

    # cProfile.run("run_sim(1.0, max_time, draw_interval = 20)","profilestats")
    with open("results.txt","w") as f:
        random_100 = run_sim(1.0, max_time, draw_interval = 10, pathing = 'closest')
        print(f"the agents exited the stadium in {random_100} seconds")
        f.write(str(random_100) + "\n")