# https://arxiv.org/pdf/1708.03080.pdf
from math import *
from typing import List, Sequence
import matplotlib.pyplot as plt
from random import *
import numpy as np
from PIL import Image
import heapq
import cv2

AGENT_RADIUS = 0.2  # size of agent (m)
AGENT_MAX_ANGLE_CHANGE = pi/2  # maximum agent direction change
AGENT_DISTANCE_WEIGHT = 0.6  # lower = more likely to slow down
AGENT_COMFORTABLE_SPACING = 1.0 # additional distance the agent likes to keep to others (m)
AGENT_SPACING_SHIFT = 0.1 # agent shift away from others (per step) (per person) (m)
# AGENT_SPACING_WEIGHT = 0.2 # scalar for agent's direction shift to avoid people
AGENT_STEP = 0.4  # step size of the agent (m)
AGENT_SEATING_PENALTY = 5 # scalar for the agent's dislike for seats (1 implies no avoidance of seats)
WORLD_FILE = "Stadium-map.png"
WORLD_EXTENTS = Image.open(WORLD_FILE).size

values = {}
for a in range(0, 11):
    for theta in range(-5, 6):
        direction = theta / 5 * AGENT_MAX_ANGLE_CHANGE
        distance = a / 10
        value = AGENT_DISTANCE_WEIGHT * a + \
            (1 - AGENT_DISTANCE_WEIGHT) * \
            (1 - abs(direction)/AGENT_MAX_ANGLE_CHANGE)
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

    def __init__(self, pos, speed, goal):
        self.x_pos, self.y_pos = pos
        self.desired_step_size = speed
        self.desired_location = goal
        self.compute_path()

    def compute_path(self):
        global world_collisions
        self.path = world_collisions.compute_path((round(self.x_pos), round(
            self.y_pos)), (round(self.desired_location[0]), round(self.desired_location[1])))

    def update(self):
        global agent_collisions
        if len(self.path) != 0:
            self.desired_direction = atan2(
                self.path[0][1] - self.y_pos, self.path[0][0] - self.x_pos)

            shift = self.shift_avoid_people()
            original_length = self.desired_step_size
            original_direction = self.desired_direction
            self.desired_step_size = shift[0]
            self.desired_direction = shift[1]

            move = self.find_next_move()
            self.x_pos, self.y_pos = self.next_position(move)
            agent_collisions.check_for_updates(self)

            self.desired_step_size = original_length
            self.desired_direction = original_direction

            if sqrt((self.x_pos - self.path[0][0])**2 + (self.y_pos - self.path[0][1])**2) < 2:
                self.path.pop(0)

    def shift_avoid_people(self):
        global agent_collisions
        people_pos_sum = [0, 0]
        n_people = 0
        for obstacle in agent_collisions.get_all_collision_considerations(self.collision_grid_pos):
            if self != obstacle:
                if self.distance_to(obstacle) < 2 * AGENT_RADIUS + AGENT_COMFORTABLE_SPACING:
                    people_pos_sum[0] += obstacle.x_pos
                    people_pos_sum[1] += obstacle.y_pos
                    n_people += 1
        if n_people > 0:
            people_center = [people_pos_sum[0] / n_people, people_pos_sum[1] / n_people]
            avoidance_vector = [self.x_pos - people_center[0], self.y_pos - people_center[1]]
            avoidance_dir = atan2(avoidance_vector[1], avoidance_vector[0])
            shift = [
                cos(avoidance_dir) * AGENT_SPACING_SHIFT * n_people,
                sin(avoidance_dir) * AGENT_SPACING_SHIFT * n_people
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
            # shifted_angle = max(min(shifted_angle, AGENT_MAX_ANGLE_CHANGE), -AGENT_MAX_ANGLE_CHANGE)
            shifted_length = min(shifted_length, self.desired_step_size)
            return [shifted_length, shifted_angle]
        return [self.desired_step_size, self.desired_direction]

    def next_position(self, move):
        # according to the paper, sin and cos are flipped here, but that doesn't make sense, soooooooo...
        x = move[0] * self.desired_step_size * \
            cos(self.desired_direction + move[1])
        y = move[0] * self.desired_step_size * \
            sin(self.desired_direction + move[1])
        return (self.x_pos + x, self.y_pos + y)

    def find_next_move(self):
        # max_v = 0
        # best_selection = (0.0, 0.0)
        # for a in range(0, 11):
        #     for theta in range(-5, 6):
        #         direction = theta / 5 * AGENT_MAX_ANGLE_CHANGE
        #         distance = a / 10
        #         value = self.get_move_cost([distance, direction])
        #         if value > max_v:
        #             best_selection = [distance, direction]
        #             max_v = value
        global moves
        best_selection = (0.0, 0.0)
        for move in moves:
            if self.is_legal_move(move):
                best_selection = move
                break
        return best_selection

    def is_legal_move(self, move):
        global agent_collisions
        global world_collisions
        for obstacle in agent_collisions.get_all_collision_considerations(self.collision_grid_pos):
            if self != obstacle:
                if self.hypothetical_distance_to(move, obstacle) < 2 * AGENT_RADIUS:
                    return False
        if not world_collisions.is_valid_location(self.next_position(move)):
            return False
        return True

    def get_move_cost(self, move):
        global agent_collisions
        global world_collisions
        people_factor = 0
        for obstacle in agent_collisions.get_all_collision_considerations(self.collision_grid_pos):
            if self != obstacle:
                if self.hypothetical_distance_to(move, obstacle) < 2 * AGENT_RADIUS:
                    return 0
                if self.hypothetical_distance_to(move, obstacle) < 2 * AGENT_RADIUS + AGENT_COMFORTABLE_SPACING:
                    people_factor -= 1 / \
                        self.hypothetical_distance_to(move, obstacle)
                if self.distance_to(obstacle) < 2 * AGENT_RADIUS + AGENT_COMFORTABLE_SPACING:
                    people_factor += 1/self.distance_to(obstacle)
        if not world_collisions.is_valid_location(self.next_position(move)):
            return 0
        return AGENT_DISTANCE_WEIGHT * move[0] + (1 - AGENT_DISTANCE_WEIGHT) * (1 - abs(move[1])/AGENT_MAX_ANGLE_CHANGE) + people_factor * AGENT_SPACING_WEIGHT

    def distance_to(self, other):
        return sqrt((self.x_pos - other.x_pos)**2 + (self.y_pos - other.y_pos)**2)

    def hypothetical_distance_to(self, move, other):
        n = self.next_position(move)
        return sqrt((n[0] - other.x_pos)**2 + (n[1] - other.y_pos)**2)


class AgentCollisionManager:
    square_side = 2
    squares = []

    def __init__(self, x, y):
        self.squares = [[[] for _ in range(
            int(y/self.square_side))] for _ in range(int(x/self.square_side))]

    def check_for_updates(self, member: Agent):
        square = self.squares[member.collision_grid_pos[0]
                              ][member.collision_grid_pos[1]]
        if not self.point_in_square([member.x_pos, member.y_pos], member.collision_grid_pos):
            square.remove(member)

            found = False
            for coord in self.get_neighbors(member.collision_grid_pos):
                if self.point_in_square([member.x_pos, member.y_pos], coord):
                    self.squares[coord[0]][coord[1]].append(member)
                    member.collision_grid_pos = coord
                    found = True
                    break
            if not found:
                self.register_member(member)  # blanket search

    def register_member(self, member: Agent):
        for x in range(len(self.squares)):
            for y in range(len(self.squares[0])):
                coord = [x, y]
                if self.point_in_square([member.x_pos, member.y_pos], coord):
                    self.squares[coord[0]][coord[1]].append(member)
                    member.collision_grid_pos = coord

    def point_in_square(self, pos, index):
        bounds = self.get_square_bounds(index)
        return pos[0] >= bounds[0] and pos[1] >= bounds[1] and pos[0] < bounds[2] and pos[1] < bounds[3]

    def get_square_bounds(self, index):
        min_x = index[0] * self.square_side
        min_y = index[1] * self.square_side
        max_x = (index[0] + 1) * self.square_side
        max_y = (index[1] + 1) * self.square_side
        return [min_x, min_y, max_x, max_y]

    def get_neighbors(self, index):
        possible = [
            [index[0] - 1, index[1] - 1],
            [index[0], index[1] - 1],
            [index[0] + 1, index[1] - 1],
            [index[0] - 1, index[1]],
            [index[0] + 1, index[1]],
            [index[0] - 1, index[1] + 1],
            [index[0], index[1] + 1],
            [index[0] + 1, index[1] + 1]
        ]
        valid = []
        for pos in possible:
            if not (pos[0] < 0 or pos[0] > len(self.squares) - 1 or pos[1] < 0 or pos[1] > len(self.squares[0]) - 1):
                valid.append(pos)
        return valid

    def get_all_collision_considerations(self, index):
        indices = []
        indices.append(index)
        indices += self.get_neighbors(index)
        objects = []
        for i in indices:
            objects += self.squares[i[0]][i[1]]
        return objects


class WorldManager:
    illegal_pixels = set()
    seating_pixels = set()

    def __init__(self, image):
        arr = np.asarray(image)
        for x in range(len(arr[0])):
            for y in range(len(arr)):
                px = arr[(WORLD_EXTENTS[1]-1)-y, x]
                if px[0] == 0 and px[1] == 0 and px[2] == 0:
                    self.illegal_pixels.add((float(x), float(y)))
                if px[0] == 255 and px[1] == 0 and px[2] == 0:
                    self.seating_pixels.add((float(x), float(y)))

    def is_valid_location(self, point):
        x = floor(point[0])
        y = floor(point[1])
        return (x, y) not in self.illegal_pixels
    
    def is_seating_location(self, point):
        x = floor(point[0])
        y = floor(point[1])
        return (x, y) in self.seating_pixels

    def compute_path(self, pos, goal):
        # A* algorithm
        frontier = [AstarNode(pos, None, goal)]
        heapq.heapify(frontier)
        frontier_set = set(frontier)
        visited = set()

        done = False
        while len(frontier) != 0 and not done:
            next_node = heapq.heappop(frontier)
            visited.add(next_node)
            frontier_set.remove(next_node)
            for neighbor in self.neighbors(next_node.pos):
                if not self.is_valid_location(neighbor):
                    continue
                n = AstarNode(neighbor, next_node, goal)
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
            if AstarNode(goal, None, goal) in visited:
                done = True

        if done:
            node_path = []
            for e in visited:
                if e == AstarNode(goal, None, goal):
                    node_path.append(e)
                    break
            while node_path[-1].parent != None:
                node_path.append(node_path[-1].parent)
            node_path.reverse()
            path = []
            for node in node_path:
                path.append(node.pos)
            return path
        else:
            return None

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

    def cost(self, pos, goal):
        # A* heuristic cost of a location relative to goal
        return sqrt((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2)


class AstarNode:
    pos = (0, 0)
    parent = None
    static_cost = 0
    heuristic_cost = 0
    cost = 0

    def __init__(self, pos, parent, goal):
        self.pos = pos
        self.parent = parent
        if self.parent == None:
            self.static_cost = 0
        else:
            step_cost = sqrt((self.pos[0]-self.parent.pos[0])**2 + (self.pos[1]-self.parent.pos[1])**2)
            if world_collisions.is_seating_location(self.pos):
                step_cost *= AGENT_SEATING_PENALTY
            self.static_cost = self.parent.static_cost + step_cost
        self.set_heuristic_cost(goal)
        self.set_cost()

    def set_parent(self, new_parent):
        self.parent = new_parent
        step_cost = sqrt((self.pos[0]-self.parent.pos[0])**2 + (self.pos[1]-self.parent.pos[1])**2)
        if world_collisions.is_seating_location(self.pos):
            step_cost *= AGENT_SEATING_PENALTY
        self.static_cost = self.parent.static_cost + step_cost
        self.set_cost()

    def set_heuristic_cost(self, goal):
        self.heuristic_cost = sqrt((self.pos[0]-goal[0])**2 + (self.pos[1]-goal[1])**2)

    def set_cost(self):
        self.cost = self.static_cost + self.heuristic_cost

    def __eq__(self, other) -> bool:
        if type(other) == AstarNode:
            return self.pos == other.pos
        return False

    def __ne__(self, other) -> bool:
        if type(other) == AstarNode:
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
agent_collisions = AgentCollisionManager(WORLD_EXTENTS[0], WORLD_EXTENTS[1])

goal = [5, 110]

percent_filled = 0.1 # 0 - 1
agents = []
arr = np.asarray(Image.open(WORLD_FILE))
for pos in world_collisions.seating_pixels:
    if random() < percent_filled:
        if len(agents) % 100 == 0:
            print(len(agents))
        a = Agent((pos[0] + 0.5, pos[1] + 0.5), AGENT_STEP, goal)
        agents.append(a)
        agent_collisions.register_member(a)
print(len(agents))

agent_positions = [[] for _ in range(len(agents))]

plt.imshow(plt.imread(WORLD_FILE), extent=[0, WORLD_EXTENTS[0], 0, WORLD_EXTENTS[1]])
plt.savefig("temp.png")
plt.clf()
frame = cv2.imread("temp.png")
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("movement.mp4", fourcc, 5, (width, height))

time = 500
for t in range(time):
    print(str((t+1)/time * 100) + "%")
    for i, a in enumerate(agents):
        a.update()
        agent_positions[i].append((a.x_pos, a.y_pos))

    plt.scatter(list(map(lambda x: x[-1][0], agent_positions)),
                list(map(lambda x: x[-1][1], agent_positions)), s=4)
    plt.imshow(plt.imread(WORLD_FILE), extent=[
               0, WORLD_EXTENTS[0], 0, WORLD_EXTENTS[1]])
    plt.savefig("temp.png")
    plt.clf()
    video.write(cv2.imread("temp.png"))

cv2.destroyAllWindows()
video.release()

for pos_list in agent_positions:
    plt.plot(list(map(lambda x: x[0], pos_list)), list(
        map(lambda x: x[1], pos_list)), 'o-', linewidth=1, markersize=3)
plt.imshow(plt.imread(WORLD_FILE), extent=[
           0, WORLD_EXTENTS[0], 0, WORLD_EXTENTS[1]])
plt.show()
