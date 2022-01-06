# Sources
# https://www.gkstill.com/Support/crowd-density/CrowdDensity-1.html - flow rate vs crowd density

class Node:
    name = ""
    connections = []

    def __init__(self, name):
        self.name = name
    
    def add_connection(self, connection):
        self.connections.append(connection)

class SeatingNode(Node):
    type = "seating"
    population = 0
    
    def __init__(self, name, population):
        self.population = population
        super().__init__(name)
    
    def recieve_people(self, n):
        self.population += n

class ExitNode(Node):
    type = "exit"
    
    def __init__(self, name):
        super().__init__(name)
    
    def recieve_people(self, n):
        return # people go bye bye

class Connection:
    point1: Node
    point2: Node
    width = 0
    length = 0
    density = 0

    def __init__(self, start, end, width, length):
        self.point1 = start
        self.point2 = end
        self.width = width
        self.length = length
        self.point1.add_connection(self)
        self.point2.add_connection(self)
    
    def get_next(self, point):
        if point == self.point1:
            return self.point2
        else:
            return self.point1
    
    def get_throughput(self):
        if self.density > 2.58:
            a = 0.26 * self.density + 0.286
            return (-2 * ( (a**5+1) / a**3 ) + 84) * self.width / self.length
        else:
            return 80.124 * self.width / self.length # unimpeded maximum throughput
    
    def transmit_people(self, n, origin):
        self.get_next(origin).recieve_people(n)
        origin.recieve_people(-n)
        self.density = n / self.width
    
    def get_max_density(self):
        limited = False
        if limited:
            return 2.5
        else:
            return 6.5

    def simulate(self):
        if type(self.point1) == SeatingNode:
            self.transmit_people(min(self.get_throughput(), self.point1.population), self.point1)
            if self.point1.population > 0:
                self.density = self.get_max_density()
        else:
            self.transmit_people(min(self.get_throughput(), self.point2.population), self.point2)
            if self.point2.population > 0:
                self.density = self.get_max_density()

locations = [SeatingNode("topSeating", 5000), SeatingNode("mainSeating", 15000), SeatingNode("stageSeating", 5000), ExitNode("exitNorth"), ExitNode("exitSouth"), ExitNode("exitWest"), ExitNode("exitEast")]
connections = [
    Connection(locations[0], locations[3], 20, 40),
    Connection(locations[3], locations[1], 20, 20),
    Connection(locations[1], locations[5], 20, 100),
    Connection(locations[1], locations[6], 20, 100),
    Connection(locations[2], locations[5], 10, 60),
    Connection(locations[2], locations[6], 10, 60),
    Connection(locations[2], locations[4], 10, 90),
]

time = 0
done = False
while not done:
    for connection in connections:
        connection.simulate()
        time += 1
    done = True
    for node in locations:
        if type(node) == SeatingNode:
            # print(node.population)
            if node.population > 0:
                done = False
    # print()
print(time)