from math import *

distance_weight = 0.2
max_angle_change = pi/2

values = {}
for a in range(0, 11):
    for theta in range(-10, 11):
        direction = theta / 10 * max_angle_change
        distance = a / 10
        value = distance_weight * a + (1 - distance_weight) * (1 - abs(direction)/max_angle_change)
        values[distance, direction] = value

print(values)