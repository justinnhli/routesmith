#!/usr/bin/env python3

import math

HEIGHT = 300
WIDTH = 210
ANGLE = 20

def ssv(iterable):
    return " ".join(str(i) for i in iterable)

depth = -HEIGHT * math.tan(math.radians(ANGLE))

print(ssv((depth, 0, HEIGHT)))
print(ssv((0, 0, 0)))
print(ssv((0, WIDTH, 0)))
print(ssv((depth, WIDTH, HEIGHT)))
print(ssv((depth, 0, 0)))
print(ssv((depth, WIDTH, 0)))
print()
print(ssv(range(4)))
print(ssv((0, 4, 1)))
print(ssv((2, 3, 5)))
