#!/usr/bin/env python3

import math
from sys import argv

HEIGHT = 300
WIDTH = 240

def ssv(iterable):
    return " ".join(str(i) for i in iterable)

if __name__ == "__main__":
    if len(argv) != 2:
        print("usage: {} ANGLE".format(argv[0]))
        exit(1)
    angle = float(argv[1])
    sign = math.copysign(1, angle)
    angle = abs(angle)

    depth = sign * HEIGHT * math.tan(math.radians(angle))

    print(ssv((depth, 0, HEIGHT)))
    print(ssv((0, 0, 0)))
    print(ssv((0, WIDTH, 0)))
    print(ssv((depth, WIDTH, HEIGHT)))
    if sign < 0:
        print(ssv((depth, 0, 0)))
        print(ssv((depth, WIDTH, 0)))
    elif sign > 0:
        print(ssv((0, 0, HEIGHT)))
        print(ssv((0, WIDTH, HEIGHT)))
    print()
    print(ssv(range(4)))
    if sign != 0:
        print(ssv((0, 4, 1)))
        print(ssv((2, 3, 5)))
