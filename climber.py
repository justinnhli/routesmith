#!/usr/bin/env python3

import math
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from collections import Counter, OrderedDict
from copy import copy
from itertools import product
from numbers import Real
from os.path import dirname, expanduser, join as join_path, realpath

from tkinter import Canvas, Tk, mainloop

# CONSTANTS

TOLERANCE = 0.0005

# CLIMBER CLASSES

class Pose:
    def __init__(self, left_hand=None, right_hand=None, left_foot=None, right_foot=None):
        self.limbs = {}
        self.limbs["left_hand"] = left_hand
        self.limbs["right_hand"] = right_hand
        self.limbs["left_foot"] = left_foot
        self.limbs["right_foot"] = right_foot
    @property
    def left_hand(self):
        return self.limbs["left_hand"]
    @property
    def right_hand(self):
        return self.limbs["right_hand"]
    @property
    def left_foot(self):
        return self.limbs["left_foot"]
    @property
    def right_foot(self):
        return self.limbs["right_foot"]
    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()
    def __hash__(self):
        return hash(self.as_tuple())
    def __iter__(self):
        return iter(self.limbs.items())
    def __str__(self):
        return "Pose(" + ", ".join(str(p) for p in self.as_tuple()) + ")"
    def as_tuple(self):
        return (self.left_hand, self.right_hand, self.left_foot, self.right_foot)
    def move(self, limb, hold):
        return Move(Pose(*self.as_tuple()), Pose(*(hold if limb == k else v for k, v in self.limbs.items())))

class Move:
    def __init__(self, before, after):
        self.before = before
        self.after = after
    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()
    def __hash__(self):
        return hash(self.as_tuple())
    def as_tuple(self):
        return (self.prev_pose, self.next_pose)

class Climber:
    proportions = {
        "fingers": 0.0462,
        "palm": 0.0578,
        "forearm": 0.127,
        "upperarm": 0.173,
        "shoulders": 0.191,
        "toe-ankle": 0.773,
        "ankle-heel": 0.227,
        "ankle-ground": 0.0347,
        "knee-ankle": 0.220,
        "hip-knee": 0.220,
        "waist-hip": 0.139,
        "shoulders-waist": 0.220,
        "neck-shoulders": 0.0289,
        "head-neck": 0.139,
    }
    def __init__(self, height=175, ape_index=1):
        self.height = height
        self.armspan = height + 2.5 * ape_index
    def valid_pose(self, pose):
        return True
    def valid_move(self, move):
        return True
    def evaluate_pose(self, pose):
        return 1
    def evaluate_move(self, move):
        return 1

# THE OUVRIR

class SearchNode():
    def __init__(self, path, actions, cost, heuristic):
        self.path = path
        self.actions = actions
        self.cost = cost
        self.heuristic = heuristic
    @property
    def state(self):
        return self.path[-1]
    @property
    def depth(self):
        return len(self.path)
    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()
    def __hash__(self):
        return hash(self.as_tuple())
    def as_tuple(self):
        return (self.path, self.actions, self.cost, self.heuristic)

class Ouvrir:
    def __init__(self, climber, problem):
        self.climber = climber
        self.problem = problem
        self.hold_map = {}
        for index, hold in enumerate(self.problem.holds):
            self.hold_map[hold.real_coords] = index
        self.distances = {}
        for i in range(len(self.problem.holds)):
            for j in range(i+1, len(self.problem.holds)):
                distance = self.problem.wall.surface_distance(self.problem.holds[i].wall_position, self.problem.holds[j].wall_position)
                self.distances[(i, j)] = distance
                self.distances[(j, i)] = distance
    def ouvrir(self):
        for index, sequence in enumerate(self.search()):
            if index > 10:
                break
            print(", ".join("<" + ",".join(str(self.coords_to_hold(coords)) for coords in pose.as_tuple()) + ">" for pose in sequence))
    def start_poses(self):
        start_poses = []
        hand_holds = [None,] + [self.problem.holds[i] for i in self.problem.start_holds]
        for left_hand, right_hand in product(hand_holds, repeat=2):
            if left_hand is None and right_hand is None:
                continue
            foot_holds = [None,] + list(self.problem.holds)
            for left_foot, right_foot in product(foot_holds, repeat=2):
                pose = Pose(*(self.hold_to_coords(hold) for hold in (left_hand, right_hand, left_foot, right_foot)))
                if self.climber.valid_pose(pose):
                    start_poses.append(pose)
        return start_poses
    def next_moves(self, pose):
        moves = []
        # this will only occur on the first move
        if pose.left_hand is None or pose.right_hand is None:
            for index, hold in enumerate(self.problem.holds):
                if pose.left_hand is None:
                    moves.append(pose.move("left_hand", hold.real_coords))
                else:
                    moves.append(pose.move("right_hand", hold.real_coords))
        else:
            for index, hold in enumerate(self.problem.holds):
                for limb, point in pose:
                    moves.append(pose.move(limb, hold.real_coords))
        return [move for move in moves if self.climber.valid_move(move)]
    def how_much_further(self, pose):
        return 1
    def at_finish(self, pose):
        return (pose.left_hand is not None and pose.right_hand is not None and
                self.hold_map[pose.left_hand] in self.problem.finish_holds and self.hold_map[pose.right_hand] in self.problem.finish_holds)
    def search(self):
        queue = [SearchNode([pose,], [None,], self.climber.evaluate_pose(pose), self.how_much_further(pose)) for pose in self.start_poses()]
        visited = set()
        while queue:
            cur_node = queue.pop(0)
            while cur_node.state in visited:
                cur_node = queue.pop(0)
            visited.add(cur_node.state)
            if self.at_finish(cur_node.state):
                yield cur_node.path
            for move in self.next_moves(cur_node.state):
                node = SearchNode(
                        cur_node.path + [move.after,],
                        cur_node.actions + [move,],
                        cur_node.cost + self.climber.evaluate_pose(move.after) + self.climber.evaluate_move(move),
                        self.how_much_further(move.after))
                queue.append(node)
            queue = sorted(queue, key=(lambda node: node.cost + node.heuristic))
            queue = list(filter(None, queue))
    def hold_to_coords(self, hold):
        if hold is None:
            return None
        else:
            return hold.real_coords
    def coords_to_hold(self, coords):
        if coords is None:
            return None
        else:
            return self.hold_map[coords]
