#!/usr/bin/env python3

import math
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from collections import Counter, deque
from copy import copy
from itertools import product
from numbers import Real
from os.path import dirname, expanduser, join as join_path, realpath

from tkinter import Canvas, Tk, mainloop

# CONSTANTS

TOLERANCE = 0.0005

# ABSTRACT CLASSES

class Point:
    def __init__(self, *dimensions):
        self.values = tuple(0 if abs(i) < TOLERANCE else i for i in dimensions)
    @property
    def dimensions(self):
        return len(self.values)
    @property
    def x(self):
        return self.values[0]
    @property
    def y(self):
        return self.values[1]
    @property
    def z(self):
        return self.values[2]
    @property
    def length(self):
        return math.sqrt(sum(i * i for i in self.values))
    def __eq__(self, other):
        return all(x == y for x, y in zip(self.values, other.values))
    def __hash__(self):
        return hash(self.values)
    def __str__(self):
        return "(" + ", ".join(format(i, ".3f") for i in self.values) + ")"
    def __add__(self, p):
        assert isinstance(p, Point) and self.dimensions == p.dimensions
        return Point(*(i + j for i, j in zip(self.values, p.values)))
    def __sub__(self, p):
        assert isinstance(p, Point) and self.dimensions == p.dimensions
        return Point(*(i - j for i, j in zip(self.values, p.values)))
    def __neg__(self):
        return Point(*(-i for i in self.values))
    def __rmul__(self, r):
        assert isinstance(r, Real)
        return Point(*(r * i for i in self.values))
    def dot(self, p):
        assert isinstance(p, Point) and self.dimensions == p.dimensions
        return sum(i * j for i, j in zip(self.values, p.values))
    def cross(self, p):
        assert self.dimensions == 3 and isinstance(p, Point) and p.dimensions == 3
        return Point(self.y * p.z - self.z * p.y, self.z * p.x - self.x * p.z, self.x * p.y - self.y * p.x)
    def angle(self, p):
        return math.acos(self.dot(p) / (self.length * p.length))
    def project(self, p):
        assert isinstance(p, Point) and self.dimensions == p.dimensions
        return self - (self.dot(p) / p.length) * p
    def rotate(self, theta, phi):
        assert self.dimensions == 3
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        p = Point(self.x * cos_theta + self.y * -sin_theta, self.x * sin_theta + self.y * cos_theta, self.z)
        p = Point(p.x * cos_phi + p.z * sin_phi, p.y, p.x * -sin_phi + p.z * cos_phi)
        return p
    def normalize(self):
        l = self.length
        return Point(*(i / l for i in self.values))

class Drawable(metaclass=ABCMeta):
    @abstractmethod
    def draw_wireframe(self):
        raise NotImplementedError()
    @abstractmethod
    def draw(self):
        raise NotImplementedError()

# MODELING CLASSES

class Surface:
    def __init__(self, points=None):
        if points is None:
            self.points = deque()
        else:
            self.points = deque(points)
        # find the plane (ax + by + cz = constant)
        self.normal = Counter((self.points[i-1] - self.points[i-2]).cross(self.points[i] - self.points[i-1]).normalize() for i in range(1, len(self.points))).most_common(1)[0][0]
        self.constant = self.normal.dot(self.points[0])
        # make sure points are co-planar
        assert all(abs(p.dot(self.normal) - self.constant) < TOLERANCE for p in self.points), "Vertices are not planar"
        # use the first point as a temporary origin
        self.origin = self.points[0]
        # define the first basis, depending on whether the surface is horizontal
        if abs(self.normal.z) == 1:
            self.basis_x = Point(0, self.normal.z, -self.normal.x).normalize()
            if self.normal.cross(self.basis_x).x < 0:
                self.basis_x = -self.basis_x
        else:
            self.basis_x = Point(self.normal.y, -self.normal.x, 0).normalize()
            if self.normal.cross(self.basis_x).z < 0:
                self.basis_x = -self.basis_x
        self.basis_y = self.normal.cross(self.basis_x)
        # find the lowest transformed coordinates for each basis
        min_x = min((self.points[i] - self.origin).dot(self.basis_x) for i in range(len(self.points)))
        min_y = min((self.points[i] - self.origin).dot(self.basis_y) for i in range(len(self.points)))
        # move the origin to that point
        self.origin += min_x * self.basis_x + min_y * self.basis_y
        # TODO make sure surface is simple (no line intersections)
    def pos2coord(self, point):
        return self.origin + (point.x * self.basis_x + point.y * self.basis_y)
    def coord2pos(self, point):
        assert abs(point.dot(self.normal) - self.constant) < TOLERANCE
        offset = (point - self.origin)
        return Point(offset.dot(self.basis_x), offset.dot(self.basis_y))
    def project(self, vector):
        return vector - (vector - self.points[0]).dot(self.normal) * self.normal

# GRAPHICS CLASSES

class IsometricViewer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.wireframe = True
        self.drawables = []
        self.items = {}
        self.text = ""
        self.reset_viewport()
        self.init_gui()
    def reset_viewport(self):
        self.theta = math.pi / 8
        self.phi = math.pi / 16
        self.scale = 1
        self.x_offset = self.width / 2
        self.y_offset = self.height / 2
    def init_gui(self):
        self.canvas = Canvas(Tk(), width=self.width, height=self.height)
        self.canvas.pack()
        self.canvas.focus_set()
        self.canvas.bind("<Up>", self._callback_commandline_up)
        self.canvas.bind("<Down>", self._callback_commandline_down)
        self.canvas.bind("<Left>", self._callback_commandline_left)
        self.canvas.bind("<Right>", self._callback_commandline_right)
        self.canvas.bind("=", self._callback_commandline_equal)
        self.canvas.bind("-", self._callback_commandline_minus)
        self.canvas.bind("<Shift-Up>", self._callback_commandline_shift_up)
        self.canvas.bind("<Shift-Down>", self._callback_commandline_shift_down)
        self.canvas.bind("<Shift-Left>", self._callback_commandline_shift_left)
        self.canvas.bind("<Shift-Right>", self._callback_commandline_shift_right)
        self.canvas.bind("<Shift-Return>", self._callback_commandline_shift_return)
        self.canvas.bind("<Button-1>", self._callback_button_1)
        self.canvas.bind("<B1-Motion>", self._callback_button_1_motion)
    def add_drawable(self, drawable):
        assert isinstance(drawable, Drawable)
        self.drawables.append(drawable)
    def project(self, point):
        projected = point.rotate(self.theta, self.phi)
        return Point(projected.y * self.scale + self.x_offset, -projected.z * self.scale + self.y_offset)
    def unproject(self, point):
        return point.rotate(0, -self.phi).rotate(-self.theta, 0)
    def move_camera(self, theta, phi):
        self.theta += theta
        if self.theta > 2 * math.pi:
            self.theta -= 2 * math.pi
        elif self.theta < 0:
            self.theta += 2 * math.pi
        if -math.pi / 2 <= self.phi + phi <= math.pi / 2:
            self.phi += phi
    def camera_coords(self):
        return Point(math.cos(self.theta) * math.cos(self.phi), -math.sin(self.theta) * math.cos(self.phi), math.sin(self.phi))
    def clear(self):
        for item in self.items:
            self.canvas.delete(item)
    def draw_line(self, owner, p1, p2, **kargs):
        assert isinstance(p1, Point)
        assert isinstance(p2, Point)
        p1 = self.project(p1)
        p2 = self.project(p2)
        item = self.canvas.create_line(p1.x, p1.y, p2.x, p2.y, **kargs)
        self.items[item] = owner
        return item
    def draw_circle(self, owner, p, r, **kargs):
        assert isinstance(p, Point)
        p = self.project(p)
        item = self.canvas.create_oval(p.x - r, p.y - r, p.x + r, p.y + r, **kargs)
        self.items[item] = owner
        return item
    def draw_polygon(self, owner, pts, **kargs):
        assert all(isinstance(p, Point) for p in pts)
        args = []
        for p in pts:
            p = self.project(p)
            args.extend((p.x, p.y))
        item = self.canvas.create_polygon(*args, **kargs)
        self.items[item] = owner
        return item
    def draw_wireframe(self):
        for drawable in self.drawables:
            drawable.draw_wireframe(self)
    def draw(self):
        for drawable in self.drawables:
            drawable.draw(self)
    def update(self):
        self.clear()
        header = [
                "(theta, phi): ({:.3f}, {:.3f})".format(self.theta, self.phi),
                "(x, y, z): {}".format(self.camera_coords()),
                ]
        text = "\n".join(header) + "\n\n" + self.text
        item = self.canvas.create_text((10, 10), anchor="nw", text=text)
        self.items[item] = None
        if self.wireframe:
            self.draw_wireframe()
        else:
            self.draw()
    def display(self):
        self.update()
        mainloop()
    def _callback_commandline_up(self, event):
        self.move_camera(0, math.pi / 16)
        self.update()
    def _callback_commandline_down(self, event):
        self.move_camera(0, -math.pi / 16)
        self.update()
    def _callback_commandline_left(self, event):
        self.move_camera(math.pi / 16, 0)
        self.update()
    def _callback_commandline_right(self, event):
        self.move_camera(-math.pi / 16, 0)
        self.update()
    def _callback_commandline_equal(self, event):
        self.scale *= 1.2
        self.update()
    def _callback_commandline_minus(self, event):
        self.scale /= 1.2
        self.update()
    def _callback_commandline_shift_up(self, event):
        self.y_offset -= 10
        self.update()
    def _callback_commandline_shift_down(self, event):
        self.y_offset += 10
        self.update()
    def _callback_commandline_shift_left(self, event):
        self.x_offset -= 10
        self.update()
    def _callback_commandline_shift_right(self, event):
        self.x_offset += 10
        self.update()
    def _callback_commandline_shift_return(self, event):
        self.reset_viewport()
        self.update()
    def _callback_button_1(self, event):
        text = []
        closest = self.canvas.find_closest(event.x, event.y)[0]
        if closest in self.items and self.items[closest] is not None:
            text.append(self.items[closest].clicked(self, event, closest))
        self.text = "\n".join(text)
        self.update()
    def _callback_button_1_motion(self, event):
        # TODO drag rotation
        pass

# CLIMBING CLASSES

class Wall(Drawable):
    def __init__(self, points, surfaces):
        points = [Point(*v) for v in points]
        center = Point(
                (max(p.x for p in points) + min(p.x for p in points)) / 2,
                (max(p.y for p in points) + min(p.y for p in points)) / 2,
                (max(p.z for p in points) + min(p.z for p in points)) / 2)
        self.points = tuple(p - center for p in points)
        self.surfaces = [Surface(self.points[i] for i in surface) for surface in surfaces]
        self.canvas_items = {}
    def canvas_cleared(self):
        self.canvas_items.clear()
    def draw_wireframe(self, viewer, **kargs):
        for index, surface in enumerate(self.surfaces):
            item = viewer.draw_polygon(self, surface.points, outline="#000000", fill="", **kargs)
            self.canvas_items[item] = index
    def draw(self, viewer, **kargs):
        for index, surface in enumerate(self.surfaces):
            if surface.normal.dot(viewer.camera_coords()) > 0:
                item = viewer.draw_polygon(self, surface.points, outline="#000000", fill="#AAAAAA", activefill="#CCCCCC", **kargs)
                self.canvas_items[item] = index
    def clicked(self, viewer, event, item):
        wall_num = self.canvas_items[item]
        surface = self.surfaces[wall_num]
        # undo scaling and translation
        p1 = Point(0, (event.x - viewer.x_offset) / viewer.scale, -(event.y - viewer.y_offset) / viewer.scale)
        # create a second point on the beam
        p2 = p1 + Point(100, 0, 0)
        # undo rotation; first phi, then theta
        p1, p2 = viewer.unproject(p1), viewer.unproject(p2)
        # find the vector for the line
        vector = p2 - p1
        # find a scalar from p1 that gives a point on the plane
        scalar = (surface.origin - p1).dot(surface.normal) / (vector.dot(surface.normal))
        # find the point
        intersection = p1 + scalar * vector
        position = surface.coord2pos(intersection)
        text = []
        text.append("wall surface #{} {}".format(wall_num, position))
        return "\n".join(text)

class WallPosition:
    def __init__(self, surface, position):
        self.surface = surface
        self.position = position
    def real_coords(self):
        return self.surface.pos2coord(self.position)

class Hold(Drawable):
    def __init__(self, position, index=None, start_hold=False, finish_hold=False):
        self.position = position
        self.index = index
        self.start_hold = start_hold
        self.finish_hold = finish_hold
    def real_coords(self):
        return self.position.real_coords()
    def canvas_cleared(self):
        pass
    def draw_wireframe(self, viewer, **kargs):
        viewer.draw_circle(self, self.real_coords(), 5, **kargs)
    def draw(self, viewer, **kargs):
        if self.surface.normal.dot(viewer.camera_coords()) > 0:
            viewer.draw_circle(self, self.real_coords(), 5, **kargs)
    def clicked(self, viewer, event, item):
        text = []
        text.append("hold #{}".format(self.index))
        if self.start_hold:
            text.append("start hold")
        if self.finish_hold:
            text.append("finish hold")
        return "\n".join(text)


class Problem(Drawable):
    def __init__(self, wall, holds=None, start_holds=None, finish_holds=None):
        self.wall = wall
        self.holds = []
        self.start_holds = []
        self.finish_holds = []
        if holds is not None:
            for surface, x, y in holds:
                self.add_hold(surface, x, y)
        if start_holds is not None:
            for hold in start_holds:
                self.add_start_hold(hold)
        if finish_holds is not None:
            for hold in finish_holds:
                self.add_finish_hold(hold)
    def add_hold(self, surface, x, y):
        self.holds.append(Hold(WallPosition(self.wall.surfaces[surface], Point(x, y)), index=len(self.holds)))
    def add_start_hold(self, index):
        self.start_holds.append(index)
        self.holds[index].start_hold = True
    def add_finish_hold(self, index):
        self.finish_holds.append(index)
        self.holds[index].finish_hold = True
    def draw_wireframe(self, viewer, **kargs):
        self.wall.draw_wireframe(viewer)
        for index, hold in enumerate(self.holds):
            if index in self.start_holds:
                hold.draw_wireframe(viewer, fill="#FF0000")
            elif index in self.finish_holds:
                hold.draw_wireframe(viewer, fill="#00FF00")
            else:
                hold.draw_wireframe(viewer, fill="#0000FF")
    def draw(self, viewer, **kargs):
        self.wall.draw(viewer)
        for index, hold in enumerate(self.holds):
            if index in self.start_holds:
                hold.draw(viewer, fill="#FF0000")
            elif index in self.finish_holds:
                hold.draw(viewer, fill="#00FF00")
            else:
                hold.draw(viewer, fill="#0000FF")

# CLIMBER CLASSES

class Pose:
    mapping = {"left_hand": 0, "right_hand": 1, "left_foot": 2, "right_foot":3}
    def __init__(self, left_hand, right_hand, left_foot, right_foot):
        self.limbs = [left_hand, right_hand, left_foot, right_foot]
    @property
    def left_hand(self):
        return self.limbs[Pose.mapping["left_hand"]]
    @property
    def right_hand(self):
     return self.limbs[Pose.mapping["right_hand"]]
    @property
    def left_foot(self):
     return self.limbs[Pose.mapping["left_foot"]]
    @property
    def right_foot(self):
     return self.limbs[Pose.mapping["right_foot"]]
    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()
    def __hash__(self):
        return hash(self.as_tuple())
    def __str__(self):
        return str(self.as_tuple())
    def as_tuple(self):
        return (self.left_hand, self.right_hand, self.left_foot, self.right_foot)
    def move(self, limb, hold):
        return Pose(*(hold if index == Pose.mapping[limb] else h for index, h, in enumerate(self.limbs)))

class Climber:
    def __init__(self):
        self.height = 175
        self.armspan = 175
    def climb(self, problem):
        distances = Climber.create_graph(problem)
        poses = self.start_poses(problem)
        for pose in poses:
            for next_pose in self.possible_moves(pose, problem):
                pass
    def start_poses(self, problem):
        start_poses = []
        hand_holds = [None] + list(problem.start_holds)
        for left_hand, right_hand in product(hand_holds, repeat=2):
            if left_hand is None and right_hand is None:
                continue
            hand_height = max(problem.holds[hand].real_coords().z for hand in (left_hand, right_hand) if hand is not None)
            foot_holds = [None] + list(index for index, hold in enumerate(problem.holds) if hold.real_coords().z <= hand_height)
            for left_foot, right_foot in product(foot_holds, repeat=2):
                pose = Pose(left_hand, right_hand, left_foot, right_foot)
                if self.valid_pose(pose):
                    start_poses.append(pose)
        return start_poses
    def possible_moves(self, pose, problem):
        moves = []
        # this will only occur on the first move
        if pose.left_hand is None or pose.right_hand is None:
            for hold in range(len(problem.holds)):
                if pose.left_hand is None:
                    moves.append(pose.move("left_hand", hold))
                else:
                    moves.append(pose.move("right_hand", hold))
        else:
            for hold in range(len(problem.holds)):
                for limb in Pose.mapping:
                    moves.append(pose.move(limb, hold))
        return [move for move in moves if self.valid_pose(move)]
    def valid_pose(self, pose):
        return True
    def evaluate_pose(self, pose):
        pass
    @staticmethod
    def surface_distance(problem, wp1, wp2):
        # FIXME we don't want to count dimples between points
        vector = wp2.real_coords() - wp1.real_coords()
        surface = wp1.surface
        source = wp1.real_coords()
        distance = 0
        while surface != wp2.surface:
            # project vector onto wall
            projection = surface.project(vector)
            hold_source = surface.coord2pos(source)
            hold_vector = (surface.coord2pos(surface.project(source + projection)) - hold_source).normalize()
            # find the edge that this projection crosses
            rotated_points = copy(surface.points)
            rotated_points.rotate(-1)
            for p1, p2 in zip(surface.points, rotated_points):
                # find where the edge and the projection intersects
                edge_source = surface.coord2pos(p1)
                edge_vector = (surface.coord2pos(p2) - edge_source).normalize()
                # frame the system of equations as a projection
                solution = Point((edge_source - hold_source).dot(hold_vector), -(edge_source - hold_source).dot(edge_vector))
                # discard if the solution is not, in fact, an intersection
                if solution.x <= 0 or solution.y < 0 or hold_source + solution.x * hold_vector != edge_source + solution.y * edge_vector:
                    continue
                # add to the distance
                distance += solution.x
                # find the next wall
                candidates = [s for s in problem.wall.surfaces if s != surface and p1 in s.points and p2 in s.points]
                assert len(candidates) == 1
                source = surface.pos2coord(hold_source + solution.x * hold_vector)
                surface = candidates[0]
                # move on to the next wall
                break
            else:
                print("I DON'T KNOW WHAT'S GOING ON!!!")
                exit()
        # this is the wall with the ending point
        distance += (wp2.real_coords() - source).length
        return distance
    @staticmethod
    def create_graph(problem):
        distances = {}
        for i in range(len(problem.holds)):
            for j in range(i+1, len(problem.holds)):
                distance = Climber.surface_distance(problem, problem.holds[i].position, problem.holds[j].position)
                distances[(i, j)] = distance
                distances[(j, i)] = distance
        return distances

# MAIN

def create_wall_from_file(path):
    sections = []
    with open(path) as fd:
        sections = fd.read().strip().split("\n\n")
    vertices = tuple(tuple(float(r) for r in line.split()) for line in sections[0].strip().splitlines())
    surfaces = tuple(tuple(int(n) for n in line.split()) for line in sections[1].strip().splitlines())
    return Wall(vertices, surfaces)

def create_problem_from_file(path):
    sections = []
    path = realpath(expanduser(path))
    with open(path) as fd:
        sections = fd.read().strip().split("\n\n")
    wall_path = sections[0].strip()
    if not wall_path.startswith("/"):
        wall_path = join_path(dirname(path), wall_path)
    wall = create_wall_from_file(wall_path)
    holds = []
    for line in sections[1].strip().splitlines():
        surface, x, y = line.split()
        holds.append((int(surface), float(x), float(y)))
    starts = tuple(int(n) for n in sections[2].strip().split())
    finishes = tuple(int(n) for n in sections[3].strip().split())
    return Problem(wall, holds, starts, finishes)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("problem", nargs=1, help="problem file")
    args = arg_parser.parse_args()

    prob = create_problem_from_file(args.problem[0])
    climber = Climber()
    climber.climb(prob)

    viewer = IsometricViewer(800, 600)
    viewer.add_drawable(prob)
    viewer.display()
