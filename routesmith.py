#!/usr/bin/env python3

import math
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from collections import Counter
from copy import copy
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
    def length(self):
        return math.sqrt(sum(i * i for i in self.values))
    def __eq__(self, other):
        assert type(self) == type(other)
        return all(x == y for x, y in zip(self.values, other.values))
    def __hash__(self):
        return hash(self.values)
    def __str__(self):
        return "(" + ", ".join(format(i, ".3f") for i in self.values) + ")"
    def __add__(self, p):
        assert type(self) == type(p) and self.dimensions == p.dimensions
        return self.__class__(*(i + j for i, j in zip(self.values, p.values)))
    def __sub__(self, p):
        assert type(self) == type(p) and self.dimensions == p.dimensions
        return self.__class__(*(i - j for i, j in zip(self.values, p.values)))
    def __neg__(self):
        return self.__class__(*(-i for i in self.values))
    def __rmul__(self, r):
        assert isinstance(r, Real)
        return self.__class__(*(r * i for i in self.values))
    def dot(self, p):
        assert type(self) == type(p) and self.dimensions == p.dimensions
        return sum(i * j for i, j in zip(self.values, p.values))
    def angle(self, p):
        return math.acos(self.dot(p) / (self.length * p.length))
    def project(self, p):
        assert type(self) == type(p) and self.dimensions == p.dimensions
        return self - (self.dot(p) / p.length) * p
    def normalize(self):
        l = self.length
        return self.__class__(*(i / l for i in self.values))

class Point2(Point):
    def __init__(self, x, y):
        super().__init__(x, y)
    @property
    def x(self):
        return self.values[0]
    @property
    def y(self):
        return self.values[1]
    def rotate(self, theta):
        return self.__class__(self.x * math.cos(theta) + self.y * math.sin(theta), self.x * -math.sin(theta) + self.y * math.cos(theta))

class Point3(Point):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
    @property
    def x(self):
        return self.values[0]
    @property
    def y(self):
        return self.values[1]
    @property
    def z(self):
        return self.values[2]
    def cross(self, p):
        assert type(self) == type(p)
        return self.__class__(self.y * p.z - self.z * p.y, self.z * p.x - self.x * p.z, self.x * p.y - self.y * p.x)
    def rotate(self, theta, phi):
        assert self.dimensions == 3
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        p = self.__class__(self.x * cos_theta + self.y * -sin_theta, self.x * sin_theta + self.y * cos_theta, self.z)
        p = self.__class__(p.x * cos_phi + p.z * sin_phi, p.y, p.x * -sin_phi + p.z * cos_phi)
        return p

class Plane:
    def __init__(self, points):
        assert len(points) >= 3
        points = list(points)
        # keep a reference point
        self.reference = points[0]
        # find the plane (ax + by + cz = constant)
        self.normal = Counter((points[i-1] - points[i-2]).cross(points[i] - points[i-1]).normalize() for i in range(1, len(points))).most_common(1)[0][0]
        self.constant = self.normal.dot(points[0])
        # make sure points are co-planar
        assert all(self.on_plane(p) for p in points), "Vertices are not planar"
    def on_plane(self, point):
        return abs(point.dot(self.normal) - self.constant) < TOLERANCE
    def project(self, vector):
        return vector - (vector - self.reference).dot(self.normal) * self.normal

# GRAPHICS CLASSES

class Drawable(metaclass=ABCMeta):
    @abstractmethod
    def canvas_cleared(self):
        raise NotImplementedError()
    @abstractmethod
    def draw_wireframe(self, viewer, **kargs):
        raise NotImplementedError()
    @abstractmethod
    def draw(self, viewer, **kargs):
        raise NotImplementedError()

class Clickable(metaclass=ABCMeta):
    @abstractmethod
    def clicked(self, viewer, event, item):
        raise NotImplementedError()

class IsometricViewer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.wireframe = True
        self.drawables = []
        self.items = {}
        self.text = ""
        self.theta = 0
        self.phi = 0
        self.scale = 0
        self.x_offset = 0
        self.y_offset = 0
        self.canvas = Canvas(Tk(), width=self.width, height=self.height)
        self.reset_viewport()
        self.init_gui()
    @property
    def camera_coords(self):
        return Point3(math.cos(self.theta) * math.cos(self.phi), -math.sin(self.theta) * math.cos(self.phi), math.sin(self.phi))
    def reset_viewport(self):
        self.theta = math.pi / 8
        self.phi = math.pi / 16
        self.scale = 1
        self.x_offset = self.width / 2
        self.y_offset = self.height / 2
    def init_gui(self):
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
        return Point2(projected.y * self.scale + self.x_offset, -projected.z * self.scale + self.y_offset)
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
    def clear(self):
        for item in self.items:
            self.canvas.delete(item)
    def draw_line(self, owner, p1, p2, **kargs):
        assert isinstance(p1, Point3)
        assert isinstance(p2, Point3)
        p1 = self.project(p1)
        p2 = self.project(p2)
        item = self.canvas.create_line(p1.x, p1.y, p2.x, p2.y, **kargs)
        self.items[item] = owner
        return item
    def draw_ellipse(self, owner, corners, **kargs):
        assert all(isinstance(p, Point3) for p in corners)
        item = self.draw_polygon(owner, corners, outline="#000000", smooth=1, **kargs)
        self.items[item] = owner
        return item
    def draw_polygon(self, owner, pts, **kargs):
        assert all(isinstance(p, Point3) for p in pts)
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
            "(x, y, z): {}".format(self.camera_coords),
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
        overlapping = self.canvas.find_overlapping(event.x, event.y, event.x+1, event.y+1)
        if closest in overlapping and closest in self.items and self.items[closest] is not None:
            text.append(self.items[closest].clicked(self, event, closest))
        self.text = "\n".join(text)
        self.update()
    def _callback_button_1_motion(self, event):
        pass

# CLIMBING CLASSES

class Surface:
    def __init__(self, points):
        self.points = list(points)
        assert len(self.points) >= 3
        self.plane = Plane(self.points)
        # use the first point as a temporary origin
        self.origin = self.points[0]
        # define the first basis, depending on whether the surface is horizontal
        if abs(self.normal.z) == 1:
            self.basis_x = Point3(0, self.normal.z, -self.normal.x).normalize()
            if self.normal.cross(self.basis_x).x < 0:
                self.basis_x = -self.basis_x
        else:
            self.basis_x = Point3(self.normal.y, -self.normal.x, 0).normalize()
            if self.normal.cross(self.basis_x).z < 0:
                self.basis_x = -self.basis_x
        self.basis_y = self.normal.cross(self.basis_x)
        # find the lowest transformed coordinates for each basis
        min_x = min((self.points[i] - self.origin).dot(self.basis_x) for i in range(len(self.points)))
        min_y = min((self.points[i] - self.origin).dot(self.basis_y) for i in range(len(self.points)))
        # move the origin to that point
        self.origin += min_x * self.basis_x + min_y * self.basis_y
    @property
    def normal(self):
        return self.plane.normal
    @property
    def constant(self):
        return self.plane.constant
    def pos_to_coords(self, point):
        return self.origin + (point.x * self.basis_x + point.y * self.basis_y)
    def coords_to_pos(self, point):
        assert self.plane.on_plane(point)
        offset = (point - self.origin)
        return Point2(offset.dot(self.basis_x), offset.dot(self.basis_y))
    def project(self, vector):
        return vector - (vector - self.points[0]).dot(self.normal) * self.normal

class WallPosition:
    def __init__(self, surface, position):
        self.surface = surface
        self.position = position
    @property
    def real_coords(self):
        return self.surface.pos_to_coords(self.position)

class Wall(Drawable, Clickable):
    def __init__(self, points, surfaces):
        points = [Point3(*v) for v in points]
        center = Point3(
            (max(p.x for p in points) + min(p.x for p in points)) / 2,
            (max(p.y for p in points) + min(p.y for p in points)) / 2,
            (max(p.z for p in points) + min(p.z for p in points)) / 2)
        self.points = tuple(p - center for p in points)
        self.surfaces = [Surface(self.points[i] for i in surface) for surface in surfaces]
        self.canvas_items = {}
    def surface_distance(self, wp1, wp2):
        # FIXME
        # what we actually want to is take a plane along the two holds, then
        # trace the wall to see if it ever crosses the original line. The plane
        # is defined by the direct vector and the vector perpendicular to its
        # projection on the x-y plane. That is, at either hold, the
        # plane should intersect the constant z plane such that the resulting
        # line is perpendicular to the vector's projection onto the constant
        # z plane. Some adjustment may be needed for holds along a roof.
        vector = wp2.real_coords - wp1.real_coords
        surface = wp1.surface
        source = wp1.real_coords
        distance = 0
        while surface != wp2.surface:
            # project vector onto wall
            projection = surface.project(vector)
            hold_source = surface.coords_to_pos(source)
            hold_vector = (surface.coords_to_pos(surface.project(source + projection)) - hold_source).normalize()
            # find the edge that this projection crosses
            rotated_points = copy(surface.points)
            rotated_points.rotate(-1)
            for p1, p2 in zip(surface.points, rotated_points):
                # find where the edge and the projection intersects
                edge_source = surface.coords_to_pos(p1)
                edge_vector = (surface.coords_to_pos(p2) - edge_source).normalize()
                # frame the system of equations as a projection
                solution = Point2((edge_source - hold_source).dot(hold_vector), -(edge_source - hold_source).dot(edge_vector))
                # discard if the solution is not, in fact, an intersection
                if solution.x <= 0 or solution.y < 0 or hold_source + solution.x * hold_vector != edge_source + solution.y * edge_vector:
                    continue
                # add to the distance
                distance += solution.x
                # find the next wall
                candidates = [s for s in self.surfaces if s != surface and p1 in s.points and p2 in s.points]
                assert len(candidates) == 1
                source = surface.pos_to_coords(hold_source + solution.x * hold_vector)
                surface = candidates[0]
                # move on to the next wall
                break
            else:
                print("I DON'T KNOW WHAT'S GOING ON!!!")
                exit()
        # this is the wall with the ending point
        distance += (wp2.real_coords - source).length
        return distance
    def canvas_cleared(self):
        self.canvas_items.clear()
    def draw_wireframe(self, viewer, **kargs):
        for index, surface in enumerate(self.surfaces):
            item = viewer.draw_polygon(self, surface.points, outline="#000000", fill="", **kargs)
            self.canvas_items[item] = index
    def draw(self, viewer, **kargs):
        for index, surface in enumerate(self.surfaces):
            if surface.normal.dot(viewer.camera_coords) > 0:
                item = viewer.draw_polygon(self, surface.points, outline="#000000", fill="#AAAAAA", activefill="#CCCCCC", **kargs)
                self.canvas_items[item] = index
    def clicked(self, viewer, event, item):
        wall_num = self.canvas_items[item]
        surface = self.surfaces[wall_num]
        # undo scaling and translation
        p1 = Point3(0, (event.x - viewer.x_offset) / viewer.scale, -(event.y - viewer.y_offset) / viewer.scale)
        # create a second point on the beam
        p2 = p1 + Point3(100, 0, 0)
        # undo rotation; first phi, then theta
        p1, p2 = viewer.unproject(p1), viewer.unproject(p2)
        # find the vector for the line
        vector = p2 - p1
        # find a scalar from p1 that gives a point on the plane
        scalar = (surface.origin - p1).dot(surface.normal) / (vector.dot(surface.normal))
        # find the point
        intersection = p1 + scalar * vector
        position = surface.coords_to_pos(intersection)
        text = []
        text.append("wall surface #{} {}".format(wall_num, position))
        return "\n".join(text)

class Hold():
    def __init__(self, wall_position, width, depth, angle, comment=None):
        self.wall_position = wall_position
        self.width = width
        self.depth = depth
        self.angle = angle
        self.comment = comment
    @property
    def surface(self):
        return self.wall_position.surface
    @property
    def position(self):
        return self.wall_position.position
    @property
    def real_coords(self):
        return self.wall_position.real_coords

class Problem(Drawable, Clickable):
    def __init__(self, wall, holds=None, start_holds=None, finish_holds=None):
        self.wall = wall
        self.holds = []
        self.start_holds = []
        self.finish_holds = []
        if holds is not None:
            for surface, x, y, width, depth, angle, comment in holds:
                self.add_hold(surface, x, y, width, depth, angle, comment)
        if start_holds is not None:
            for hold in start_holds:
                self.add_start_hold(hold)
        if finish_holds is not None:
            for hold in finish_holds:
                self.add_finish_hold(hold)
        self.canvas_items = {}
    def add_hold(self, surface, x, y, width, depth, angle, comment):
        self.holds.append(Hold(WallPosition(self.wall.surfaces[surface], Point2(x, y)), width, depth, angle, comment=comment))
    def add_start_hold(self, index):
        self.start_holds.append(index)
    def add_finish_hold(self, index):
        self.finish_holds.append(index)
    def canvas_cleared(self):
        self.canvas_items.clear()
    def draw_wireframe(self, viewer, **kargs):
        self.wall.draw_wireframe(viewer)
        for index in range(len(self.holds)):
            if index in self.start_holds:
                self.draw_hold_wireframe(viewer, index, fill="#FF0000", outline="#000000")
            elif index in self.finish_holds:
                self.draw_hold_wireframe(viewer, index, fill="#00FF00", outline="#000000")
            else:
                self.draw_hold_wireframe(viewer, index, fill="#0000FF", outline="#000000")
    def draw(self, viewer, **kargs):
        self.wall.draw(viewer)
        for index in range(len(self.holds)):
            if index in self.start_holds:
                self.draw_hold(viewer, index, fill="#FF0000", outline="#000000")
            elif index in self.finish_holds:
                self.draw_hold(viewer, index, fill="#00FF00", outline="#000000")
            else:
                self.draw_hold(viewer, index, fill="#0000FF", outline="#000000")
    def draw_hold_wireframe(self, viewer, hold_num, **kargs):
        hold = self.holds[hold_num]
        corners = []
        for theta in range(17):
            corners.append(hold.surface.pos_to_coords(
                hold.position - Point2((hold.width / 2) * math.cos(theta * math.pi / 16), (hold.depth) * math.sin(theta * math.pi / 16)).rotate(hold.angle)))
        item = viewer.draw_polygon(self, corners, **kargs)
        self.canvas_items[item] = hold_num
    def draw_hold(self, viewer, hold_num, **kargs):
        hold = self.holds[hold_num]
        if hold.surface.normal.dot(viewer.camera_coords) > 0:
            corners = []
            for theta in range(17):
                corners.append(hold.surface.pos_to_coords(
                    hold.position - Point2((hold.width / 2) * math.cos(theta * math.pi / 16), (hold.depth) * math.sin(theta * math.pi / 16)).rotate(hold.angle)))
            item = viewer.draw_ellipse(self, corners, **kargs)
            self.canvas_items[item] = hold_num
    def clicked(self, viewer, event, item):
        hold_num = self.canvas_items[item]
        hold = self.holds[hold_num]
        text = []
        text.append("hold #{}".format(hold_num))
        text.append("surface #{} {}".format(self.wall.surfaces.index(hold.surface), hold.position))
        if hold.comment:
            text.append(hold.comment)
        if hold_num in self.start_holds:
            text.append("start hold")
        if hold_num in self.finish_holds:
            text.append("finish hold")
        return "\n".join(text)

# MAIN

def create_wall_from_file(path):
    sections = []
    path = realpath(expanduser(path))
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
        if "#" in line:
            coords, comment = (text.strip() for text in line.split("#"))
        else:
            coords = line
            comment = ""
        surface, x, y, width, depth, angle = coords.split()
        holds.append((int(surface), float(x), float(y), float(width), float(depth), math.radians(float(angle)), comment))
    starts = tuple(int(n) for n in sections[2].strip().split())
    finishes = tuple(int(n) for n in sections[3].strip().split())
    return Problem(wall, holds, starts, finishes)

def detect_file_type(path):
    sections = []
    path = realpath(expanduser(path))
    with open(path) as fd:
        sections = fd.read().strip().split("\n\n")
    if len(sections) == 2:
        return "wall"
    elif len(sections) == 4:
        return "problem"
    else:
        return None

def main():
    arg_parser = ArgumentParser()
    arg_parser.set_defaults(climb=False, view=True)
    arg_parser.add_argument("path", nargs=1, help="wall or problem file")
    arg_parser.add_argument("--blind", action="store_false", help="don't show visualization")
    args = arg_parser.parse_args()
    args.path = args.path[0]

    file_type = detect_file_type(args.path)
    interactable = None
    if file_type == "wall":
        wall = create_wall_from_file(args.path)
        interactable = wall
    elif file_type == "problem":
        interactable = create_problem_from_file(args.path)
    else:
        print("unknown file type")
        exit(1)

    viewer = IsometricViewer(800, 600)
    viewer.add_drawable(interactable)
    viewer.display()

if __name__ == "__main__":
    main()
