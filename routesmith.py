#!/usr/bin/env python3

import math
from abc import ABCMeta, abstractmethod
from numbers import Real

from tkinter import *

# MODELING CLASSES

class Point:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z
	def __eq__(self, other):
		return hash(self) == hash(other)
	def __ne__(self, other):
		return not (self == other)
	def __hash__(self):
		return hash((self.x, self.y, self.z))
	def __str__(self):
		return "({}, {}, {})".format(self.x, self.y, self.z)
	def __add__(self, p):
		assert isinstance(p, Point)
		return Point(self.x + p.x, self.y + p.y, self.z + p.z)
	def __sub__(self, p):
		assert isinstance(p, Point)
		return Point(self.x - p.x, self.y - p.y, self.z - p.z)
	def __neg__(self):
		return Point(-self.x, -self.y, -self.z)
	def __rmul__(self, r):
		assert isinstance(r, Real)
		return Point(r * self.x, r * self.y, r * self.z)
	def dot(self, p):
		assert isinstance(p, Point)
		return self.x * p.x + self.y * p.y + self.z * p.z
	def cross(self, p):
		assert isinstance(p, Point)
		return Point(self.y * p.z - self.z * p.y, self.z * p.x - self.x * p.z, self.x * p.y - self.y * p.x)
	def rotate(self, theta, phi):
		sin_theta = math.sin(theta)
		cos_theta = math.cos(theta)
		sin_phi = math.sin(phi)
		cos_phi = math.cos(phi)
		x = self.x * cos_theta + self.z * sin_theta
		y = self.x * sin_theta * sin_phi
		y += self.y * cos_phi
		y -= self.z * cos_theta * sin_phi
		z = -self.x * sin_theta * cos_phi
		z += self.y * sin_theta
		z += self.z * cos_theta * cos_phi
		return Point(x, y, z)
	def clone(self):
		return Point(self.x, self.y, self.z)
	def length(self):
		return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
	def normalize(self):
		l = self.length()
		return Point(self.x / l, self.y / l, self.z / l)

class Surface:
	TOLERANCE = 0.0005
	def __init__(self, points=None):
		if points is None:
			self.points = []
		else:
			self.points = list(points)
		# find the plane (ax + by + cz = constant)
		self.normal = (self.points[1] - self.points[0]).cross(self.points[2] - self.points[1]).normalize()
		self.constant = self.normal.dot(self.points[0])
		# make sure points are co-planar
		assert all(abs(p.dot(self.normal) - self.constant) < Surface.TOLERANCE for p in self.points), "Surfaces are not planar"
		# define an origin for this surface
		self.origin = Point(min(p.x for p in self.points), min(p.y for p in self.points), 0)
		if self.normal.z != 0:
			self.origin.z = (self.normal.dot(self.origin) - self.constant) / -self.normal.z
		# define the basis vetors
		self.basis_x = Point(self.normal.z, 0, -self.normal.x).normalize()
		self.basis_y = self.normal.cross(self.basis_x)
		# TODO make sure surface is simple (no line intersections)

# GRAPHICS CLASSES

class Drawable(metaclass=ABCMeta):
	@abstractmethod
	def draw_wiremesh(self):
		raise NotImplementedError()

class IsometricViewer:
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.wiremesh = True
		self.reset_viewport()
		self.init_gui()
		self.drawables = []
	def reset_viewport(self):
		self.theta = (math.pi / 4)
		self.phi = -(math.pi / 16)
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
	def add_drawable(self, drawable):
		assert isinstance(drawable, Drawable)
		self.drawables.append(drawable)
	def project(self, point):
		projected = point.rotate(self.theta, self.phi)
		return (projected.x * self.scale + self.x_offset, -projected.y * self.scale + self.y_offset)
	def move_camera(self, theta, phi):
		self.theta += theta
		if self.theta > 2 * math.pi:
			self.theta -= 2 * math.pi
		elif self.theta < 0:
			self.theta += 2 * math.pi
		self.phi += phi
		if self.phi > math.pi / 2:
			self.theta = -self.theta
			self.phi = (math.pi / 2) - self.phi;
		elif self.theta < 0:
			self.theta = -self.theta
			self.phi -= math.pi / 2;
	def clear(self):
		self.canvas.create_rectangle(0, 0, self.width + 10, self.height + 10, fill="white")
	def draw_line(self, p1, p2):
		assert isinstance(p1, Point)
		assert isinstance(p2, Point)
		p1x, p1y = self.project(p1)
		p2x, p2y = self.project(p2)
		self.canvas.create_line(p1x, p1y, p2x, p2y)
	def draw_wiremesh(self):
		for drawable in self.drawables:
			drawable.draw_wiremesh(self)
	def update(self):
		self.clear()
		if self.wiremesh:
			self.draw_wiremesh()
		else:
			self.draw()
	def display(self):
		self.update()
		mainloop()
	def _callback_commandline_up(self, *args):
		self.move_camera(0, -math.pi / 16)
		self.update()
	def _callback_commandline_down(self, *args):
		self.move_camera(0, math.pi / 16)
		self.update()
	def _callback_commandline_left(self, *args):
		self.move_camera(-math.pi / 16, 0)
		self.update()
	def _callback_commandline_right(self, *args):
		self.move_camera(math.pi / 16, 0)
		self.update()
	def _callback_commandline_equal(self, *args):
		self.scale *= 1.2
		self.update()
	def _callback_commandline_minus(self, *args):
		self.scale /= 1.2
		self.update()
	def _callback_commandline_shift_up(self, *args):
		self.y_offset -= 10
		self.update()
	def _callback_commandline_shift_down(self, *args):
		self.y_offset += 10
		self.update()
	def _callback_commandline_shift_left(self, *args):
		self.x_offset -= 10
		self.update()
	def _callback_commandline_shift_right(self, *args):
		self.x_offset += 10
		self.update()
	def _callback_commandline_shift_return(self, *args):
		self.reset()
		self.update()

# CLIMBING CLASSES

class Wall(Drawable):
	def __init__(self, points, surfaces):
		center = Point(
				(max(p.x for p in points) + min(p.x for p in points)) / 2,
				(max(p.y for p in points) + min(p.y for p in points)) / 2,
				(max(p.z for p in points) + min(p.z for p in points)) / 2)
		self.points = tuple(p - center for p in points)
		self.surfaces = []
		for surface in surfaces:
			self.surfaces.append(Surface(self.points[i] for i in surface))
	def draw_wiremesh(self, viewer):
		for surface in self.surfaces:
			for i in range(len(surface.points)):
				viewer.draw_line(surface.points[i-1], surface.points[i])

class Hold(Drawable):
	def __init__(self, surface, x, y):
		self.surface = surface
		self.x = x
		self.y = y
	def draw_wiremesh(self, viewer):
		viewer.draw_line(self.surface.origin, self.surface.origin + (self.x * self.surface.basis_x + self.y * self.surface.basis_y))

class Problem(Drawable):
	def __init__(self, wall):
		self.wall = wall
		self.holds = []
		self.start_holds = []
		self.finish_holds = []
	def add_hold(self, surface, x, y):
		self.holds.append(Hold(self.wall.surfaces[surface], x, y))
	def set_start_hold(self, index):
		self.start_holds.append(index)
	def set_finish_hold(self, index):
		self.finish_holds.append(index)
	def draw_wiremesh(self, viewer):
		self.wall.draw_wiremesh(viewer)
		for hold in self.holds:
			hold.draw_wiremesh(viewer)

CUSTOM = Wall((
				Point(  0.0, 180,     0), # 0
				Point(  0.0,   0,     0),
				Point( 67.5, 180, 103.5),
				Point( 67.5,  45,     0),
				Point(135.0, 180,     0),
				Point(157.5, 180,     0), # 5
				Point(261.0, 180, 103.5),
				Point(261.0,  45,     0),
				Point(450.0, 180, 103.5),
				Point(450.0,  45,     0),
				Point(450.0,   0,     0), # 10
				Point(553.5,  45, 103.5),
				Point(553.5,   0, 103.5),
				Point(450.0, 180, 225.0),
				Point(553.5,  45, 225.0),
				Point(553.5,   0, 225.0), # 15
		), (
				(0, 1, 10, 9, 7, 5, 4, 3),
				(0, 3, 2),
				(3, 4, 2),
				(5, 7, 6),
				(6, 7, 9, 8),
				(8, 9, 11),
				(9, 10, 12, 11),
				(8, 11, 14, 13),
				(11, 12, 15, 14),
		))

if __name__ == "__main__":
	prob = Problem(CUSTOM)
	prob.add_hold(0, 300, 20)
	viewer = IsometricViewer(800, 600)
	viewer.add_drawable(prob)
	viewer.display()
