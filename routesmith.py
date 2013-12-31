#!/usr/bin/env python3

import math # FIXME should use numpy

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
		return Point(self.x + p.x, self.y + p.y, self.z + p.z)
	def __sub__(self, p):
		return Point(self.x - p.x, self.y - p.y, self.z - p.z)
	def dot(self, p):
		return self.x * p.x + self.y * p.y + self.z * p.z
	def cross(self, p):
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

class Line:
	def __init__(self, p1, p2):
		if hash(p1) < hash(p2):
			self.p1 = p1
			self.p2 = p2
		else:
			self.p1 = p2
			self.p2 = p1
	def __eq__(self, other):
		return self.p1 == other.p1 and self.p2 == other.p2
	def __ne__(self, other):
		return not (self == other)
	def __hash__(self):
		return hash((hash(self.p1), hash(self.p2)))
	def __str__(self):
		return "({}, {})".format(self.p1, self.p2)
	def clone(self):
		return Line(self.p1, self.p2)

class Surface:
	def __init__(self, points=None):
		if points is None:
			self.points = []
		else:
			self.points = list(points)
	def get_lines(self):
		return [Line(self.points[i], self.points[i + 1]) for i in range(-1, len(self.points) - 1)]

# GRAPHICS CLASSES

class Viewport:
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.reset()
	def reset(self):
		self.theta = 0
		self.phi = 0
		self.scale = 1
		self.x_offset = self.width / 2
		self.y_offset = self.height / 2
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

class Viewer:
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.canvas = Canvas(Tk(), width=self.width, height=self.height)
		self.canvas.pack()
		self.viewport = Viewport(self.width, self.height)
		self.viewport.theta = (math.pi / 4)
		self.viewport.phi = -(math.pi / 16)
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
		self.surfaces = []
		self.canvas.focus_set()
	def add_surface(self, surface):
		self.surfaces.append(surface)
	def clear(self):
		self.canvas.create_rectangle(0, 0, self.width + 10, self.height + 10, fill="white")
	def draw(self):
		lines = set()
		for surface in self.surfaces:
			lines = lines.union(surface.get_lines())
		for line in lines:
			x1, y1 = self.viewport.project(line.p1)
			x2, y2 = self.viewport.project(line.p2)
			self.canvas.create_line(x1, y1, x2, y2)
	def mainloop(self):
		self.draw()
		mainloop()
	def _callback_commandline_up(self, *args):
		self.viewport.move_camera(0, -math.pi / 16)
		self.clear()
		self.draw()
	def _callback_commandline_down(self, *args):
		self.viewport.move_camera(0, math.pi / 16)
		self.clear()
		self.draw()
	def _callback_commandline_left(self, *args):
		self.viewport.move_camera(-math.pi / 16, 0)
		self.clear()
		self.draw()
	def _callback_commandline_right(self, *args):
		self.viewport.move_camera(math.pi / 16, 0)
		self.clear()
		self.draw()
	def _callback_commandline_equal(self, *args):
		self.viewport.scale *= 1.2
		self.clear()
		self.draw()
	def _callback_commandline_minus(self, *args):
		self.viewport.scale /= 1.2
		self.clear()
		self.draw()
	def _callback_commandline_shift_up(self, *args):
		self.viewport.y_offset -= 10
		self.clear()
		self.draw()
	def _callback_commandline_shift_down(self, *args):
		self.viewport.y_offset += 10
		self.clear()
		self.draw()
	def _callback_commandline_shift_left(self, *args):
		self.viewport.x_offset -= 10
		self.clear()
		self.draw()
	def _callback_commandline_shift_right(self, *args):
		self.viewport.x_offset += 10
		self.clear()
		self.draw()
	def _callback_commandline_shift_return(self, *args):
		self.viewport.reset()
		self.viewport.theta = (math.pi / 4)
		self.viewport.phi = -(math.pi / 16)
		self.clear()
		self.draw()

# CLIMBING CLASSES

class Wall:
	def __init__(self, points, surfaces):
		self.points = points
		self.surfaces = surfaces
		center = Point(
				(max(p.x for p in self.points) + min(p.x for p in self.points)) / 2,
				(max(p.y for p in self.points) + min(p.y for p in self.points)) / 2,
				(max(p.z for p in self.points) + min(p.z for p in self.points)) / 2)
		self.points = tuple(p - center for p in self.points)

class Hold:
	def __init__(self, wall, surface, position):
		self.wall = wall
		self.surface = surface
		self.position = position
		pass

class Problem:
	def __init__(self):
		self.holds = []
		self.start_holds = []
		self.finish_holds = []
	def set_start_hold(self, index):
		pass
	def set_finish_hold(self, index):
		pass

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
	viewer = Viewer(800, 600)
	for surface in CUSTOM.surfaces:
		viewer.add_surface(Surface(CUSTOM.points[i] for i in surface))
	viewer.mainloop()
