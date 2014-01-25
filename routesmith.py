#!/usr/bin/env python3

import math
from abc import ABCMeta, abstractmethod
from collections import Counter, deque
from copy import copy
from numbers import Real

from tkinter import *

# CONSTANTS

TOLERANCE = 0.0005

# ABSTRACT CLASSES

class Point():
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
	def __eq__(self, other):
		return hash(self) == hash(other)
	def __ne__(self, other):
		return not (self == other)
	def __hash__(self):
		return hash(self.values)
	def __str__(self):
		return "(" + ", ".join("{}".format(i) for i in self.values) + ")"
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
		return math.acos(self.dot(p) / (self.length() * p.length()))
	def rotate(self, theta, phi):
		assert self.dimensions == 3
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
		return Point(*self.values)
	def length(self):
		return math.sqrt(sum(i * i for i in self.values))
	def normalize(self):
		l = self.length()
		return Point(*(i / l for i in self.values))

class Drawable(metaclass=ABCMeta):
	@abstractmethod
	def draw_wireframe(self):
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
		# define the first basis
		if abs(self.normal.y) == 1:
			# if the surface is horizontal, use the plane along z=0
			self.basis_x = Point(self.normal.y, -self.normal.x, 0).normalize()
		else:
			# otherwise, use the plane along y=0
			self.basis_x = Point(self.normal.z, 0, -self.normal.x).normalize()
		self.basis_y = self.normal.cross(self.basis_x)
		if self.basis_y.y < 0:
			self.basis_x = -self.basis_x
			self.basis_y = -self.basis_y
		# find the lowest transformed coordinates for each basis
		min_x = min((self.points[i] - self.origin).dot(self.basis_x) for i in range(len(self.points)))
		min_y = min((self.points[i] - self.origin).dot(self.basis_y) for i in range(len(self.points)))
		# move the origin to that point
		self.origin += min_x * self.basis_x + min_y * self.basis_y
		# TODO make sure surface is simple (no line intersections)
	def real_coords(self, point):
		return self.origin + (point.x * self.basis_x + point.y * self.basis_y)

# GRAPHICS CLASSES

class IsometricViewer:
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.wireframe = True
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
		return Point(projected.x * self.scale + self.x_offset, -projected.y * self.scale + self.y_offset)
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
	def draw_line(self, p1, p2, **kargs):
		assert isinstance(p1, Point)
		assert isinstance(p2, Point)
		p1 = self.project(p1)
		p2 = self.project(p2)
		self.canvas.create_line(p1.x, p1.y, p2.x, p2.y, **kargs)
	def draw_circle(self, p, r, **kargs):
		assert isinstance(p, Point)
		p = self.project(p)
		self.canvas.create_oval(p.x - r, p.y - r, p.x + r, p.y + r, **kargs)
	def draw_polygon(self, pts, **kargs):
		assert all(isinstance(p, Point) for p in pts)
		args = []
		for p in pts:
			p = self.project(p)
			args.extend((p.x, p.y))
		self.canvas.create_polygon(*args, **kargs)
	def draw_wireframe(self):
		for drawable in self.drawables:
			drawable.draw_wireframe(self)
	def update(self):
		self.clear()
		if self.wireframe:
			self.draw_wireframe()
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
		self.reset_viewport()
		self.update()

# CLIMBING CLASSES

class Wall(Drawable):
	def __init__(self, points, surfaces):
		points = [Point(*v) for v in points]
		center = Point(
				(max(p.x for p in points) + min(p.x for p in points)) / 2,
				(max(p.y for p in points) + min(p.y for p in points)) / 2,
				(max(p.z for p in points) + min(p.z for p in points)) / 2)
		center = Point(0, 0, 0)
		self.points = tuple(p - center for p in points)
		self.surfaces = []
		for surface in surfaces:
			self.surfaces.append(Surface(self.points[i] for i in surface))
	def draw_wireframe(self, viewer, **kargs):
		for surface in self.surfaces:
			viewer.draw_polygon(surface.points, outline="#000000", fill="", **kargs)

class Hold(Drawable):
	def __init__(self, surface, x, y):
		self.surface = surface
		self.position = Point(x, y)
	def real_coords(self):
		return self.surface.real_coords(self.position)
	def draw_wireframe(self, viewer, **kargs):
		viewer.draw_circle(self.real_coords(), 5, **kargs)

class Problem(Drawable):
	def __init__(self, wall):
		self.wall = wall
		self.holds = []
		self.start_holds = []
		self.finish_holds = []
	def add_hold(self, surface, x, y):
		self.holds.append(Hold(self.wall.surfaces[surface], x, y))
	def add_start_hold(self, index):
		self.start_holds.append(index)
	def add_finish_hold(self, index):
		self.finish_holds.append(index)
	def draw_wireframe(self, viewer):
		self.wall.draw_wireframe(viewer)
		for index, hold in enumerate(self.holds):
			if index in self.start_holds:
				hold.draw_wireframe(viewer, fill="#FF0000")
			elif index in self.finish_holds:
				hold.draw_wireframe(viewer, fill="#00FF00")
			else:
				hold.draw_wireframe(viewer, fill="#0000FF")
	def create_graph(self):
		distances = {}
		for i in range(len(self.holds)):
			for j in range(i+1, len(self.holds)):
				h1 = self.holds[i]
				h2 = self.holds[j]
				vector = h2.real_coords() - h1.real_coords()
				# figure out if the wall between is convex or concave
				if vector.angle(h1.surface.normal) <= math.pi / 2:
					# it's not a bulge; distance is simply straight line distance
					# FIXME technically, a bulge could appear *between* the holds
					distances[(i, j)] = vector.length()
				else:
					# it's a bulge; need to calculate wall distance
					print(("start hold = ", str(h1.real_coords())))
					print(("end hold = ", str(h2.real_coords())))
					print()
					surface = h1.surface
					print(list(str(p) for p in surface.points))
					print()
					source = h1.real_coords()
					distance = 0
					done = False
					while not done:
						# project vector onto wall
						projection = vector - (vector.dot(surface.normal)) * surface.normal
						# find the edge that this projection crosses
						rotated_points = copy(surface.points)
						rotated_points.rotate(-1)
						for p1, p2 in zip(surface.points, rotated_points):
							# find where the edge and the projection intersects
							edge = p2 - p1
							print(p1)
							print(p2)
							print(("edge = ", str(edge)))
							print(("projection = ", str(projection)))
							# find the coefficients that allows one dimension to match up
							coefficient = 0
							# FIXME need to make sure edge is not parallel to projection
							# FIXME although, if p1 == source...
							if edge.x != 0:
								print("using xy for coefficient")
								print(projection.y - (edge.y / edge.x) * projection.x)
								coefficient = (p1.y + (edge.y / edge.x) * (p1.x - source.x) - source.y) / (projection.y - (edge.y / edge.x) * projection.x)
							if edge.y != 0:
								print("using yz for coefficient")
								print(projection.z - (edge.z / edge.y) * projection.y)
								coefficient = (p1.z + (edge.z / edge.y) * (p1.y - source.y) - source.z) / (projection.z - (edge.z / edge.y) * projection.y)
							else:
								print("using zx for coefficient")
								print(projection.x - (edge.x / edge.z) * projection.z)
								coefficient = (p1.x + (edge.x / edge.z) * (p1.z + source.z) - source.x) / (projection.x - (edge.x / edge.z) * projection.z)
							# check that the intersection is within the edge
							print(("coefficient = ", coefficient))
							print(p1 + coefficient * edge)
							print(source + coefficient * projection)
							print()
							if (p1 + coefficient * edge) != (source + coefficient * projection):
								continue
							intersection = source + coefficient * projection
							print(("intersection = ", str(intersection)))
							# add to the distance
							distance += (intersection - p1).length()
							print(("added distance = ", (intersection - p1).length()))
							# find the next wall
							candidates = [s for s in self.wall.surfaces if s != surface and p1 in s.points and p2 in s.points]
							print(len(candidates))
							assert len(candidates) == 1
							surface = candidates[0]
							print(list(str(p) for p in surface.points))
							source = intersection
							# move on to the next wall
							print()
							break
						else:
							# this is the wall with the ending hold
							distance += (h2.real_coords() - source).length()
							distances[(i, j)] = distance
							done = True
		print(distances)



if __name__ == "__main__":
	simple_wall = ((
					(   0, 180, 0), # 0
					(   0,   0, 0),
					( 180,   0, 0),
					( 180, 180, 0),
					( 180,   0, 180),
					( 180, 180, 180), # 5
					( 360,   0, 180),
					( 360, 180, 180),
			), (
					(0, 1, 2, 3),
					(3, 2, 4, 5),
					(5, 4, 6, 7),
			))
	simple_prob = (
			simple_wall,
			(
				(
					(0,  90,  90),
					(1,  90,  90),
					(2,  90,  90),
				),
				[],
				[],
			))
	wall = Wall(*simple_wall)
	prob = Problem(wall)
	for hold in simple_prob[1][0]:
		prob.add_hold(*hold)
	viewer = IsometricViewer(800, 600)
	viewer.add_drawable(prob)
	prob.create_graph()
#	viewer.display()

	"""
	problem = CUSTOM2_PROB_3
	prob = Problem(Wall(*problem[0]))
	for hold in problem[1][0]:
		prob.add_hold(*hold)
	for hold in problem[1][1]:
		prob.add_start_hold(hold)
	for hold in problem[1][2]:
		prob.add_finish_hold(hold)
	viewer = IsometricViewer(800, 600)
	viewer.add_drawable(prob)
	viewer.display()
	"""
