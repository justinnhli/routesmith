#!/usr/bin/env python3

import math # FIXME should use numpy

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
		#return "wall.push(new Line(new Point{}, new Point{}));".format(self.p1, self.p2)
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

class Wall:
	def __init__(self, points, surfaces):
		self.points = points
		self.surfaces = surfaces
		center = Point(
				(max(p.x for p in self.points) + min(p.x for p in self.points)) / 2,
				(max(p.y for p in self.points) + min(p.y for p in self.points)) / 2,
				(max(p.z for p in self.points) + min(p.z for p in self.points)) / 2)
		self.points = tuple(p - center for p in self.points)

CUSTOM = Wall((
				Point(  0,  80,   0), # 0
				Point(  0,   0,   0),
				Point( 30,  80,  46),
				Point( 30,  20,   0),
				Point( 60,  80,   0),
				Point( 70,  80,   0), # 5
				Point(116,  80,  46),
				Point(116,  20,   0),
				Point(200,  80,  46),
				Point(200,  20,   0),
				Point(200,   0,   0), # 10
				Point(246,  20,  46),
				Point(246,   0,  46),
				Point(200,  80, 100),
				Point(246,  20, 100),
				Point(246,   0, 100), # 15
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
	surfaces = [Surface(CUSTOM.points[i] for i in surface) for surface in CUSTOM.surfaces]
	lines = set()
	for surface in surfaces:
		lines = lines.union(surface.get_lines())
	for line in lines:
		print(line)
