# Copyright (C) 2017 Björn Lindqvist
from geom2d import LineSegment, Point, \
    colinear_intersect, line_segments_intersect, orientation

def test_line_segment():
    ls = LineSegment(Point(0, 3), Point(10, 0))
    assert ls.p1.x == 0
    assert ls.p1.y == 3

def test_orientation():
    ls = LineSegment(Point(0, 0), Point(10, 0))
    p = Point(3, 0)
    assert not orientation(ls, p)

def test_does_intersect():
    ls1 = LineSegment(Point(-10, 0), Point(10, 0))
    ls2 = LineSegment(Point(0, -10), Point(0, 10))
    assert line_segments_intersect(ls1, ls2)
    ls3 = LineSegment(Point(0, 0), Point(10, 0))
    assert line_segments_intersect(ls1, ls3)

def test_colinear_intersect():
    ls = LineSegment(Point(0, 0), Point(10, 0))
    p1 = Point(5, 0)
    p2 = Point(0, 0)
    assert colinear_intersect(ls, p1)
    assert not colinear_intersect(ls, p2)

def test_does_not_intersect():
    ls1 = LineSegment(Point(-10, 0), Point(10, 0))
    ls2 = LineSegment(Point(10, 0), Point(20, 0))
    assert not line_segments_intersect(ls1, ls2)
