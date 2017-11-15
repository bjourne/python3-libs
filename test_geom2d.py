# Copyright (C) 2017 Björn Lindqvist
from geom2d import LineSegment, Point, \
    colinear_intersect, line_segment, orientation, \
    segments_intersect2, \
    segments_intersect_fast, segments_intersect_naive

def poly_to_segments(poly):
    segments = list(zip(poly, poly[1:])) + [(poly[-1], poly[0])]
    segments = [line_segment(*d) for d in segments]
    return segments

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
    assert segments_intersect2(ls1, ls2)
    ls3 = LineSegment(Point(0, 0), Point(10, 0))
    assert segments_intersect2(ls1, ls3)
    ls4 = LineSegment(Point(1, 6), Point(6, 1))
    ls5 = LineSegment(Point(2, 3), Point(9, 4))
    assert segments_intersect2(ls4, ls5)
    ls6 = LineSegment(Point(1, 1), Point(1, 4))
    ls7 = LineSegment(Point(1, 4), Point(1, 3))
    assert segments_intersect2(ls6, ls7)

def test_colinear_intersect():
    ls = LineSegment(Point(0, 0), Point(10, 0))
    p1 = Point(5, 0)
    p2 = Point(0, 0)
    assert colinear_intersect(ls, p1)
    assert not colinear_intersect(ls, p2)

def test_horiz_vert_lines():
    ls1 = LineSegment(Point(0, 0), Point(10, 0))
    ls2 = LineSegment(Point(5, 0), Point(15, 0))
    assert segments_intersect2(ls1, ls2)
    ls1 = LineSegment(Point(0, 0), Point(0, 10))
    ls2 = LineSegment(Point(0, 5), Point(0, 15))
    assert segments_intersect2(ls1, ls2)

def test_does_not_intersect():
    ls1 = LineSegment(Point(-10, 0), Point(10, 0))
    ls2 = LineSegment(Point(10, 0), Point(20, 0))
    assert not segments_intersect2(ls1, ls2)
    ls3 = LineSegment(Point(1, 6), Point(5, 7))
    ls4 = LineSegment(Point(5, 7), Point(9, 4))
    assert not segments_intersect2(ls3, ls4)

def test_segments_intersect():
    poly = [(1, 6), (5, 7), (9, 4), (2, 3), (6, 1)]
    segments = poly_to_segments(poly)
    assert segments_intersect_naive(segments)
    assert segments_intersect_fast(segments)

def test_segments_no_intersect():
    poly = [(1, 6), (5, 7), (9, 4), (4, 3), (7, 4), (4, 6), (3, 1)]
    segments = poly_to_segments(poly)
    assert not segments_intersect_naive(segments)
    assert not segments_intersect_fast(segments)

def test_segments_horrible_intersect():
    poly = [(1, 1), (1, 4), (1, 3), (2, 2), (3, 1), (3, 2), (2, 2)]
    segments = poly_to_segments(poly)
    print(segments)
    assert segments_intersect_naive(segments)
    assert segments_intersect_fast(segments)

def test_segments_horizontal_overlap():
    ls1 = LineSegment(Point(0, 0), Point(10, 0))
    ls2 = LineSegment(Point(5, 0), Point(15, 0))
    segments = [ls1, ls2]
    assert segments_intersect_fast(segments)

def test_segments_vertical_overlap():
    ls1 = LineSegment(Point(0, 0), Point(0, 10))
    ls2 = LineSegment(Point(0, 5), Point(0, 15))
    segments = [ls1, ls2]
    assert segments_intersect_naive(segments)
    assert segments_intersect_fast(segments)
    ls1 = LineSegment(Point(1, 1), Point(1, 4))
    ls2 = LineSegment(Point(1, 4), Point(1, 3))
    segments = [ls1, ls2]
    assert segments_intersect_naive(segments)
    assert segments_intersect_fast(segments)
