# Copyright (C) 2017 Björn Lindqvist
from bisect import bisect_left, insort
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

class LineSegment:
    def __init__(self, p1, p2):
        if p2.x < p1.x:
            p1, p2 = p2, p1
        elif p1.x == p2.x and p2.y < p1.y:
            p1, p2 = p2, p1
        self.p1 = p1
        self.p2 = p2

    def __eq__(self, o):
        return self.p1 == o.p1 and self.p2 == o.p2

    def __repr__(self):
        return "LineSegment(%s,%s)" % (self.p1, self.p2)

def line_segment(p1, p2):
    return LineSegment(Point(*p1), Point(*p2))

def orientation(seg, r):
    p = seg.p1
    q = seg.p2
    d = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    return 1 if d > 0 else (-1 if d < 0 else 0)

def colinear_intersect(seg, p):
    return p != seg.p1 and p != seg.p2 and \
        p.x <= seg.p2.x and \
        p.x >= seg.p1.x and \
        p.y <= max(seg.p1.y, seg.p2.y) and \
        p.y >= min(seg.p1.y, seg.p2.y)

def segments_intersect2(seg1, seg2):
    '''Determine whether two open line segments intersect.'''
    p1, p2, p3, p4 = seg1.p1, seg1.p2, seg2.p1, seg2.p2

    o1 = orientation(seg1, p3)
    o2 = orientation(seg1, p4)
    o3 = orientation(seg2, p1)
    o4 = orientation(seg2, p2)
    if o1 == 0:
        return colinear_intersect(seg1, p3)
    if o2 == 0:
        return colinear_intersect(seg1, p4)
    if o3 == 0:
        return colinear_intersect(seg2, p1)
    if o4 == 0:
        return colinear_intersect(seg2, p2)
    return o1 != o2 and o3 != o4

def segments_intersect_naive(segments):
    '''Naive method to determine if a sequence of line segments has any
    intersections.'''
    for seg1 in segments:
        for seg2 in segments:
            if seg1 != seg2 and segments_intersect2(seg1, seg2):
                return True
    return False

# Borrowed/stolen code from:
class search:
    def __init__(self):
        self.values = []
    def insert(self, value):
        assert not value in self.values
        insort(self.values, value)

    def delete(self, value):
        assert value in self.values
        assert(self.values.pop(self.position(value)) == value)

    def find_neighbors(self, value):
        p = self.position(value)
        l = None
        r = None
        if p > 0: l = self.values[p-1]
        if p < len(self.values)-1: r = self.values[p+1]
        return (l,r)

    def position(self, value):
        i = bisect_left(self.values, value)
        if i != len(self.values) and self.values[i] == value:
            return i
        raise ValueError

from itertools import groupby

def segments_intersect_fast(segments):
    def x1_key(seg):
        return seg.p1.x
    def y1_key(seg):
        return seg.p1.y

    # Special case for handling horizontal segments.
    horiz = [seg for seg in segments if seg.p1.y == seg.p2.y]
    for y, group in groupby(sorted(horiz, key = y1_key), key = y1_key):
        group = sorted(group, key = x1_key)
        scan = None
        for seg in group:
            if scan is not None and seg.p1.x < scan:
                return True
            scan = seg.p2.x

    # Same kind of special casing for vertial segments.
    vert = [seg for seg in segments if seg.p1.x == seg.p2.x]
    for x, group in groupby(sorted(vert, key = x1_key), key = x1_key):
        group = sorted(group, key = y1_key)
        scan = None
        for seg in group:
            if scan is not None and seg.p1.y < scan:
                return True
            scan = seg.p2.y

    # No handling for arbitrary colinear segments. They look way
    # harder to handle.
    end_points = []
    for i, seg in enumerate(segments):
        x1, y1, x2, y2 = seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y
        is_right = x1 >= x2
        end_points.append((x1, i, is_right))
        end_points.append((x2, i, not is_right))
    end_points = sorted(end_points)

    tree = search()
    for _, label, is_right in end_points:
        seg = segments[label]
        print(seg, is_right, tree.values)
        if not is_right:
            tree.insert(label)
            for n in tree.find_neighbors(label):
                if n is not None:
                    if segments_intersect2(seg, segments[n]):
                        return True
        else:
            p, s = tree.find_neighbors(label)
            if p is not None and s is not None:
                pred = segments[p]
                succ = segments[s]
                if segments_intersect2(pred, succ):
                    return True
            tree.delete(label)
    return False
