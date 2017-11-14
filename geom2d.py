# Copyright (C) 2017 Björn Lindqvist
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
LineSegment = namedtuple('LineSegment', ['p1', 'p2'])

def orientation(p, q, r):
    d = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    return 1 if d > 0 else (-1 if d < 0 else 0)

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

def line_segments_intersect(seg1, seg2):
    (p1, p2), (p3, p4) = seg1, seg2
    o1 = orientation(seg1, p3)
    o2 = orientation(seg1, p4)
    o3 = orientation(seg2, p1)
    o4 = orientation(seg2, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and colinear_intersect(seg1, p3):
        return True
    if o2 == 0 and colinear_intersect(seg1, p4):
        return True
    if o3 == 0 and colinear_intersect(seg2, p1):
        return True
    if o4 == 0 and colinear_intersect(seg2, p2):
        return True
    return False
