# algorithms/continuous/utils.py
import math

class Node:
    """모든 RRT 계열 알고리즘에서 사용할 공통 노드"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0 

    def __repr__(self):
        return f"Node({self.x:.2f}, {self.y:.2f})"

def get_dist(n1, n2):
    x1 = n1.x if hasattr(n1, 'x') else n1[0]
    y1 = n1.y if hasattr(n1, 'y') else n1[1]
    x2 = n2.x if hasattr(n2, 'x') else n2[0]
    y2 = n2.y if hasattr(n2, 'y') else n2[1]
    return math.hypot(x1 - x2, y1 - y2)

class CollisionChecker:
    def __init__(self, obstacles, map_dim, resolution=0.3, margin=0.5, probe_radius=0.35):
        self.obstacles = obstacles
        self.map_dim = map_dim
        self.resolution = resolution
        self.margin = margin
        self.probe_radius = probe_radius

    def is_point_colliding(self, x, y):
        if not (self.margin <= x <= self.map_dim - self.margin and 
                self.margin <= y <= self.map_dim - self.margin):
            return True 
        
        for (cx, cy, sx, sy) in self.obstacles:
            if (abs(x - cx) < (sx + self.probe_radius)) and \
               (abs(y - cy) < (sy + self.probe_radius)):
                return True
        return False

    def is_line_colliding(self, x1, y1, x2, y2):
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < self.resolution:
            return self.is_point_colliding(x2, y2)

        steps = int(dist / self.resolution) + 1
        for i in range(steps + 1):
            t = i / steps
            ix = x1 + (x2 - x1) * t
            iy = y1 + (y2 - y1) * t
            if self.is_point_colliding(ix, iy):
                return True
        return False