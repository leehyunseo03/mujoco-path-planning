import numpy as np
import math
import heapq
from .utils import Node, get_dist, CollisionChecker

def get_informed_sample(start_node, goal_node, c_max, map_dim):
    if c_max == float('inf'):
        return np.random.uniform(0, map_dim), np.random.uniform(0, map_dim)

    c_min = get_dist(start_node, goal_node)
    
    x_center = np.array([(start_node.x + goal_node.x) / 2.0, 
                         (start_node.y + goal_node.y) / 2.0])
    
    dy = goal_node.y - start_node.y
    dx = goal_node.x - start_node.x
    theta = math.atan2(dy, dx)
    
    rot = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])
    
    r1 = c_max / 2.0
    r2 = math.sqrt(max(0, c_max**2 - c_min**2)) / 2.0
    diag = np.array([[r1, 0], [0, r2]])
    
    r = math.sqrt(np.random.random())
    phi = np.random.random() * 2 * math.pi
    ball = np.array([r * math.cos(phi), r * math.sin(phi)])
    
    res = np.dot(rot, np.dot(diag, ball)) + x_center
    return res[0], res[1]

def bit_star_step_generator(start_pos, goal_pos, map_dim, obstacles, 
                            batch_size=80, eta=1.5, max_iter=5000):
    
    cc = CollisionChecker(obstacles, map_dim)
    
    start_node = Node(start_pos[0], start_pos[1])
    goal_node = Node(goal_pos[0], goal_pos[1])
    
    tree_nodes = [start_node]
    unconnected_samples = [goal_node] 
    
    edge_queue = []
    
    c_best = float('inf') 
    
    yield "sampling", (start_node.x, start_node.y)
    yield "sampling", (goal_node.x, goal_node.y)

    q = 1 

    for i in range(max_iter):
        
        if not edge_queue:
            
            new_samples = []
            for _ in range(batch_size):
                rx, ry = get_informed_sample(start_node, goal_node, c_best, map_dim)
                if 0 <= rx <= map_dim and 0 <= ry <= map_dim:
                    if not cc.is_point_colliding(rx, ry):
                        new_node = Node(rx, ry)
                        new_samples.append(new_node)
                        yield "sampling", (rx, ry)
            
            unconnected_samples.extend(new_samples)
            q += len(new_samples)
            
            curr_r = eta * math.sqrt((math.log(q) / q)) * map_dim
            curr_r = min(curr_r, 20.0)
            
            for v in tree_nodes:
                for x in unconnected_samples:
                    dist = get_dist(v, x)
                    if dist <= curr_r:
                        heuristic = get_dist(x, goal_node)
                        estimated_f = v.cost + dist + heuristic
                        
                        if estimated_f < c_best:
                            heapq.heappush(edge_queue, (estimated_f, v.cost + dist, id(v), id(x), v, x))
                            
        if not edge_queue:
            continue

        estimated_f, new_g, _, _, parent, child = heapq.heappop(edge_queue)
        
        if estimated_f >= c_best:
            continue
            
        if child.parent is None or new_g < child.cost:
             if not cc.is_line_colliding(parent.x, parent.y, child.x, child.y):
                
                child.parent = parent
                child.cost = new_g
                
                if child not in tree_nodes:
                    tree_nodes.append(child)
                
                if child in unconnected_samples:
                    unconnected_samples.remove(child)
                
                yield "edge", ((parent.x, parent.y), (child.x, child.y))
                
                r_threshold = 20.0
                
                for x in unconnected_samples:
                    dist = get_dist(child, x)
                    if dist <= r_threshold:
                        heuristic = get_dist(x, goal_node)
                        est_f = child.cost + dist + heuristic
                        if est_f < c_best:
                            heapq.heappush(edge_queue, (est_f, child.cost + dist, id(child), id(x), child, x))
                            
                dist_to_goal = get_dist(child, goal_node)
                if dist_to_goal < 1.0 or child == goal_node:
                    if child.cost < c_best:
                        c_best = child.cost
                        
                        path = []
                        curr = child
                        while curr:
                            path.append((curr.x, curr.y))
                            curr = curr.parent
                        
                        yield "path", path[::-1]
                        
                        new_queue = []
                        for item in edge_queue:
                            if item[0] < c_best:
                                new_queue.append(item)
                        heapq.heapify(new_queue)
                        edge_queue = new_queue