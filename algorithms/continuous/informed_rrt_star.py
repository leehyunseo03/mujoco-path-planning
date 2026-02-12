import numpy as np
import math
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
    
    rotation_matrix = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])
    
    r1 = c_max / 2.0
    r2 = math.sqrt(max(0, c_max**2 - c_min**2)) / 2.0
    
    diag_matrix = np.array([
        [r1, 0],
        [0, r2]
    ])
    
    r = math.sqrt(np.random.random())
    phi = np.random.random() * 2 * math.pi
    x_ball = np.array([r * math.cos(phi), r * math.sin(phi)])
    
    x_rand = np.dot(rotation_matrix, np.dot(diag_matrix, x_ball)) + x_center
    
    return x_rand[0], x_rand[1]

def informed_rrt_star_step_generator(start_pos, goal_pos, map_dim, obstacles, 
                                     step_size=2.0, goal_threshold=3.0, search_radius=15.0, max_iter=4000):
    
    cc = CollisionChecker(obstacles, map_dim)
    
    start_node = Node(start_pos[0], start_pos[1])
    goal_node = Node(goal_pos[0], goal_pos[1])
    node_list = [start_node]
    
    c_best = float('inf')
    
    yield "sampling", (start_node.x, start_node.y)

    for i in range(max_iter):
        
        if c_best < float('inf'):
            if np.random.rand() < 0.05:
                 rnd_x, rnd_y = goal_node.x, goal_node.y
            else:
                rnd_x, rnd_y = get_informed_sample(start_node, goal_node, c_best, map_dim)
        else:
            if np.random.rand() < 0.05:
                rnd_x, rnd_y = goal_node.x, goal_node.y
            else:
                rnd_x = np.random.uniform(0, map_dim)
                rnd_y = np.random.uniform(0, map_dim)
            
        if not (0 <= rnd_x <= map_dim and 0 <= rnd_y <= map_dim):
            continue
            
        nearest_node = node_list[0]
        min_dist_sq = (nearest_node.x - rnd_x)**2 + (nearest_node.y - rnd_y)**2
        
        for node in node_list:
            d_sq = (node.x - rnd_x)**2 + (node.y - rnd_y)**2
            if d_sq < min_dist_sq:
                nearest_node = node
                min_dist_sq = d_sq
        
        theta = math.atan2(rnd_y - nearest_node.y, rnd_x - nearest_node.x)
        new_x = nearest_node.x + step_size * math.cos(theta)
        new_y = nearest_node.y + step_size * math.sin(theta)
        
        if cc.is_line_colliding(nearest_node.x, nearest_node.y, new_x, new_y):
            continue
            
        new_node = Node(new_x, new_y)
        
        neighbors = []
        for node in node_list:
            if get_dist(node, new_node) <= search_radius:
                neighbors.append(node)
        
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + get_dist(nearest_node, new_node)
        
        for neighbor in neighbors:
            tentative_cost = neighbor.cost + get_dist(neighbor, new_node)
            if tentative_cost < new_node.cost:
                if not cc.is_line_colliding(neighbor.x, neighbor.y, new_node.x, new_node.y):
                    new_node.parent = neighbor
                    new_node.cost = tentative_cost
        
        node_list.append(new_node)
        yield "sampling", (new_node.x, new_node.y)
        yield "edge", ((new_node.parent.x, new_node.parent.y), (new_node.x, new_node.y))
        
        for neighbor in neighbors:
            tentative_cost = new_node.cost + get_dist(new_node, neighbor)
            if tentative_cost < neighbor.cost:
                if not cc.is_line_colliding(new_node.x, new_node.y, neighbor.x, neighbor.y):
                    neighbor.parent = new_node
                    neighbor.cost = tentative_cost
                    yield "edge", ((new_node.x, new_node.y), (neighbor.x, neighbor.y))

        dist_to_goal = get_dist(new_node, goal_node)
        if dist_to_goal <= goal_threshold:
            if not cc.is_line_colliding(new_node.x, new_node.y, goal_node.x, goal_node.y):
                
                current_path_cost = new_node.cost + dist_to_goal
                
                if current_path_cost < c_best:
                    c_best = current_path_cost
                
                final_path = []
                curr = new_node
                while curr is not None:
                    final_path.append((curr.x, curr.y))
                    curr = curr.parent
                final_path = final_path[::-1] 
                final_path.append((goal_node.x, goal_node.y))
                
                yield "path", final_path