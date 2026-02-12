import numpy as np
import math
from .utils import Node, get_dist, CollisionChecker

def rrt_connect_step_generator(start_pos, goal_pos, map_dim, obstacles, 
                               step_size=2.0, max_iter=5000):
    
    cc = CollisionChecker(obstacles, map_dim)
    
    start_node = Node(start_pos[0], start_pos[1])
    goal_node = Node(goal_pos[0], goal_pos[1])
    
    nodes_start = [start_node]
    nodes_goal = [goal_node]
    
    yield "sampling", (start_node.x, start_node.y)
    yield "sampling", (goal_node.x, goal_node.y)

    for i in range(max_iter):
        if len(nodes_start) < len(nodes_goal):
            tree_a = nodes_start
            tree_b = nodes_goal
            is_forward = True 
        else:
            tree_a = nodes_goal
            tree_b = nodes_start
            is_forward = False

        rnd_x = np.random.uniform(0, map_dim)
        rnd_y = np.random.uniform(0, map_dim)
        
        nearest_a = min(tree_a, key=lambda n: (n.x - rnd_x)**2 + (n.y - rnd_y)**2)
        
        theta = math.atan2(rnd_y - nearest_a.y, rnd_x - nearest_a.x)
        new_a_x = nearest_a.x + step_size * math.cos(theta)
        new_a_y = nearest_a.y + step_size * math.sin(theta)
        
        if cc.is_line_colliding(nearest_a.x, nearest_a.y, new_a_x, new_a_y):
            continue
            
        new_node_a = Node(new_a_x, new_a_y)
        new_node_a.parent = nearest_a
        tree_a.append(new_node_a)
        
        yield "sampling", (new_node_a.x, new_node_a.y)
        yield "edge", ((nearest_a.x, nearest_a.y), (new_node_a.x, new_node_a.y))
        
        nearest_b = min(tree_b, key=lambda n: (n.x - new_node_a.x)**2 + (n.y - new_node_a.y)**2)
        dist = math.hypot(new_node_a.x - nearest_b.x, new_node_a.y - nearest_b.y)
        
        if dist <= step_size:
            connect_x, connect_y = new_node_a.x, new_node_a.y
        else:
            theta = math.atan2(new_node_a.y - nearest_b.y, new_node_a.x - nearest_b.x)
            connect_x = nearest_b.x + step_size * math.cos(theta)
            connect_y = nearest_b.y + step_size * math.sin(theta)
            
        if not cc.is_line_colliding(nearest_b.x, nearest_b.y, connect_x, connect_y):
            new_node_b = Node(connect_x, connect_y)
            new_node_b.parent = nearest_b
            tree_b.append(new_node_b)
            
            yield "sampling", (new_node_b.x, new_node_b.y)
            yield "edge", ((nearest_b.x, nearest_b.y), (new_node_b.x, new_node_b.y))
            
            final_dist = get_dist(new_node_a, new_node_b)
            
            if final_dist <= step_size:
                if not cc.is_line_colliding(new_node_a.x, new_node_a.y, new_node_b.x, new_node_b.y):
                    yield "edge", ((new_node_a.x, new_node_a.y), (new_node_b.x, new_node_b.y))
                    
                    path_a = []
                    curr = new_node_a
                    while curr is not None:
                        path_a.append((curr.x, curr.y))
                        curr = curr.parent
                    
                    path_b = []
                    curr = new_node_b
                    while curr is not None:
                        path_b.append((curr.x, curr.y))
                        curr = curr.parent
                    
                    if is_forward:
                        final_path = path_a[::-1] + path_b
                    else:
                        final_path = path_b[::-1] + path_a

                    yield "path", final_path
                    return