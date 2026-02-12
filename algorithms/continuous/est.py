import numpy as np
import math
from .utils import Node, get_dist, CollisionChecker

def est_step_generator(start_pos, goal_pos, map_dim, obstacles, 
                       step_size=3.0, goal_threshold=3.0, density_radius=5.0, max_iter=5000):
    
    cc = CollisionChecker(obstacles, map_dim)
    
    start_node = Node(start_pos[0], start_pos[1])
    goal_node = Node(goal_pos[0], goal_pos[1])
    node_list = [start_node]
    
    yield "sampling", (start_node.x, start_node.y)

    for i in range(max_iter):
        
        selected_node = None
        
        for _ in range(10):
            rand_idx = np.random.randint(len(node_list))
            candidate = node_list[rand_idx]
            
            neighbors = 0
            for other in node_list:
                dist_sq = (candidate.x - other.x)**2 + (candidate.y - other.y)**2
                if dist_sq < density_radius**2:
                    neighbors += 1
            
            prob = 1.0 / (neighbors + 1)
            
            if np.random.rand() < prob:
                selected_node = candidate
                break
        
        if selected_node is None:
            selected_node = node_list[np.random.randint(len(node_list))]

        if np.random.rand() < 0.1:
            angle = math.atan2(goal_node.y - selected_node.y, goal_node.x - selected_node.x)
        else:
            angle = np.random.uniform(0, 2 * math.pi)
        
        current_step = np.random.uniform(step_size * 0.5, step_size)
        
        new_x = selected_node.x + current_step * math.cos(angle)
        new_y = selected_node.y + current_step * math.sin(angle)
        
        if not (0 <= new_x <= map_dim and 0 <= new_y <= map_dim):
            continue

        if cc.is_line_colliding(selected_node.x, selected_node.y, new_x, new_y):
            continue
            
        new_node = Node(new_x, new_y)
        new_node.parent = selected_node
        node_list.append(new_node)
        
        yield "sampling", (new_node.x, new_node.y)
        yield "edge", ((selected_node.x, selected_node.y), (new_node.x, new_node.y))
        
        dist_to_goal = math.hypot(new_node.x - goal_node.x, new_node.y - goal_node.y)
        
        if dist_to_goal <= goal_threshold:
            if not cc.is_line_colliding(new_node.x, new_node.y, goal_node.x, goal_node.y):
                
                yield "edge", ((new_node.x, new_node.y), (goal_node.x, goal_node.y))
                
                final_path = []
                final_path.append((goal_node.x, goal_node.y))
                curr = new_node
                while curr is not None:
                    final_path.append((curr.x, curr.y))
                    curr = curr.parent
                final_path = final_path[::-1] 
                
                yield "path", final_path
                return