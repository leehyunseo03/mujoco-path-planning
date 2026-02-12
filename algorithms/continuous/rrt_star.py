import numpy as np
import math
from .utils import Node, get_dist, CollisionChecker

def rrt_star_step_generator(start_pos, goal_pos, map_dim, obstacles, 
                            step_size=2.0, goal_threshold=3.0, search_radius=10.0, max_iter=3000):
    
    cc = CollisionChecker(obstacles, map_dim)
    
    start_node = Node(start_pos[0], start_pos[1])
    goal_node = Node(goal_pos[0], goal_pos[1])
    node_list = [start_node]
    
    yield "sampling", (start_node.x, start_node.y)

    for i in range(max_iter):
        if np.random.rand() < 0.1:
            rnd_x, rnd_y = goal_node.x, goal_node.y
        else:
            rnd_x = np.random.uniform(0, map_dim)
            rnd_y = np.random.uniform(0, map_dim)
        
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
                final_path = []
                curr = new_node
                while curr is not None:
                    final_path.append((curr.x, curr.y))
                    curr = curr.parent
                
                final_path = final_path[::-1] 
                final_path.append((goal_node.x, goal_node.y))
                
                yield "path", final_path
                return