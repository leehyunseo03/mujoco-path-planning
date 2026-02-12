import numpy as np
import math
import heapq
from .utils import Node, get_dist, CollisionChecker

def fmt_star_step_generator(start_pos, goal_pos, map_dim, obstacles, 
                            num_samples=600, radius_multiplier=1.5):
    
    cc = CollisionChecker(obstacles, map_dim)
    
    start_node = Node(start_pos[0], start_pos[1])
    start_node.cost = 0.0
    
    goal_node = Node(goal_pos[0], goal_pos[1])
    
    samples = [start_node, goal_node]
    
    yield "sampling", (start_node.x, start_node.y)
    yield "sampling", (goal_node.x, goal_node.y)

    for _ in range(num_samples):
        rx = np.random.uniform(0, map_dim)
        ry = np.random.uniform(0, map_dim)
        if not cc.is_point_colliding(rx, ry):
            new_node = Node(rx, ry)
            samples.append(new_node)
            yield "sampling", (rx, ry)
            
    unvisited_set = set(samples)
    unvisited_set.remove(start_node)
    
    open_set = {start_node}
    
    open_list = [(start_node.cost, id(start_node), start_node)]
    
    unit_ball_volume = math.pi
    gamma = 2.0 * ((1.0 / unit_ball_volume) ** 0.5) 
    search_radius = radius_multiplier * gamma * (math.log(len(samples)) / len(samples)) ** 0.5 * map_dim
    search_radius = max(search_radius, 4.0)
    
    while open_list:
        current_cost, _, z = heapq.heappop(open_list)
        
        if z not in open_set:
            continue
            
        neighbors_in_unvisited = []
        for x in unvisited_set:
            if get_dist(z, x) <= search_radius:
                neighbors_in_unvisited.append(x)
        
        for x in neighbors_in_unvisited:
            y_min = None
            min_cost = float('inf')
            
            candidate_parents = []
            for y in open_set:
                if get_dist(y, x) <= search_radius:
                    candidate_parents.append(y)
            
            for y in candidate_parents:
                temp_cost = y.cost + get_dist(y, x)
                if temp_cost < min_cost:
                    if not cc.is_line_colliding(y.x, y.y, x.x, x.y):
                        min_cost = temp_cost
                        y_min = y
            
            if y_min:
                x.parent = y_min
                x.cost = min_cost
                
                if x in unvisited_set:
                    unvisited_set.remove(x)
                
                open_set.add(x)
                heapq.heappush(open_list, (x.cost, id(x), x))
                
                yield "edge", ((y_min.x, y_min.y), (x.x, x.y))
                
                dist_to_goal = get_dist(x, goal_node)
                if dist_to_goal < 1.0: 
                    path = []
                    curr = x
                    while curr:
                        path.append((curr.x, curr.y))
                        curr = curr.parent
                    yield "path", path[::-1]
                    return

        if z in open_set:
            open_set.remove(z)