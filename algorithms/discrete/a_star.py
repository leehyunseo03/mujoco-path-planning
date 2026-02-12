import heapq
from .utils import get_neighbors, reconstruct_path, heuristic

def a_star_step_generator(grid, start, goal):
    start = tuple(start)
    goal = tuple(goal)
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {start: None}
    g_score = {start: 0}
    
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current != start and current != goal:
            yield "visiting", current

        if current == goal:
            break

        visited.add(current)
        
        for neighbor in get_neighbors(grid, current):
            if neighbor in visited:
                continue
                
            tentative_g_score = g_score[current] + 1
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    if goal in came_from:
        yield "path", reconstruct_path(came_from, start, goal)