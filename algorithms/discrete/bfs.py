from collections import deque
from .utils import get_neighbors, reconstruct_path

def bfs_step_generator(grid, start, goal):
    queue = deque([start])
    visited = {start}
    came_from = {start: None}
    
    while queue:
        current = queue.popleft()
        yield "visiting", current
        
        if current == goal:
            break
        
        for neighbor in get_neighbors(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
    
    if goal in came_from:
        yield "path", reconstruct_path(came_from, start, goal)