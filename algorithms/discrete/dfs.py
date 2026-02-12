from .utils import get_neighbors, reconstruct_path

def dfs_step_generator(grid, start, goal):
    stack = [start]
    visited = {start}
    came_from = {start: None}

    while stack:
        current = stack.pop()
        yield "visiting", current
        
        if current == goal:
            break
        
        for neighbor in get_neighbors(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
    
    if goal in came_from:
        yield "path", reconstruct_path(came_from, start, goal)