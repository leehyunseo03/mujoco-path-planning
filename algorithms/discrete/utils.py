import math

def get_neighbors(grid, node):
    rows, cols = grid.shape
    x, y = node
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in directions:
        nx, ny = int(x + dx), int(y + dy)
        
        if 0 <= nx < rows and 0 <= ny < cols and not grid[nx, ny]:
            yield (nx, ny)

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return []
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    return path[::-1]

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])