import numpy as np
import math
import heapq
from .utils import Node, get_dist, CollisionChecker

def a_star_search(start_node, goal_node):
    open_set = []
    heapq.heappush(open_set, (0, start_node.id, start_node))
    
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: get_dist(start_node, goal_node)}
    
    visited_ids = set()
    
    while open_set:
        current_f, _, current = heapq.heappop(open_set)
        
        if current.id in visited_ids:
            continue
        visited_ids.add(current.id)
        
        if current == goal_node:
            path = []
            while current in came_from:
                path.append((current.x, current.y))
                current = came_from[current]
            path.append((start_node.x, start_node.y))
            return path[::-1]

        for neighbor, cost in current.edges:
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + get_dist(neighbor, goal_node)
                f_score[neighbor] = f
                heapq.heappush(open_set, (f, neighbor.id, neighbor))
                
    return None

# --- PRM 메인 제너레이터 ---
def prm_step_generator(start_pos, goal_pos, map_dim, obstacles, 
                       num_samples=300, connection_radius=15.0, k_neighbors=10):
    
    cc = CollisionChecker(obstacles, map_dim)
    
    nodes = []
    
    # 1. 시작점/목표점 설정 및 초기화
    start_node = Node(start_pos[0], start_pos[1])
    start_node.id = 0
    start_node.edges = []
    
    goal_node = Node(goal_pos[0], goal_pos[1])
    goal_node.id = 1
    goal_node.edges = []
    
    nodes.append(start_node)
    nodes.append(goal_node)
    
    yield "sampling", (start_node.x, start_node.y)
    yield "sampling", (goal_node.x, goal_node.y)

    valid_samples = 0
    while valid_samples < num_samples:
        rx = np.random.uniform(0, map_dim)
        ry = np.random.uniform(0, map_dim)
        
        if not cc.is_point_colliding(rx, ry):
            new_node = Node(rx, ry)
            new_node.id = len(nodes)
            new_node.edges = [] # 명시적 초기화
            nodes.append(new_node)
            valid_samples += 1
            yield "sampling", (rx, ry)

    for i in range(len(nodes)):
        n1 = nodes[i]
        candidates = []
        
        for j in range(len(nodes)):
            if i == j: continue
            n2 = nodes[j]
            d = get_dist(n1, n2)
            
            if d <= connection_radius:
                candidates.append((d, n2))
        
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:k_neighbors]

        for dist, n2 in candidates:
            already_connected = False
            for edge in n1.edges:
                if edge[0] == n2:
                    already_connected = True
                    break
            if already_connected:
                continue

            if not cc.is_line_colliding(n1.x, n1.y, n2.x, n2.y):
                n1.edges.append((n2, dist))
                n2.edges.append((n1, dist))
                yield "edge", ((n1.x, n1.y), (n2.x, n2.y))

    # 4. 그래프 탐색 (Query Phase)
    print("🔹 PRM: 로드맵 구축 완료. 최단 경로 탐색 시작...")
    final_path = a_star_search(start_node, goal_node)
    
    if final_path:
        print(f"🎉 경로 발견! (길이: {len(final_path)})")
        yield "path", final_path
    else:
        print("⚠️ 경로를 찾을 수 없습니다. (로드맵이 끊겨있음)")