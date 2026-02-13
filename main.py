import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from map_manager import MapManager
from record import MujocoRecorder

from algorithms.discrete.bfs import bfs_step_generator
from algorithms.discrete.dfs import dfs_step_generator
from algorithms.discrete.a_star import a_star_step_generator

from algorithms.continuous.rrt import rrt_step_generator
from algorithms.continuous.rrt_star import rrt_star_step_generator
from algorithms.continuous.informed_rrt_star import informed_rrt_star_step_generator
from algorithms.continuous.rrt_connect import rrt_connect_step_generator
from algorithms.continuous.est import est_step_generator
from algorithms.continuous.bit_star import bit_star_step_generator
from algorithms.continuous.fmt_star import fmt_star_step_generator
from algorithms.continuous.prm import prm_step_generator

def get_algorithm(name, m_map):
    name = name.lower()
    
    if name == 'bfs':
        return 'discrete', bfs_step_generator(m_map.grid, m_map.start, m_map.goal)
    elif name == 'dfs':
        return 'discrete', dfs_step_generator(m_map.grid, m_map.start, m_map.goal)
    elif name == 'a_star':
        return 'discrete', a_star_step_generator(m_map.grid, m_map.start, m_map.goal)
    
    elif name == 'rrt':
        return 'continuous', rrt_step_generator(m_map.start, m_map.goal, m_map.dim, m_map.obstacles)
    elif name == 'rrt_star':
        return 'continuous', rrt_star_step_generator(m_map.start, m_map.goal, m_map.dim, m_map.obstacles, search_radius=15.0)
    elif name == 'informed_rrt_star':
        return 'continuous', informed_rrt_star_step_generator(m_map.start, m_map.goal, m_map.dim, m_map.obstacles, search_radius=15.0)
    elif name == 'rrt_connect':
        return 'continuous', rrt_connect_step_generator(m_map.start, m_map.goal, m_map.dim, m_map.obstacles)
    elif name == 'est':
        return 'continuous', est_step_generator(m_map.start, m_map.goal, m_map.dim, m_map.obstacles, step_size=15.0, density_radius=15.0)
    elif name == 'bit_star':
        return 'continuous', bit_star_step_generator(m_map.start, m_map.goal, m_map.dim, m_map.obstacles, batch_size=80)
    elif name == 'fmt_star':
        return 'continuous', fmt_star_step_generator(m_map.start, m_map.goal, m_map.dim, m_map.obstacles, num_samples=600)
    elif name == 'prm':
        return 'continuous', prm_step_generator(m_map.start, m_map.goal, m_map.dim, m_map.obstacles, num_samples=200)
    else:
        raise ValueError(f"Unknown algorithm: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', type=str, help='Algorithm name (e.g., bfs, rrt, astar, informed_rrt_star)')
    parser.add_argument('--record', action='store_true', help='Record to GIF')
    args = parser.parse_args()

    temp_mode = 'discrete' if args.algo.lower() in ['bfs', 'dfs', 'a_star'] else 'continuous'
    
    mujoco_map = MapManager(dim=50, mode=temp_mode)
    model, data = mujoco_map.create_mujoco_model()
    mujoco_map.reset_scene()

    recorder = None
    if args.record:
        recorder = MujocoRecorder(model, height=480, width=640, fps=20)

    mode, algo_gen = get_algorithm(args.algo, mujoco_map)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [25, 25, 0]
        viewer.cam.distance = 75.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -70

        for i in range(40):
            mujoco_map.update_wall_animation(i/40)
            viewer.sync()
            time.sleep(0.01)
            
        for i in range(50):
            prog = 1 - (1 - i/50)**3
            mujoco_map.update_obstacle_animation(prog)
            viewer.sync()
            time.sleep(0.01)

        if mode == 'discrete':
            for i in range(30):
                mujoco_map.update_grid_animation(i/30)
                viewer.sync()
                time.sleep(0.01)

        time.sleep(0.5)

        path_nodes = []
        graph_edges = []
        path_found = False
        frame_skip = 0
        path_completion_time = None

        while viewer.is_running():
            step_start = time.time()
            
            if not path_found:
                try:
                    event_type, payload = next(algo_gen)
                    
                    if mode == 'discrete':
                        if event_type == "visiting":
                            mujoco_map.set_tile_color(payload[0], payload[1], [0, 1, 1, 0.4])
                        elif event_type == "path":
                            for (px, py) in payload:
                                mujoco_map.set_tile_color(px, py, [0, 1, 0, 0.9])
                            path_found = True
                            path_completion_time = time.time()

                    else: 
                        if event_type == "sampling":
                            mujoco_map.move_probe(payload[0], payload[1])
                            mujoco_map.add_trace_marker(payload[0], payload[1]) 

                        elif event_type == "edge":
                            graph_edges.append(payload)

                        elif event_type == "path":
                            path_nodes = payload
                            path_found = True
                            path_completion_time = time.time()
                            
                except StopIteration:

                    path_completion_time = time.time()
                    path_found = True
                    
            
            else:
                if args.record and path_completion_time is not None and (time.time() - path_completion_time > 5.0):
                    print("animation finished")
                    break

                if mode == 'continuous' and path_nodes:
                    t = (time.time() % 3.0) / 3.0
                    idx = int(t * (len(path_nodes) - 1))
                    p1 = np.array(path_nodes[idx])
                    p2 = np.array(path_nodes[min(idx+1, len(path_nodes)-1)])
                    curr = p1 + (p2 - p1) * ((t * (len(path_nodes) - 1)) - idx)
                    mujoco_map.move_probe(curr[0], curr[1])

            if mode == 'continuous' and viewer.user_scn:
                viewer.user_scn.ngeom = 0 
                
                for (p1, p2) in graph_edges:
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break 
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                        mujoco.mjtGeom.mjGEOM_LINE,            
                        2.0,                              
                        np.array([p1[0], p1[1], 0.1]),          
                        np.array([p2[0], p2[1], 0.1])                 
                    )
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = np.array([1, 1, 1, 0.3])
                    viewer.user_scn.ngeom += 1

                if path_found and len(path_nodes) > 1:
                    for i in range(len(path_nodes) - 1):
                        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
                        
                        curr_node = path_nodes[i]
                        next_node = path_nodes[i+1]

                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            mujoco.mjtGeom.mjGEOM_LINE,
                            6.0, 
                            np.array([curr_node[0], curr_node[1], 0.15]),
                            np.array([next_node[0], next_node[1], 0.15])
                        )
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = np.array([1, 0, 0, 1.0])
                        viewer.user_scn.ngeom += 1

            viewer.sync()
            if recorder and frame_skip % 2 == 0:
                recorder.capture_frame(data, viewer=viewer)
            frame_skip += 1

            time.sleep(max(0, 0.01 - (time.time() - step_start)))
    if args.record and recorder:
        recorder.save_gif(f"./assets/{args.algo}.gif")

if __name__ == "__main__":
    main()