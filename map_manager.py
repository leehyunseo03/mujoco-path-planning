import numpy as np
import mujoco

class MapManager:
    def __init__(self, dim=50, mode='discrete'):
        self.dim = dim
        self.mode = mode
        
        self.model = None
        self.data = None
        self.wall_mocap_ids = [] 
        self.obs_mocap_ids = [] 
        
        self.grid = np.zeros((dim, dim), dtype=bool)
        self.cell_geom_ids = []
        
        self.obstacles = []
        self.probe_mocap_id = -1
        

        self.max_traces = 600
        self.trace_mocap_ids = []
        self.current_trace_idx = 0
        
        self.start = (0, 0)
        self.goal = (dim-1, dim-1)

        np.random.seed(3) 
        
        if self.mode == 'discrete':
            self._generate_maze()
        elif self.mode == 'continuous':
            self._generate_continuous_obstacles()

    def _generate_random_obstacles(self):
        self.start = (0, 0)
        self.goal = (self.dim-1, self.dim-1)
        num_obstacles = 50
        for _ in range(num_obstacles):
            x, y = np.random.randint(0, self.dim, 2)
            dx, dy = np.random.randint(1, 5, 2)
            end_x = min(x + dx, self.dim)
            end_y = min(y + dy, self.dim)
            self.grid[x:end_x, y:end_y] = True
        self.grid[0:5, 0:5] = False
        self.grid[self.dim-5:self.dim, self.dim-5:self.dim] = False

    def _generate_maze(self):
        self.start = (0, 0)
        self.goal = (self.dim-1, self.dim-1)
        self.grid.fill(True)
        sx, sy = 0, 0
        self.grid[sx, sy] = False
        stack = [(sx, sy)]
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.dim and 0 <= ny < self.dim:
                    if self.grid[nx, ny]:
                        neighbors.append((nx, ny, dx, dy))
            if neighbors:
                idx = np.random.randint(len(neighbors))
                nx, ny, dx, dy = neighbors[idx]
                self.grid[cx + dx // 2, cy + dy // 2] = False
                self.grid[nx, ny] = False
                stack.append((nx, ny))
            else:
                stack.pop()
        
        self.grid[self.dim-1, self.dim-1] = False
        self.grid[self.dim-2, self.dim-1] = False
        self.grid[self.dim-1, self.dim-2] = False

    def _generate_continuous_obstacles(self):
        self.start = np.array([2.0, 2.0])
        self.goal = np.array([self.dim-2.0, self.dim-2.0])
        
        num_obstacles = 50
        for _ in range(num_obstacles):
            cx = np.random.uniform(5, self.dim - 5)
            cy = np.random.uniform(5, self.dim - 5)
            sx = np.random.uniform(1.0, 3.0)
            sy = np.random.uniform(1.0, 3.0)
            
            dist_start = np.linalg.norm([cx - self.start[0], cy - self.start[1]])
            dist_goal = np.linalg.norm([cx - self.goal[0], cy - self.goal[1]])
            
            if dist_start > 5.0 and dist_goal > 5.0:
                self.obstacles.append((cx, cy, sx, sy))

    def create_mujoco_model(self):
        center = self.dim / 2.0 - 0.5
        wall_len = self.dim / 2.0 + 1.0
        wall_z = -5.0 

        xml_string = f"""
        <mujoco>
          <option timestep="0.005" gravity="0 0 -9.81"/>
          <statistic center="{center} {center} 0" extent="{self.dim * 1.2}"/>
          <visual>
            <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0.0 0.0 0.0"/>
            <rgba haze="0 0 0 0"/>
            <global azimuth="-90" elevation="-45"/>
            <quality shadowsize="4096"/>
          </visual>
          <asset>
            <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" rgb1=".9 .9 .9" rgb2=".85 .85 .85"/>
            <material name="floor_mat" texture="floor_tex" texrepeat="5 5" texuniform="true" reflectance="0.0"/>
            <texture name="wall_tex" type="2d" builtin="flat" height="32" width="32" rgb1="0.2 0.2 0.2"/>
            <material name="wall_mat" texture="wall_tex" reflectance="0.0"/>
          </asset>
          <worldbody>
            <light pos="{center} {center} 50" dir="0 0 -1" castshadow="true" diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3"/>
            <geom name="floor" type="plane" pos="{center} {center} 0" size="{self.dim} {self.dim} 0.1" material="floor_mat"/>
            
            <body name="wall_south" mocap="true" pos="{center} -1.0 {wall_z}"><geom type="box" size="{wall_len} 0.5 1.5" material="wall_mat" rgba="0.1 0.1 0.1 1"/></body>
            <body name="wall_north" mocap="true" pos="{center} {self.dim} {wall_z}"><geom type="box" size="{wall_len} 0.5 1.5" material="wall_mat" rgba="0.1 0.1 0.1 1"/></body>
            <body name="wall_west" mocap="true" pos="-1.0 {center} {wall_z}"><geom type="box" size="0.5 {wall_len} 1.5" material="wall_mat" rgba="0.1 0.1 0.1 1"/></body>
            <body name="wall_east" mocap="true" pos="{self.dim} {center} {wall_z}"><geom type="box" size="0.5 {wall_len} 1.5" material="wall_mat" rgba="0.1 0.1 0.1 1"/></body>

            <body pos="{self.start[0]} {self.start[1]} 0">
                <geom type="cylinder" size="0.3 0.1" rgba="0.0 0.8 0.4 1"/>
            </body>
            <body pos="{self.goal[0]} {self.goal[1]} 0">
                <geom type="cylinder" size="0.3 5.0" rgba="0.9 0.2 0.2 1.0"/>
                <geom type="box" pos="0 0 5.0" size="0.8 0.1 0.5" rgba="0.9 0.2 0.2 1"/>
            </body>
        """
        
        if self.mode == 'discrete':
            for x in range(self.dim):
                for y in range(self.dim):
                    if not self.grid[x, y]:
                        xml_string += f"""<geom name="cell_{x}_{y}" type="box" pos="{x} {y} 0.02" size="0.45 0.45 0.02" rgba="0.5 0.5 0.5 0"/>"""
            
            for x in range(self.dim):
                for y in range(self.dim):
                    if self.grid[x, y]:
                        h_vary = 1.0 + np.random.uniform(0.0, 0.2)
                        c_vary = np.random.uniform(-0.05, 0.05)
                        base_c = 0.3
                        xml_string += f"""
                        <body name="obs_{x}_{y}" mocap="true" pos="{x} {y} 1.0">
                            <geom type="box" size="0.5 0.5 {h_vary}" rgba="{base_c+c_vary} {base_c+c_vary} {base_c+c_vary+0.05} 1"/>
                        </body>"""
                        
        elif self.mode == 'continuous':
            xml_string += f"""
            <body name="probe" mocap="true" pos="{self.start[0]} {self.start[1]} 0.15">
                <geom type="box" size="0.35 0.22 0.1" pos="0 0 0.1" rgba="1 0.6 0.1 1"/>
                
                <geom type="box" size="0.2 0.2 0.08" pos="-0.1 0 0.25" rgba="1 0.8 0.3 1"/>
                
                <geom type="cylinder" size="0.12 0.05" pos="0.25 0.25 0.0" zaxis="0 1 0" rgba="0.2 0.2 0.2 1"/>
                <geom type="cylinder" size="0.12 0.05" pos="0.25 -0.25 0.0" zaxis="0 1 0" rgba="0.2 0.2 0.2 1"/>
                <geom type="cylinder" size="0.12 0.05" pos="-0.25 0.25 0.0" zaxis="0 1 0" rgba="0.2 0.2 0.2 1"/>
                <geom type="cylinder" size="0.12 0.05" pos="-0.25 -0.25 0.0" zaxis="0 1 0" rgba="0.2 0.2 0.2 1"/>
                
                <geom type="sphere" size="0.08" pos="0.35 0.12 0.12" rgba="1 1 1 1"/>
                <geom type="sphere" size="0.08" pos="0.35 -0.12 0.12" rgba="1 1 1 1"/>
                
                <light pos="0 0 2" dir="0 0 -1" diffuse="1 0.8 0.6" attenuation="1 0 0"/>
            </body>"""
            
            for i, (x, y, sx, sy) in enumerate(self.obstacles):
                h_vary = 1.0 + np.random.uniform(0.0, 0.3)

                c_base = 0.25
                c_r = c_base + np.random.uniform(-0.05, 0.05)
                c_g = c_base + np.random.uniform(-0.05, 0.05)
                c_b = c_base + 0.05 + np.random.uniform(-0.05, 0.05)
                
                xml_string += f"""
                <body name="obs_{i}" mocap="true" pos="{x} {y} 1.0">
                    <geom type="box" size="{sx} {sy} {h_vary}" rgba="{c_r} {c_g} {c_b} 1"/>
                </body>"""

            for i in range(self.max_traces):
                xml_string += f"""
                <body name="trace_{i}" mocap="true" pos="0 0 -5.0">
                    <geom type="sphere" size="0.15" rgba="0 0.5 1 1" contype="0" conaffinity="0"/>
                </body>"""

        xml_string += "</worldbody></mujoco>"
        
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        return self.model, self.data
    
    def reset_scene(self):
        if self.model is None: return

        wall_names = ["wall_south", "wall_north", "wall_west", "wall_east"]
        self.wall_mocap_ids = []
        for name in wall_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                mid = self.model.body_mocapid[bid]
                self.wall_mocap_ids.append(mid)
                self.data.mocap_pos[mid][2] = -5.0

        self.obs_mocap_ids = []
        if self.mode == 'discrete':
            for x in range(self.dim):
                for y in range(self.dim):
                    if self.grid[x, y]:
                        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{x}_{y}")
                        if bid != -1: self.obs_mocap_ids.append(self.model.body_mocapid[bid])
        elif self.mode == 'continuous':
            for i in range(len(self.obstacles)):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{i}")
                if bid != -1: self.obs_mocap_ids.append(self.model.body_mocapid[bid])
        
        for mid in self.obs_mocap_ids:
            self.data.mocap_pos[mid][2] = -3.0

        if self.mode == 'discrete':
            self.cell_geom_ids = []
            for x in range(self.dim):
                for y in range(self.dim):
                    if not self.grid[x, y]:
                        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cell_{x}_{y}")
                        if gid != -1:
                            self.cell_geom_ids.append(gid)
                            self.model.geom_rgba[gid][3] = 0.0
                            
        if self.mode == 'continuous':
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "probe")
            if bid != -1: self.probe_mocap_id = self.model.body_mocapid[bid]
            
            self.trace_mocap_ids = []
            self.current_trace_idx = 0
            for i in range(self.max_traces):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"trace_{i}")
                if bid != -1:
                    mid = self.model.body_mocapid[bid]
                    self.trace_mocap_ids.append(mid)
                    self.data.mocap_pos[mid][2] = -5.0 

        mujoco.mj_step(self.model, self.data)

    def update_wall_animation(self, progress):
        z = -5.0 + (1.5 - (-5.0)) * progress
        for mid in self.wall_mocap_ids: self.data.mocap_pos[mid][2] = z
        mujoco.mj_step(self.model, self.data)

    def update_obstacle_animation(self, progress):
        z = -3.0 + (1.0 - (-3.0)) * progress
        for mid in self.obs_mocap_ids: self.data.mocap_pos[mid][2] = z
        mujoco.mj_step(self.model, self.data)

    def update_grid_animation(self, progress):
        if self.mode != 'discrete': return
        alpha = 0.3 * progress 
        for gid in self.cell_geom_ids: self.model.geom_rgba[gid][3] = alpha

    def set_tile_color(self, x, y, color):
        if self.mode != 'discrete': return
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cell_{x}_{y}")
        if gid != -1: self.model.geom_rgba[gid] = color

    def move_probe(self, x, y):
        if self.mode != 'continuous' or self.probe_mocap_id == -1: return
        self.data.mocap_pos[self.probe_mocap_id][0] = x
        self.data.mocap_pos[self.probe_mocap_id][1] = y
        mujoco.mj_step(self.model, self.data)

    def add_trace_marker(self, x, y):
        if self.mode != 'continuous': return
        if self.current_trace_idx < len(self.trace_mocap_ids):
            mid = self.trace_mocap_ids[self.current_trace_idx]
            self.data.mocap_pos[mid][0] = x
            self.data.mocap_pos[mid][1] = y
            self.data.mocap_pos[mid][2] = 0.1 
            self.current_trace_idx += 1
            mujoco.mj_step(self.model, self.data)