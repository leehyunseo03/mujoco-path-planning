import imageio
import mujoco
import numpy as np

class MujocoRecorder:
    def __init__(self, model, height=480, width=640, fps=20):
        self.renderer = mujoco.Renderer(model, height, width)
        self.frames = []
        self.fps = fps

    def capture_frame(self, data, viewer=None):
        if viewer is not None:
            self.renderer.update_scene(data, camera=viewer.cam, scene_option=viewer.opt)
            
            if viewer.user_scn.ngeom > 0:
                for i in range(viewer.user_scn.ngeom):
                    if self.renderer.scene.ngeom >= self.renderer.scene.maxgeom:
                        break
                    src_geom = viewer.user_scn.geoms[i]
                    dst_geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]

                    dst_geom.type = src_geom.type
                    dst_geom.size[:] = src_geom.size[:]
                    dst_geom.pos[:] = src_geom.pos[:]
                    dst_geom.mat[:] = src_geom.mat[:]
                    dst_geom.rgba[:] = src_geom.rgba[:]
                    self.renderer.scene.ngeom += 1
        else:
            self.renderer.update_scene(data)

        pixel_values = self.renderer.render()
        self.frames.append(pixel_values)

    def save_gif(self, filename="output.gif"):
        if not self.frames: return
        imageio.mimsave(filename, self.frames, fps=self.fps, loop=0)
        print(f"Saved: {filename}")