import pybullet as p
import pybullet_data


class Simulator:
    def __init__(self, gui=True, dt=1 / 240):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)

        p.loadURDF("plane.urdf")

    def step(self):
        p.stepSimulation()
