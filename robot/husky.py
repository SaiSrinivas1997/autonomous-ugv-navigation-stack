import pybullet as p


class Husky:
    def __init__(self):
        self.id = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

        self.left_wheels = []
        self.right_wheels = []

        for i in range(p.getNumJoints(self.id)):
            name = p.getJointInfo(self.id, i)[1].decode()
            if "left" in name:
                self.left_wheels.append(i)
            elif "right" in name:
                self.right_wheels.append(i)
