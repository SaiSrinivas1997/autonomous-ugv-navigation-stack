import pybullet as p


class World:
    def __init__(self):
        self.obstacles = []

    def add_box(self, pos, size, mass=0):
        """
        pos  : [x, y, z]
        size : [sx, sy, sz]  (half extents)
        mass : 0 = static obstacle
        """
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)

        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=size, rgbaColor=[1, 0, 0, 1]  # red
        )

        body = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )

        self.obstacles.append(body)
        return body
