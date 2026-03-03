import pybullet as p


def get_pose(robot_id):
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    return pos, orn
