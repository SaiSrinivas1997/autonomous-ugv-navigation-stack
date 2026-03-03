import pybullet as p


class SkidSteerController:
    def __init__(self, robot_id, left, right, wheel_radius, track_width):
        self.robot = robot_id
        self.left = left
        self.right = right
        self.r = wheel_radius
        self.b = track_width

    def cmd_vel(self, v, w):
        vl = (v - w * self.b / 2) / self.r
        vr = (v + w * self.b / 2) / self.r

        for j in self.left:
            p.setJointMotorControl2(
                self.robot, j, p.VELOCITY_CONTROL, targetVelocity=vl
            )
        for j in self.right:
            p.setJointMotorControl2(
                self.robot, j, p.VELOCITY_CONTROL, targetVelocity=vr
            )
