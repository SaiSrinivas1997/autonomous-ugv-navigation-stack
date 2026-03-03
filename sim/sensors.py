import math
import random
import time
import pybullet as p


class SimulatedIMU:
    def __init__(
        self,
        robot_id,
        gyro_noise_std=0.05,
        gyro_bias_drift=0.001,
        accel_noise_std=0.05,
        accel_bias=None,
    ):
        self.robot_id = robot_id
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_drift = gyro_bias_drift
        self.accel_noise_std = accel_noise_std
        self.accel_bias = accel_bias or [
            random.gauss(0, 0.02),
            random.gauss(0, 0.02),
            random.gauss(0, 0.02),
        ]
        self.gyro_bias = [0.0, 0.0, 0.0]
        self.last_time = time.time()

    def read(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        R = [
            [rot_matrix[0], rot_matrix[3], rot_matrix[6]],
            [rot_matrix[1], rot_matrix[4], rot_matrix[7]],
            [rot_matrix[2], rot_matrix[5], rot_matrix[8]],
        ]
        wx = R[0][0] * ang_vel[0] + R[0][1] * ang_vel[1] + R[0][2] * ang_vel[2]
        wy = R[1][0] * ang_vel[0] + R[1][1] * ang_vel[1] + R[1][2] * ang_vel[2]
        wz = R[2][0] * ang_vel[0] + R[2][1] * ang_vel[1] + R[2][2] * ang_vel[2]

        self.gyro_bias[0] += random.gauss(0, self.gyro_bias_drift * dt)
        self.gyro_bias[1] += random.gauss(0, self.gyro_bias_drift * dt)
        self.gyro_bias[2] += random.gauss(0, self.gyro_bias_drift * dt)

        noisy_gyro = [
            wx + self.gyro_bias[0] + random.gauss(0, self.gyro_noise_std),
            wy + self.gyro_bias[1] + random.gauss(0, self.gyro_noise_std),
            wz + self.gyro_bias[2] + random.gauss(0, self.gyro_noise_std),
        ]

        gravity_world = [0, 0, -9.81]
        gx = (
            R[0][0] * gravity_world[0]
            + R[0][1] * gravity_world[1]
            + R[0][2] * gravity_world[2]
        )
        gy = (
            R[1][0] * gravity_world[0]
            + R[1][1] * gravity_world[1]
            + R[1][2] * gravity_world[2]
        )
        gz = (
            R[2][0] * gravity_world[0]
            + R[2][1] * gravity_world[1]
            + R[2][2] * gravity_world[2]
        )

        noisy_accel = [
            gx + self.accel_bias[0] + random.gauss(0, self.accel_noise_std),
            gy + self.accel_bias[1] + random.gauss(0, self.accel_noise_std),
            gz + self.accel_bias[2] + random.gauss(0, self.accel_noise_std),
        ]

        return noisy_gyro, noisy_accel, now


class SimulatedGPS:
    def __init__(self, robot_id, position_noise_std=0.5, drift_speed=0.01):
        self.robot_id = robot_id
        self.position_noise_std = position_noise_std
        self.drift = [0.0, 0.0]
        self.drift_speed = drift_speed

    def read(self):
        now = time.time()
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)

        self.drift[0] += random.gauss(0, self.drift_speed * 0.01)
        self.drift[1] += random.gauss(0, self.drift_speed * 0.01)
        self.drift[0] = max(min(self.drift[0], 1.0), -1.0)
        self.drift[1] = max(min(self.drift[1], 1.0), -1.0)

        noisy_pos = [
            pos[0] + self.drift[0] + random.gauss(0, self.position_noise_std),
            pos[1] + self.drift[1] + random.gauss(0, self.position_noise_std),
            pos[2],
        ]
        return noisy_pos, now


class SimulatedEncoders:
    def __init__(
        self,
        robot_id,
        left_joints,
        right_joints,
        wheel_radius=0.165,
        track_width=0.55,
        velocity_noise_std=0.02,
        slip_probability=0.001,
    ):
        self.robot_id = robot_id
        self.left_joints = left_joints
        self.right_joints = right_joints
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.velocity_noise_std = velocity_noise_std
        self.slip_probability = slip_probability

    def _get_avg_velocity(self, joints):
        total = 0.0
        for j in joints:
            state = p.getJointState(self.robot_id, j)
            total += state[1]
        return total / len(joints)

    def read(self):
        now = time.time()
        omega_left = self._get_avg_velocity(self.left_joints)
        omega_right = self._get_avg_velocity(self.right_joints)

        omega_left += random.gauss(0, self.velocity_noise_std)
        omega_right += random.gauss(0, self.velocity_noise_std)

        if random.random() < self.slip_probability:
            omega_left *= random.uniform(0.5, 0.9)
        if random.random() < self.slip_probability:
            omega_right *= random.uniform(0.5, 0.9)

        v_left = omega_left * self.wheel_radius
        v_right = omega_right * self.wheel_radius
        linear_vel = (v_right + v_left) / 2.0
        angular_vel = (v_right - v_left) / self.track_width

        return linear_vel, angular_vel, v_left, v_right, now


class SimulatedUltrasonic:
    def __init__(
        self,
        robot_id,
        height=0.2,
        range_max=3.0,
        range_min=0.02,
        noise_std=0.02,
        crosstalk_probability=0.005,
    ):
        self.robot_id = robot_id
        self.height = height
        self.range_max = range_max
        self.range_min = range_min
        self.noise_std = noise_std
        self.crosstalk_probability = crosstalk_probability
        self.sensor_angles = [math.radians(20), math.radians(-20)]

    def _raycast(self, pos, yaw, angle_offset):
        angle = yaw + angle_offset
        from_pt = [pos[0], pos[1], pos[2] + self.height]
        to_pt = [
            pos[0] + self.range_max * math.cos(angle),
            pos[1] + self.range_max * math.sin(angle),
            pos[2] + self.height,
        ]
        result = p.rayTest(from_pt, to_pt)
        hit_fraction = result[0][2]
        return hit_fraction * self.range_max if hit_fraction < 1.0 else self.range_max

    def read(self):
        now = time.time()
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        distances = []
        for angle_offset in self.sensor_angles:
            dist = self._raycast(pos, yaw, angle_offset)
            dist += random.gauss(0, self.noise_std)
            if random.random() < self.crosstalk_probability:
                dist = random.uniform(self.range_min, self.range_max * 0.5)
            dist = max(self.range_min, min(dist, self.range_max))
            distances.append(dist)

        return distances, now


class SimulatedMagnetometer:
    def __init__(
        self,
        robot_id,
        heading_noise_std=0.02,
        hard_iron_bias=None,
        soft_iron_scale=None,
    ):
        self.robot_id = robot_id
        self.heading_noise_std = heading_noise_std
        self.hard_iron_bias = (
            hard_iron_bias if hard_iron_bias is not None else random.gauss(0, 0.05)
        )
        self.soft_iron_scale = (
            soft_iron_scale
            if soft_iron_scale is not None
            else random.uniform(0.95, 1.05)
        )

    def read(self):
        now = time.time()
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        true_yaw = p.getEulerFromQuaternion(orn)[2]

        noisy_yaw = true_yaw * self.soft_iron_scale
        noisy_yaw += self.hard_iron_bias
        noisy_yaw += random.gauss(0, self.heading_noise_std)
        noisy_yaw = (noisy_yaw + math.pi) % (2 * math.pi) - math.pi

        return float(noisy_yaw), now
