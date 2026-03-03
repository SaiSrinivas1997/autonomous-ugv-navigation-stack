import pybullet as p


class ArrowTeleop:
    def __init__(self, v_step=0.02, w_step=0.5, v_max=1.0, w_max=1.0):
        self.v = 0.0
        self.w = 0.0
        self.V_STEP = v_step
        self.W_STEP = w_step
        self.V_MAX = v_max
        self.W_MAX = w_max

    def update(self):
        keys = p.getKeyboardEvents()

        # Linear velocity
        if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN):
            self.v += self.V_STEP
        elif p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN):
            self.v -= self.V_STEP
        else:
            self.v *= 0.95

        # Angular velocity
        if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN):
            self.w += self.W_STEP
        elif p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN):
            self.w -= self.W_STEP
        else:
            self.w *= 0.95

        # Stop
        if p.B3G_SPACE in keys and (keys[p.B3G_SPACE] & p.KEY_IS_DOWN):
            self.v, self.w = 0.0, 0.0

        self.v = max(min(self.v, self.V_MAX), -self.V_MAX)
        self.w = max(min(self.w, self.W_MAX), -self.W_MAX)

        return self.v, self.w
