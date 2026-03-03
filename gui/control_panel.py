import tkinter as tk
import threading
import math
import pybullet as p


class UGVControlPanel:
    """
    Tkinter GUI — dark terminal aesthetic.

    Shows:
    - Robot position + heading arrow
    - D* Lite planned path
    - LiDAR obstacle points
    - Click on map to set goal
    - Live telemetry
    - Mode toggle buttons
    """

    MAP_SIZE = 520  # canvas pixels
    WORLD_RANGE = 12.0  # meters visible from robot center

    # Color palette
    BG = "#0d1117"
    BG2 = "#161b22"
    BORDER = "#30363d"
    ACCENT = "#00ff88"
    ACCENT2 = "#00aaff"
    WARN = "#ff6b35"
    TEXT = "#e6edf3"
    DIM = "#8b949e"
    GRID = "#1c2128"
    PATH_COL = "#00ff88"
    OBS_COL = "#ff6b35"
    ROBOT_COL = "#00aaff"
    GOAL_COL = "#ffcc00"

    def __init__(self, app_ref):
        self.app = app_ref
        self.running = True
        self._thread = None

        # Cached state
        self._rx = 0.0
        self._ry = 0.0
        self._ryaw = 0.0
        self._gx = 5.0
        self._gy = 3.0
        self._path = []
        self._obs = []
        self._ekf = {}
        self._truth = (0.0, 0.0, 0.0)
        self._mode = "TELEOP"

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name="GUI-Thread", daemon=True
        )
        self._thread.start()

    def stop(self):
        self.running = False

    # ------------------------------------------------------------------ #
    #  Build UI                                                            #
    # ------------------------------------------------------------------ #

    def _run(self):
        self.root = tk.Tk()
        self.root.title("UGV Control Panel")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)
        self._build()
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.mainloop()

    def _build(self):
        r = self.root

        # Title bar
        tk.Label(
            r,
            text="⬡  UGV AUTONOMY STACK",
            font=("Courier", 13, "bold"),
            fg=self.ACCENT,
            bg=self.BG,
        ).pack(pady=(10, 6))

        # Horizontal layout: sidebar | map
        body = tk.Frame(r, bg=self.BG)
        body.pack(padx=12, pady=4)

        side = tk.Frame(body, bg=self.BG, width=200)
        side.pack(side="left", fill="y", padx=(0, 10))
        side.pack_propagate(False)

        self._build_telemetry(side)
        self._build_mode(side)
        self._build_instructions(side)

        self._build_map(body)

    # ── Sidebar sections ───────────────────────────────────────────────

    def _sec(self, parent, title):
        f = tk.Frame(
            parent, bg=self.BG2, highlightbackground=self.BORDER, highlightthickness=1
        )
        f.pack(fill="x", pady=4)
        tk.Label(
            f, text=title, font=("Courier", 8, "bold"), fg=self.ACCENT, bg=self.BG2
        ).pack(anchor="w", padx=8, pady=(6, 2))
        return f

    def _build_telemetry(self, parent):
        sec = self._sec(parent, "▸ TELEMETRY")
        self._tlabels = {}
        rows = [
            ("pos", "EKF pos "),
            ("yaw", "EKF yaw "),
            ("spd", "speed   "),
            ("truth", "truth   "),
            ("dist", "to goal "),
            ("wpts", "waypts  "),
        ]
        for key, label in rows:
            row = tk.Frame(sec, bg=self.BG2)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(
                row,
                text=label,
                font=("Courier", 8),
                fg=self.DIM,
                bg=self.BG2,
                anchor="w",
                width=9,
            ).pack(side="left")
            lbl = tk.Label(
                row,
                text="—",
                font=("Courier", 8, "bold"),
                fg=self.TEXT,
                bg=self.BG2,
                anchor="w",
            )
            lbl.pack(side="left")
            self._tlabels[key] = lbl
        tk.Frame(sec, bg=self.BG2, height=4).pack()

    def _build_mode(self, parent):
        sec = self._sec(parent, "▸ MODE")

        self._mode_lbl = tk.Label(
            sec, text="TELEOP", font=("Courier", 14, "bold"), fg=self.WARN, bg=self.BG2
        )
        self._mode_lbl.pack(pady=(4, 4))

        bf = tk.Frame(sec, bg=self.BG2)
        bf.pack(fill="x", padx=8, pady=(0, 8))

        self._btn_teleop = tk.Button(
            bf,
            text="TELEOP",
            font=("Courier", 9, "bold"),
            fg=self.BG,
            bg=self.WARN,
            relief="flat",
            cursor="hand2",
            activebackground="#cc5522",
            command=lambda: self._switch("TELEOP"),
        )
        self._btn_teleop.pack(side="left", expand=True, fill="x", padx=(0, 2))

        self._btn_auto = tk.Button(
            bf,
            text="AUTO",
            font=("Courier", 9, "bold"),
            fg=self.BG,
            bg=self.BORDER,
            relief="flat",
            cursor="hand2",
            activebackground="#00cc66",
            command=lambda: self._switch("AUTONOMOUS"),
        )
        self._btn_auto.pack(side="left", expand=True, fill="x", padx=(2, 0))

    def _build_instructions(self, parent):
        sec = self._sec(parent, "▸ CONTROLS")
        lines = [
            ("click map", "set goal + auto"),
            ("'a' key", "toggle mode"),
            ("arrows", "teleop drive"),
        ]
        for key, action in lines:
            row = tk.Frame(sec, bg=self.BG2)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(
                row,
                text=key,
                font=("Courier", 8, "bold"),
                fg=self.ACCENT2,
                bg=self.BG2,
                width=10,
                anchor="w",
            ).pack(side="left")
            tk.Label(
                row,
                text=action,
                font=("Courier", 8),
                fg=self.DIM,
                bg=self.BG2,
                anchor="w",
            ).pack(side="left")

        # Legend
        tk.Frame(sec, bg=self.BORDER, height=1).pack(fill="x", padx=8, pady=6)
        legend = [
            (self.ROBOT_COL, "Robot + heading"),
            (self.PATH_COL, "D* Lite path"),
            (self.OBS_COL, "LiDAR obstacles"),
            (self.GOAL_COL, "Goal"),
        ]
        for color, label in legend:
            row = tk.Frame(sec, bg=self.BG2)
            row.pack(fill="x", padx=8, pady=1)
            dot = tk.Canvas(row, width=10, height=10, bg=self.BG2, highlightthickness=0)
            dot.pack(side="left")
            dot.create_oval(1, 1, 9, 9, fill=color, outline="")
            tk.Label(
                row, text=label, font=("Courier", 8), fg=self.DIM, bg=self.BG2
            ).pack(side="left", padx=4)
        tk.Frame(sec, bg=self.BG2, height=4).pack()

    # ── Map canvas ─────────────────────────────────────────────────────

    def _build_map(self, parent):
        mf = tk.Frame(parent, bg=self.BG)
        mf.pack(side="left")

        tk.Label(
            mf, text="click to set goal", font=("Courier", 8), fg=self.DIM, bg=self.BG
        ).pack()

        self._cv = tk.Canvas(
            mf,
            width=self.MAP_SIZE,
            height=self.MAP_SIZE,
            bg=self.BG,
            highlightthickness=1,
            highlightbackground=self.BORDER,
            cursor="crosshair",
        )
        self._cv.pack()
        self._cv.bind("<Button-1>", self._on_click)

        self._coord_lbl = tk.Label(
            mf, text="", font=("Courier", 8), fg=self.DIM, bg=self.BG
        )
        self._coord_lbl.pack(pady=(2, 0))
        self._cv.bind("<Motion>", self._on_mouse_move)

    # ------------------------------------------------------------------ #
    #  Coordinate helpers                                                  #
    # ------------------------------------------------------------------ #

    def _w2c(self, wx, wy):
        """World → canvas pixels (robot centered)."""
        scale = self.MAP_SIZE / (2 * self.WORLD_RANGE)
        cx = self.MAP_SIZE / 2 + (wx - self._rx) * scale
        cy = self.MAP_SIZE / 2 - (wy - self._ry) * scale
        return cx, cy

    def _c2w(self, cx, cy):
        """Canvas pixels → world coordinates."""
        scale = 2 * self.WORLD_RANGE / self.MAP_SIZE
        wx = self._rx + (cx - self.MAP_SIZE / 2) * scale
        wy = self._ry - (cy - self.MAP_SIZE / 2) * scale
        return wx, wy

    def _in_canvas(self, cx, cy):
        return 0 <= cx <= self.MAP_SIZE and 0 <= cy <= self.MAP_SIZE

    # ------------------------------------------------------------------ #
    #  Events                                                              #
    # ------------------------------------------------------------------ #

    def _on_click(self, event):
        wx, wy = self._c2w(event.x, event.y)
        self._gx, self._gy = wx, wy
        self.app.planning_runner.set_goal(wx, wy)
        self.app.auto_ctrl.set_mode("AUTONOMOUS")
        print(f"[GUI] Goal → ({wx:.1f}, {wy:.1f})")

    def _on_mouse_move(self, event):
        wx, wy = self._c2w(event.x, event.y)
        self._coord_lbl.config(text=f"world ({wx:.1f}, {wy:.1f})")

    def _switch(self, mode):
        self.app.auto_ctrl.set_mode(mode)

    def _close(self):
        self.running = False
        self.root.destroy()

    # ------------------------------------------------------------------ #
    #  Update loop                                                         #
    # ------------------------------------------------------------------ #

    def _tick(self):
        if not self.running:
            return
        try:
            self._fetch()
            self._update_telemetry()
            self._update_mode()
            self._draw()
        except Exception:
            pass
        self.root.after(100, self._tick)  # 10Hz

    def _fetch(self):
        ekf = self.app.ekf_runner.get_state()
        if ekf:
            self._rx = ekf["x"]
            self._ry = ekf["y"]
            self._ryaw = ekf["yaw"]
            self._ekf = ekf

        self._path = self.app.planning_runner.get_path()
        self._mode = self.app.auto_ctrl.get_mode()

        # LiDAR local → world
        lidar = self.app.lidar_buffer.read() or []
        cy, sy = math.cos(self._ryaw), math.sin(self._ryaw)
        self._obs = [
            (self._rx + cy * lx - sy * ly, self._ry + sy * lx + cy * ly)
            for lx, ly in lidar
        ]

        try:
            gtp, gto = p.getBasePositionAndOrientation(self.app.robot.id)
            self._truth = (gtp[0], gtp[1], p.getEulerFromQuaternion(gto)[2])
        except Exception:
            pass

    def _update_telemetry(self):
        e = self._ekf
        if not e:
            return
        dist = math.sqrt(
            (e.get("x", 0) - self._gx) ** 2 + (e.get("y", 0) - self._gy) ** 2
        )
        self._tlabels["pos"].config(text=f"({e.get('x',0):.2f}, {e.get('y',0):.2f})")
        self._tlabels["yaw"].config(text=f"{math.degrees(e.get('yaw',0)):.1f}°")
        self._tlabels["spd"].config(text=f"{e.get('v',0):.2f} m/s")
        self._tlabels["truth"].config(
            text=f"({self._truth[0]:.2f}, {self._truth[1]:.2f})"
        )
        self._tlabels["dist"].config(
            text=f"{dist:.2f} m", fg=self.ACCENT if dist < 0.5 else self.TEXT
        )
        self._tlabels["wpts"].config(text=str(len(self._path)))

    def _update_mode(self):
        if self._mode == "AUTONOMOUS":
            self._mode_lbl.config(text="AUTONOMOUS", fg=self.ACCENT)
            self._btn_auto.config(bg=self.ACCENT)
            self._btn_teleop.config(bg=self.BORDER)
        else:
            self._mode_lbl.config(text="TELEOP", fg=self.WARN)
            self._btn_teleop.config(bg=self.WARN)
            self._btn_auto.config(bg=self.BORDER)

    # ------------------------------------------------------------------ #
    #  Drawing                                                             #
    # ------------------------------------------------------------------ #

    def _draw(self):
        cv = self._cv
        cv.delete("all")
        self._draw_grid(cv)
        self._draw_obstacles(cv)
        self._draw_path(cv)
        self._draw_goal(cv)
        self._draw_robot(cv)
        self._draw_compass(cv)

    def _draw_grid(self, cv):
        """1m grid lines with coordinate labels."""
        for i in range(int(-self.WORLD_RANGE), int(self.WORLD_RANGE) + 2):
            # vertical
            wx = math.floor(self._rx) + i
            cx, _ = self._w2c(wx, 0)
            if 0 <= cx <= self.MAP_SIZE:
                cv.create_line(cx, 0, cx, self.MAP_SIZE, fill=self.GRID, width=1)
                cv.create_text(
                    cx + 2,
                    self.MAP_SIZE - 8,
                    text=f"{wx:.0f}",
                    font=("Courier", 6),
                    fill=self.BORDER,
                    anchor="w",
                )
            # horizontal
            wy = math.floor(self._ry) + i
            _, cy = self._w2c(0, wy)
            if 0 <= cy <= self.MAP_SIZE:
                cv.create_line(0, cy, self.MAP_SIZE, cy, fill=self.GRID, width=1)
                cv.create_text(
                    4,
                    cy - 6,
                    text=f"{wy:.0f}",
                    font=("Courier", 6),
                    fill=self.BORDER,
                    anchor="w",
                )

    def _draw_obstacles(self, cv):
        """LiDAR hits as small orange dots."""
        for wx, wy in self._obs:
            cx, cy = self._w2c(wx, wy)
            if self._in_canvas(cx, cy):
                cv.create_oval(
                    cx - 2, cy - 2, cx + 2, cy + 2, fill=self.OBS_COL, outline=""
                )

    def _draw_path(self, cv):
        """D* Lite path as dashed green line with waypoint markers."""
        if len(self._path) < 2:
            return

        pts = []
        for wx, wy in self._path:
            cx, cy = self._w2c(wx, wy)
            pts += [cx, cy]

        cv.create_line(pts, fill=self.PATH_COL, width=2, dash=(8, 4))

        for i, (wx, wy) in enumerate(self._path):
            cx, cy = self._w2c(wx, wy)
            if self._in_canvas(cx, cy):
                cv.create_oval(
                    cx - 4,
                    cy - 4,
                    cx + 4,
                    cy + 4,
                    fill=self.PATH_COL,
                    outline=self.BG,
                    width=1,
                )
                cv.create_text(
                    cx + 8,
                    cy - 8,
                    text=str(i + 1),
                    font=("Courier", 7),
                    fill=self.PATH_COL,
                )

    def _draw_goal(self, cv):
        """Goal as yellow crosshair star."""
        cx, cy = self._w2c(self._gx, self._gy)
        if self._in_canvas(cx, cy):
            s = 10
            for dx, dy in [
                (s, 0),
                (-s, 0),
                (0, s),
                (0, -s),
                (s, s),
                (-s, -s),
                (s, -s),
                (-s, s),
            ]:
                cv.create_line(
                    cx,
                    cy,
                    cx + dx // 2 if dx else cx,
                    cy + dy // 2 if dy else cy,
                    fill=self.GOAL_COL,
                    width=2,
                )
            cv.create_line(cx - s, cy, cx + s, cy, fill=self.GOAL_COL, width=2)
            cv.create_line(cx, cy - s, cx, cy + s, fill=self.GOAL_COL, width=2)
            cv.create_oval(
                cx - 3, cy - 3, cx + 3, cy + 3, fill=self.GOAL_COL, outline=""
            )
            cv.create_text(
                cx,
                cy - 18,
                text=f"GOAL ({self._gx:.1f}, {self._gy:.1f})",
                font=("Courier", 7, "bold"),
                fill=self.GOAL_COL,
            )

    def _draw_robot(self, cv):
        """Robot as blue circle with heading arrow. Always centered."""
        cx, cy = self.MAP_SIZE / 2, self.MAP_SIZE / 2
        yaw = self._ryaw
        r = 10

        # Body
        cv.create_oval(
            cx - r,
            cy - r,
            cx + r,
            cy + r,
            fill=self.ROBOT_COL,
            outline="#ffffff",
            width=1,
        )

        # Heading arrow
        L = 22
        ax = cx + L * math.cos(yaw)
        ay = cy - L * math.sin(yaw)
        cv.create_line(
            cx, cy, ax, ay, fill="#ffffff", width=2, arrow="last", arrowshape=(8, 10, 3)
        )

        # Position label
        cv.create_text(
            cx,
            cy + r + 10,
            text=f"({self._rx:.1f}, {self._ry:.1f})",
            font=("Courier", 7, "bold"),
            fill=self.ROBOT_COL,
        )

    def _draw_compass(self, cv):
        ox, oy = self.MAP_SIZE - 28, 28
        cv.create_text(
            ox, oy - 14, text="N", font=("Courier", 8, "bold"), fill=self.DIM
        )
        cv.create_line(ox, oy, ox, oy - 10, fill=self.DIM, width=1)
        cv.create_line(ox, oy, ox + 10, oy, fill=self.DIM, width=1)
        cv.create_text(
            ox + 16, oy, text="E", font=("Courier", 8, "bold"), fill=self.DIM
        )
