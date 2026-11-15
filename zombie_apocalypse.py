r"""
闯关 7：生化危机 —— T 病毒传播模拟

无控制台运行：
    双击“启动生化危机.lnk”

也可以使用完整 Python 路径启动：
    C:\Users\pc\AppData\Local\Programs\Python\Python311\pythonw.exe zombie_apocalypse.py

命令行调试或无界面测试：
    C:\Users\pc\AppData\Local\Programs\Python\Python311\python.exe zombie_apocalypse.py

只依赖 Python 标准库，不需要安装第三方软件包。

快捷键：
    Space  暂停 / 继续
    R      重新开始
    B      基础版
    U      升级版
"""

from __future__ import annotations

import argparse
import ctypes
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import Literal


Mode = Literal["basic", "upgraded"]


def torus_delta(value: float, size: float) -> float:
    """把坐标差转换为环面上的最短位移，范围为 [-size/2, size/2)。"""
    return (value + size / 2.0) % size - size / 2.0


@dataclass
class InfectionEvent:
    sim_time: float
    victim: int
    source: int


class Simulation:
    """与界面无关的 T 病毒传播模型。"""

    HUMAN_SPEED_BASIC = 2.0
    HUMAN_SPEED_FLEE = 2.0
    ZOMBIE_SPEED_BASIC = 2.0
    ZOMBIE_SPEED_CHASE = 3.0
    INFECTION_RADIUS = 2.5

    def __init__(
        self,
        width: float = 100.0,
        height: float = 70.0,
        population: int = 100,
        seed: int | None = None,
        mode: Mode = "basic",
    ) -> None:
        if population < 2:
            raise ValueError("人口数量至少需要为 2")
        if mode not in ("basic", "upgraded"):
            raise ValueError("mode 必须是 'basic' 或 'upgraded'")

        self.width = width
        self.height = height
        self.population = population
        self.seed = seed if seed is not None else random.randrange(1, 2**31)
        self.rng = random.Random(self.seed)
        self.mode: Mode = mode

        self.positions = [
            (self.rng.uniform(0.0, width), self.rng.uniform(0.0, height))
            for _ in range(population)
        ]
        self.headings = [self.rng.uniform(0.0, 2.0 * math.pi) for _ in range(population)]
        self.infected = [False] * population
        self.patient_zero = self.rng.randrange(population)
        self.infected[self.patient_zero] = True

        self.infected_count = 1
        self.elapsed = 0.0
        self.events = [
            InfectionEvent(sim_time=0.0, victim=self.patient_zero, source=self.patient_zero)
        ]
        self.flash_until: dict[int, float] = {self.patient_zero: 1.2}

    @property
    def living_count(self) -> int:
        return self.population - self.infected_count

    @property
    def finished(self) -> bool:
        return self.infected_count >= self.population

    @property
    def infection_rate(self) -> float:
        return self.infected_count / self.population

    def torus_offset(
        self, source: tuple[float, float], target: tuple[float, float]
    ) -> tuple[float, float]:
        """返回 source 指向 target 的环面最短位移。"""
        return (
            torus_delta(target[0] - source[0], self.width),
            torus_delta(target[1] - source[1], self.height),
        )

    def torus_distance(
        self, first: tuple[float, float], second: tuple[float, float]
    ) -> float:
        dx, dy = self.torus_offset(first, second)
        return math.hypot(dx, dy)

    def _nearest(
        self, index: int, candidates: list[int]
    ) -> tuple[int | None, float, float, float]:
        """在环面空间内找到最近候选，返回编号、dx、dy、距离。"""
        nearest_index: int | None = None
        nearest_dx = 0.0
        nearest_dy = 0.0
        nearest_distance = math.inf
        position = self.positions[index]

        for candidate in candidates:
            dx, dy = self.torus_offset(position, self.positions[candidate])
            distance = math.hypot(dx, dy)
            if distance < nearest_distance:
                nearest_index = candidate
                nearest_dx = dx
                nearest_dy = dy
                nearest_distance = distance

        return nearest_index, nearest_dx, nearest_dy, nearest_distance

    def _move(self, index: int, vx: float, vy: float, speed: float, dt: float) -> None:
        length = math.hypot(vx, vy)
        if length < 1e-12:
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            vx, vy, length = math.cos(angle), math.sin(angle), 1.0

        x, y = self.positions[index]
        x += speed * dt * vx / length
        y += speed * dt * vy / length
        self.positions[index] = (x % self.width, y % self.height)

    def _random_walk(self, index: int, dt: float) -> None:
        # 方向持续变化，但不会每帧完全乱跳，从而形成自然的随机游走轨迹。
        self.headings[index] += self.rng.gauss(0.0, 0.8 * dt)
        angle = self.headings[index]
        speed = (
            self.ZOMBIE_SPEED_BASIC
            if self.infected[index]
            else self.HUMAN_SPEED_BASIC
        )
        self._move(index, math.cos(angle), math.sin(angle), speed, dt)

    def _move_upgraded(self, dt: float) -> None:
        living = [i for i in range(self.population) if not self.infected[i]]
        zombies = [i for i in range(self.population) if self.infected[i]]
        if not living or not zombies:
            return

        human_directions: dict[int, tuple[float, float]] = {}
        zombie_directions: dict[int, tuple[float, float]] = {}

        # 使用移动前的状态计算目标，保证同一帧内所有单位同时决策。
        for human in living:
            _, dx, dy, _ = self._nearest(human, zombies)
            human_directions[human] = (-dx, -dy)

        for zombie in zombies:
            _, dx, dy, _ = self._nearest(zombie, living)
            zombie_directions[zombie] = (dx, dy)

        for human, (vx, vy) in human_directions.items():
            self._move(human, vx, vy, self.HUMAN_SPEED_FLEE, dt)

        for zombie, (vx, vy) in zombie_directions.items():
            self._move(zombie, vx, vy, self.ZOMBIE_SPEED_CHASE, dt)

    def _spread_infection(self) -> list[int]:
        zombies = [i for i in range(self.population) if self.infected[i]]
        newly_infected: list[int] = []

        for victim in range(self.population):
            if self.infected[victim]:
                continue

            source, _, _, distance = self._nearest(victim, zombies)
            if source is not None and distance < self.INFECTION_RADIUS:
                self.infected[victim] = True
                self.infected_count += 1
                newly_infected.append(victim)
                self.events.append(
                    InfectionEvent(self.elapsed, victim, source)
                )
                self.flash_until[victim] = self.elapsed + 1.2

        return newly_infected

    def tick(self, dt: float) -> list[int]:
        """推进一次仿真，返回本帧新感染的个体编号。"""
        if dt <= 0.0 or self.finished:
            return []

        if self.mode == "basic":
            for index in range(self.population):
                self._random_walk(index, dt)
        else:
            self._move_upgraded(dt)

        self.elapsed += dt
        return self._spread_infection()


class ZombieApocalypseApp:
    """Tkinter 动画界面。"""

    BG = "#061017"
    PANEL = "#0b1820"
    PANEL_LIGHT = "#10232d"
    TEXT = "#e7f5ef"
    MUTED = "#87a69f"
    GREEN = "#62f58d"
    GREEN_DARK = "#173f2a"
    RED = "#ff365c"
    RED_DARK = "#451823"
    GRID = "#142932"
    BORDER = "#294854"
    CYAN = "#59d6d3"

    def __init__(self, root, mode: Mode = "basic", seed: int | None = None) -> None:
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.mode: Mode = mode
        self.seed = seed
        self.simulation = Simulation(seed=seed, mode=mode)
        self.paused = False
        self.speed_scale = 1.0
        self.last_wall_time = time.perf_counter()
        self.last_status_update = 0.0
        self.log_lines: list[str] = [
            "00:00  封锁区建立，100 人在区域内随机走动",
            "00:00  发现首例 T 病毒感染者",
        ]
        self._mode_buttons: dict[Mode, object] = {}

        self._configure_window()
        self._build_layout()
        self._refresh_mode_buttons()
        self._draw()
        self._schedule_tick()

    def _configure_window(self) -> None:
        tk = self.tk
        self.root.title("闯关 7 · 生化危机：T 病毒传播模拟")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1180, max(900, screen_width - 100))
        window_height = min(720, max(620, screen_height - 120))
        x = max(20, (screen_width - window_width) // 2)
        y = max(20, (screen_height - window_height) // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(900, 620)
        self.root.configure(bg=self.BG)
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.root.bind("<space>", lambda _event: self.toggle_pause())
        self.root.bind("<Key-r>", lambda _event: self.restart())
        self.root.bind("<Key-R>", lambda _event: self.restart())
        self.root.bind("<Key-b>", lambda _event: self.set_mode("basic"))
        self.root.bind("<Key-B>", lambda _event: self.set_mode("basic"))
        self.root.bind("<Key-u>", lambda _event: self.set_mode("upgraded"))
        self.root.bind("<Key-U>", lambda _event: self.set_mode("upgraded"))

    def _build_layout(self) -> None:
        tk = self.tk
        outer = tk.Frame(self.root, bg=self.BG, padx=18, pady=14)
        outer.pack(fill="both", expand=True)

        header = tk.Frame(outer, bg=self.BG)
        header.pack(fill="x", pady=(0, 12))

        title_box = tk.Frame(header, bg=self.BG)
        title_box.pack(side="left")
        tk.Label(
            title_box,
            text="生化危机",
            bg=self.BG,
            fg=self.TEXT,
            font=("Microsoft YaHei UI", 22, "bold"),
        ).pack(side="left")
        tk.Label(
            title_box,
            text="  T 病毒传播模拟",
            bg=self.BG,
            fg=self.CYAN,
            font=("Microsoft YaHei UI", 13),
        ).pack(side="left", pady=(8, 0))

        self.badge = tk.Label(
            header,
            text="基础版 · 随机行走",
            bg=self.GREEN_DARK,
            fg=self.GREEN,
            padx=13,
            pady=6,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.badge.pack(side="right")

        body = tk.Frame(outer, bg=self.BG)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=300)
        body.grid_rowconfigure(0, weight=1)

        world_frame = tk.Frame(
            body, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1
        )
        world_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.canvas = tk.Canvas(
            world_frame,
            bg="#08131a",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, bg=self.PANEL, width=300)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

    def _build_sidebar(self, parent) -> None:
        tk = self.tk

        status = tk.Frame(parent, bg=self.PANEL_LIGHT, padx=18, pady=15)
        status.pack(fill="x", padx=14, pady=(14, 10))

        tk.Label(
            status,
            text="实时态势",
            bg=self.PANEL_LIGHT,
            fg=self.MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w")

        numbers = tk.Frame(status, bg=self.PANEL_LIGHT)
        numbers.pack(fill="x", pady=(7, 10))

        living_box = tk.Frame(numbers, bg=self.PANEL_LIGHT)
        living_box.pack(side="left", fill="x", expand=True)
        tk.Label(
            living_box, text="存活人类", bg=self.PANEL_LIGHT, fg=self.MUTED
        ).pack(anchor="w")
        self.living_label = tk.Label(
            living_box,
            text="99",
            bg=self.PANEL_LIGHT,
            fg=self.GREEN,
            font=("Consolas", 25, "bold"),
        )
        self.living_label.pack(anchor="w")

        zombie_box = tk.Frame(numbers, bg=self.PANEL_LIGHT)
        zombie_box.pack(side="left", fill="x", expand=True)
        tk.Label(
            zombie_box, text="感染丧尸", bg=self.PANEL_LIGHT, fg=self.MUTED
        ).pack(anchor="w")
        self.zombie_label = tk.Label(
            zombie_box,
            text="1",
            bg=self.PANEL_LIGHT,
            fg=self.RED,
            font=("Consolas", 25, "bold"),
        )
        self.zombie_label.pack(anchor="w")

        self.progress_canvas = tk.Canvas(
            status,
            height=6,
            bg="#1e343c",
            highlightthickness=0,
            bd=0,
        )
        self.progress_canvas.pack(fill="x")
        self.progress_fill = self.progress_canvas.create_rectangle(
            0, 0, 1, 6, fill=self.RED, outline=""
        )
        self.rate_label = tk.Label(
            status,
            text="感染率  1.0%",
            bg=self.PANEL_LIGHT,
            fg=self.MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        self.rate_label.pack(anchor="e", pady=(5, 0))

        tk.Label(
            parent,
            text="模拟模式",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", padx=16, pady=(4, 6))

        mode_row = tk.Frame(parent, bg=self.PANEL)
        mode_row.pack(fill="x", padx=14)
        self._mode_buttons["basic"] = tk.Button(
            mode_row,
            text="基础版",
            command=lambda: self.set_mode("basic"),
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=8,
            pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._mode_buttons["basic"].pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._mode_buttons["upgraded"] = tk.Button(
            mode_row,
            text="升级版",
            command=lambda: self.set_mode("upgraded"),
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=8,
            pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self._mode_buttons["upgraded"].pack(side="left", fill="x", expand=True, padx=(5, 0))

        self.mode_hint = tk.Label(
            parent,
            text="所有人随机行走，感染后继续随机接触",
            bg=self.PANEL,
            fg=self.MUTED,
            justify="left",
            wraplength=262,
            font=("Microsoft YaHei UI", 9),
        )
        self.mode_hint.pack(anchor="w", padx=16, pady=(6, 11))

        control_row = tk.Frame(parent, bg=self.PANEL)
        control_row.pack(fill="x", padx=14)
        self.pause_button = tk.Button(
            control_row,
            text="暂停",
            command=self.toggle_pause,
            relief="flat",
            bd=0,
            cursor="hand2",
            bg=self.PANEL_LIGHT,
            fg=self.TEXT,
            activebackground="#17313c",
            activeforeground=self.TEXT,
            pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.pause_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        restart_button = tk.Button(
            control_row,
            text="重新开始",
            command=self.restart,
            relief="flat",
            bd=0,
            cursor="hand2",
            bg=self.PANEL_LIGHT,
            fg=self.TEXT,
            activebackground="#17313c",
            activeforeground=self.TEXT,
            pady=8,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        restart_button.pack(side="left", fill="x", expand=True, padx=(5, 0))

        speed_head = tk.Frame(parent, bg=self.PANEL)
        speed_head.pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(
            speed_head,
            text="时间倍速",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(side="left")
        self.speed_label = tk.Label(
            speed_head,
            text="1.0×",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Consolas", 9, "bold"),
        )
        self.speed_label.pack(side="right")
        self.speed_slider = tk.Scale(
            parent,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient="horizontal",
            showvalue=False,
            command=self._set_speed,
            bg=self.PANEL,
            fg=self.TEXT,
            troughcolor="#1c3540",
            activebackground=self.CYAN,
            highlightthickness=0,
            bd=0,
            sliderrelief="flat",
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack(fill="x", padx=14, pady=(3, 8))

        legend = tk.Frame(parent, bg=self.PANEL)
        legend.pack(fill="x", padx=16, pady=(1, 7))
        tk.Label(
            legend,
            text="●",
            bg=self.PANEL,
            fg=self.GREEN,
            font=("Arial", 14),
        ).pack(side="left")
        tk.Label(
            legend, text="活人", bg=self.PANEL, fg=self.MUTED, font=("Microsoft YaHei UI", 9)
        ).pack(side="left", padx=(2, 15))
        tk.Label(
            legend,
            text="●",
            bg=self.PANEL,
            fg=self.RED,
            font=("Arial", 14),
        ).pack(side="left")
        tk.Label(
            legend, text="丧尸", bg=self.PANEL, fg=self.MUTED, font=("Microsoft YaHei UI", 9)
        ).pack(side="left", padx=(2, 0))
        tk.Label(
            legend,
            text="感染半径 2.5",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(side="right")

        tk.Label(
            parent,
            text="感染记录",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", padx=16, pady=(2, 5))
        self.log_box = tk.Text(
            parent,
            width=30,
            height=5,
            bg="#08131a",
            fg="#b5cec8",
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
            wrap="word",
            font=("Microsoft YaHei UI", 8),
            state="disabled",
            cursor="arrow",
        )
        self.log_box.pack(fill="both", expand=True, padx=14, pady=(0, 9))
        self._refresh_log()

        tk.Label(
            parent,
            text="Space 暂停/继续   R 重开   B/U 切换模式",
            bg=self.PANEL,
            fg="#58747a",
            font=("Microsoft YaHei UI", 8),
        ).pack(side="bottom", pady=(0, 12))

    def _refresh_mode_buttons(self) -> None:
        for name, button in self._mode_buttons.items():
            selected = name == self.mode
            button.configure(
                bg=self.GREEN_DARK if selected else self.PANEL_LIGHT,
                fg=self.GREEN if selected else self.MUTED,
                activebackground=self.GREEN_DARK if selected else "#17313c",
                activeforeground=self.GREEN if selected else self.TEXT,
            )

        if self.mode == "basic":
            self.badge.configure(
                text="基础版 · 随机行走", bg=self.GREEN_DARK, fg=self.GREEN
            )
            self.mode_hint.configure(
                text="所有人随机行走，感染后继续随机接触"
            )
        else:
            self.badge.configure(
                text="升级版 · 逃逸追击", bg=self.RED_DARK, fg=self.RED
            )
            self.mode_hint.configure(
                text="人类速度 2，逃离最近丧尸\n丧尸速度 3，追逐最近人类"
            )

    def _set_speed(self, value: str) -> None:
        self.speed_scale = float(value)
        self.speed_label.configure(text=f"{self.speed_scale:.1f}×")

    def _world_rect(self) -> tuple[float, float, float, float]:
        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 300)
        margin = 28
        scale = min(
            (width - 2 * margin) / self.simulation.width,
            (height - 2 * margin) / self.simulation.height,
        )
        world_width = self.simulation.width * scale
        world_height = self.simulation.height * scale
        left = (width - world_width) / 2
        top = (height - world_height) / 2
        return left, top, scale, world_height

    def _draw(self) -> None:
        tk = self.tk
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 300)
        left, top, scale, world_height = self._world_rect()
        right = left + self.simulation.width * scale
        bottom = top + world_height

        # 细网格帮助观察坐标回绕。
        for x in range(0, int(self.simulation.width) + 1, 10):
            px = left + x * scale
            canvas.create_line(
                px, top, px, bottom, fill=self.GRID, width=1, dash=(2, 8)
            )
        for y in range(0, int(self.simulation.height) + 1, 10):
            py = top + y * scale
            canvas.create_line(
                left, py, right, py, fill=self.GRID, width=1, dash=(2, 8)
            )

        canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            outline=self.BORDER,
            width=2,
        )

        # 四边的同色短线提示：从一边离开会从对面进入。
        tick_length = 8
        for x in (left, right):
            for offset in range(0, int(world_height) + 1, 38):
                canvas.create_line(
                    x,
                    top + offset,
                    x,
                    min(top + offset + tick_length, bottom),
                    fill=self.CYAN,
                    width=1,
                )
        for y in (top, bottom):
            for offset in range(0, int(right - left) + 1, 38):
                canvas.create_line(
                    left + offset,
                    y,
                    min(left + offset + tick_length, right),
                    y,
                    fill=self.CYAN,
                    width=1,
                )

        radius = max(3.5, min(5.5, scale * 0.38))
        show_halos = self.simulation.infected_count <= 18
        halo_radius = self.simulation.INFECTION_RADIUS * scale

        for index, (x, y) in enumerate(self.simulation.positions):
            px = left + x * scale
            py = top + y * scale

            if self.simulation.infected[index]:
                if show_halos:
                    canvas.create_oval(
                        px - halo_radius,
                        py - halo_radius,
                        px + halo_radius,
                        py + halo_radius,
                        outline="#57212b",
                        width=1,
                        dash=(2, 5),
                    )
                canvas.create_oval(
                    px - radius - 2,
                    py - radius - 2,
                    px + radius + 2,
                    py + radius + 2,
                    fill="#4d1421",
                    outline="",
                )
                canvas.create_oval(
                    px - radius,
                    py - radius,
                    px + radius,
                    py + radius,
                    fill=self.RED,
                    outline="#ff8094",
                    width=1,
                )
            else:
                canvas.create_oval(
                    px - radius,
                    py - radius,
                    px + radius,
                    py + radius,
                    fill=self.GREEN,
                    outline="#b7ffca",
                    width=1,
                )

            if self.simulation.flash_until.get(index, 0.0) > self.simulation.elapsed:
                flash_radius = radius + 7
                canvas.create_oval(
                    px - flash_radius,
                    py - flash_radius,
                    px + flash_radius,
                    py + flash_radius,
                    outline="#ff718a",
                    width=2,
                )

        mode_text = "基础版 · 随机行走" if self.mode == "basic" else "升级版 · 逃逸追击"
        canvas.create_rectangle(
            left + 12,
            top + 11,
            left + 170,
            top + 37,
            fill="#08131a",
            outline="",
        )
        canvas.create_text(
            left + 17,
            top + 24,
            anchor="w",
            text=f"{self._format_time(self.simulation.elapsed)}   {mode_text}",
            fill="#9cbab4",
            font=("Consolas", 9),
        )
        canvas.create_text(
            (left + right) / 2,
            bottom + 14,
            text="周期性边界：从一侧离开，会立即从对面出现",
            fill="#5f7d7c",
            font=("Microsoft YaHei UI", 9),
        )

        if self.paused and not self.simulation.finished:
            canvas.create_rectangle(
                left, top, right, bottom, fill="#061017", stipple="gray25", outline=""
            )
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="已暂停",
                fill=self.TEXT,
                font=("Microsoft YaHei UI", 23, "bold"),
            )

        if self.simulation.finished:
            overlay_left = left + (right - left) * 0.14
            overlay_right = right - (right - left) * 0.14
            overlay_top = top + (bottom - top) * 0.30
            overlay_bottom = bottom - (bottom - top) * 0.28
            canvas.create_rectangle(
                overlay_left,
                overlay_top,
                overlay_right,
                overlay_bottom,
                fill="#160c11",
                outline=self.RED,
                width=2,
            )
            canvas.create_text(
                (left + right) / 2,
                (overlay_top + overlay_bottom) / 2 - 18,
                text="封锁区已沦陷",
                fill=self.RED,
                font=("Microsoft YaHei UI", 24, "bold"),
            )
            canvas.create_text(
                (left + right) / 2,
                (overlay_top + overlay_bottom) / 2 + 24,
                text=f"100 人全部感染  ·  用时 {self._format_time(self.simulation.elapsed)}",
                fill=self.MUTED,
                font=("Microsoft YaHei UI", 10),
            )

        self._refresh_status()

    def _refresh_status(self) -> None:
        sim = self.simulation
        self.living_label.configure(text=str(sim.living_count))
        self.zombie_label.configure(text=str(sim.infected_count))
        self.rate_label.configure(text=f"感染率  {sim.infection_rate * 100:.1f}%")
        self.progress_canvas.update_idletasks()
        width = max(self.progress_canvas.winfo_width(), 1)
        self.progress_canvas.coords(
            self.progress_fill, 0, 0, max(1, width * sim.infection_rate), 6
        )

    def _refresh_log(self) -> None:
        visible = self.log_lines[-9:]
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("end", "\n".join(visible))
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    @staticmethod
    def _format_time(seconds: float) -> str:
        total_seconds = int(seconds)
        return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

    def _record_events(self, newly_infected: list[int]) -> None:
        for victim in newly_infected:
            self.log_lines.append(
                f"{self._format_time(self.simulation.elapsed)}  #{victim + 1:03d} 被感染"
            )
        if newly_infected:
            self._refresh_log()

    def _tick(self) -> None:
        now = time.perf_counter()
        wall_dt = now - self.last_wall_time
        self.last_wall_time = now

        if not self.paused and not self.simulation.finished:
            dt = min(max(wall_dt, 0.0), 0.05) * self.speed_scale
            newly_infected = self.simulation.tick(dt)
            self._record_events(newly_infected)
            if self.simulation.finished:
                self.log_lines.append(
                    f"{self._format_time(self.simulation.elapsed)}  最后一名人类感染，模拟结束"
                )
                self._refresh_log()
                self.pause_button.configure(text="已结束")

        self._draw()
        self._schedule_tick()

    def _schedule_tick(self) -> None:
        self.root.after(33, self._tick)

    def toggle_pause(self) -> None:
        if self.simulation.finished:
            return
        self.paused = not self.paused
        self.pause_button.configure(text="继续" if self.paused else "暂停")
        self.last_wall_time = time.perf_counter()
        self._draw()

    def restart(self) -> None:
        self.simulation = Simulation(seed=None, mode=self.mode)
        self.paused = False
        self.last_wall_time = time.perf_counter()
        self.pause_button.configure(text="暂停")
        self.log_lines = [
            "00:00  封锁区建立，100 人在区域内随机走动",
            "00:00  发现首例 T 病毒感染者",
        ]
        self._refresh_log()
        self._draw()

    def set_mode(self, mode: Mode) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self.simulation = Simulation(seed=None, mode=mode)
        self.paused = False
        self.last_wall_time = time.perf_counter()
        self.pause_button.configure(text="暂停")
        mode_name = "基础版" if mode == "basic" else "升级版"
        self.log_lines = [
            f"00:00  {mode_name}启动，100 人进入封锁区",
            "00:00  发现首例 T 病毒感染者",
        ]
        self._refresh_mode_buttons()
        self._refresh_log()
        self._draw()


def run_headless(mode: Mode, seed: int, duration: float) -> None:
    """命令行测试模式；不打开窗口，便于验证仿真逻辑。"""
    simulation = Simulation(seed=seed, mode=mode)
    dt = 1.0 / 30.0
    next_report = 0.0
    mode_name = "基础版" if mode == "basic" else "升级版"
    print(f"{mode_name}开始，随机种子={seed}")

    while simulation.elapsed < duration and not simulation.finished:
        simulation.tick(dt)
        if simulation.elapsed >= next_report:
            print(
                f"t={simulation.elapsed:6.1f}s  "
                f"人类={simulation.living_count:3d}  丧尸={simulation.infected_count:3d}"
            )
            next_report += 5.0

    result = "全部感染" if simulation.finished else "在时限内仍有幸存者"
    print(
        f"结束：{result}，模拟时间={simulation.elapsed:.1f}s，"
        f"最终感染率={simulation.infection_rate * 100:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="闯关 7：T 病毒传播模拟")
    parser.add_argument(
        "--mode",
        choices=("basic", "upgraded"),
        default="basic",
        help="basic=随机行走，upgraded=逃逸追击（默认 basic）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不显示动画，只在终端输出仿真结果",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="无界面模式最长模拟秒数（默认 180）",
    )
    args = parser.parse_args()

    if args.headless:
        run_headless(
            mode=args.mode,
            seed=args.seed if args.seed is not None else 20261114,
            duration=args.duration,
        )
        return

    # Windows 默认会再次缩放 Tk 窗口，在 125%/150% 屏幕上可能使界面超出屏幕。
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass

    import tkinter as tk

    root = tk.Tk()
    ZombieApocalypseApp(root, mode=args.mode, seed=args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
