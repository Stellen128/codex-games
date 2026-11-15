r"""
闯关 4：面向对象电子宠物。

运行：
    C:\Users\pc\AppData\Local\Programs\Python\Python311\python.exe electronic_pet.py

界面采用连续对话流：
    - 状态只在启动和输入 7 时显示。
    - 每次输入与宠物回应紧挨着显示。
    - 自动求助触发时只输出一条系统通知。
"""

from __future__ import annotations

import ctypes
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable


@dataclass(frozen=True)
class PetSnapshot:
    name: str
    birth_time: str
    health: float
    seconds_since_feed: float
    emergency_rations: int
    alive: bool


class ElectronicPet:
    """具有生命、出生时间和自动求助能力的电子宠物。"""

    MAX_HEALTH = 100.0
    DECAY_PER_SECOND = 1.0
    AUTO_HELP_THRESHOLD = 30.0
    AUTO_HELP_RATIONS = 3

    def __init__(self, name: str) -> None:
        self.name = name
        self.birth_time = datetime.now()
        self.health = self.MAX_HEALTH
        self.emergency_rations = self.AUTO_HELP_RATIONS

        now = time.monotonic()
        self.last_feed_time = now
        self._last_health_update = now
        self._state_lock = threading.RLock()
        self._auto_events: list[str] = []
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._low_health_warned = False

    def _update_health_unlocked(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_health_update)
        self.health = max(
            0.0,
            self.health - elapsed * self.DECAY_PER_SECOND,
        )
        self._last_health_update = now

    @property
    def is_alive(self) -> bool:
        with self._state_lock:
            self._update_health_unlocked()
            return self.health > 0.0

    @property
    def seconds_since_feed(self) -> float:
        return max(0.0, time.monotonic() - self.last_feed_time)

    def snapshot(self) -> PetSnapshot:
        with self._state_lock:
            self._update_health_unlocked()
            return PetSnapshot(
                name=self.name,
                birth_time=self.birth_time.strftime("%Y-%m-%d %H:%M:%S"),
                health=self.health,
                seconds_since_feed=self.seconds_since_feed,
                emergency_rations=self.emergency_rations,
                alive=self.health > 0.0,
            )

    def feed(self) -> None:
        """手动喂食，不消耗自动求助口粮。"""
        with self._state_lock:
            now = time.monotonic()
            self.health = self.MAX_HEALTH
            self.last_feed_time = now
            self._last_health_update = now
            self._low_health_warned = False

    def check_auto_help(self) -> bool:
        """生命值低于危险线时，自动使用一份紧急口粮。"""
        with self._state_lock:
            self._update_health_unlocked()
            if self.health <= 0.0:
                return False

            if self.health > self.AUTO_HELP_THRESHOLD:
                self._low_health_warned = False
                return False

            if self.emergency_rations > 0:
                now = time.monotonic()
                self.emergency_rations -= 1
                self.health = self.MAX_HEALTH
                self.last_feed_time = now
                self._last_health_update = now
                self._low_health_warned = False
                self._auto_events.append(
                    "自动求助已启用，紧急口粮已恢复到 100，"
                    f"剩余 {self.emergency_rations} 份。"
                )
                return True

            if not self._low_health_warned:
                self._low_health_warned = True
                self._auto_events.append(
                    "紧急口粮已经耗尽，请立即输入 2 喂食。"
                )
            return False

    def take_auto_events(self) -> list[str]:
        with self._state_lock:
            events = self._auto_events[:]
            self._auto_events.clear()
            return events

    def start_auto_care(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._auto_care_loop,
            name="pet-auto-care",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_auto_care(self) -> None:
        self._monitor_stop.set()

    def _auto_care_loop(self) -> None:
        while not self._monitor_stop.is_set():
            self.check_auto_help()
            self._monitor_stop.wait(0.2)

    def ask_name(self) -> list[str]:
        return ["你叫什么名字呀？", f"我叫：{self.name}"]

    def ask_birthday(self) -> list[str]:
        birth_text = self.birth_time.strftime("%Y-%m-%d %H:%M:%S")
        return ["你的生日是什么时候啊？", f"我的生日是：{birth_text}"]

    def greet(self) -> list[str]:
        return ["你好，主人！", f"主人和我说话了！{self.name}好开心！"]

    def say_love(self) -> list[str]:
        return ["我爱你", "讨厌，人家还小！"]

    def sing(self) -> list[str]:
        return [
            f"{self.name}，给我唱支歌吧？",
            "叮叮当，叮叮当，铃儿响叮当",
        ]

    def respond(self, command: str) -> list[str]:
        actions: dict[str, Callable[[], list[str]]] = {
            "1": self.ask_name,
            "3": self.ask_birthday,
            "4": self.greet,
            "5": self.say_love,
            "6": self.sing,
        }

        with self._state_lock:
            self._update_health_unlocked()
            if self.health <= 0.0:
                return ["……没有回应。"]

            if command == "2":
                self.feed()
                return ["给你好吃的！", f"谢谢，{self.name}吃饱了！"]

            action = actions.get(command)
            if action is None:
                return ["这个命令不可用，请输入 0 到 7。"]

        return action()


class PetConsoleUI:
    """简洁的连续对话式控制台。"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    MUTED = "\033[38;5;244m"
    WHITE = "\033[38;5;255m"
    GREEN = "\033[38;5;114m"
    YELLOW = "\033[38;5;220m"
    RED = "\033[38;5;203m"

    def __init__(self, pet: ElectronicPet) -> None:
        self.pet = pet
        self.commands: queue.Queue[str | None] = queue.Queue()
        try:
            sys.stdout.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass
        self.use_ansi = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        if self.use_ansi:
            self._enable_windows_ansi()
            self._set_title()

    @staticmethod
    def _enable_windows_ansi() -> None:
        if os.name != "nt":
            return
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except (AttributeError, OSError):
            pass

    def _set_title(self) -> None:
        if os.name == "nt":
            try:
                ctypes.windll.kernel32.SetConsoleTitleW(
                    f"NEO PET CARE · {self.pet.name}"
                )
            except (AttributeError, OSError):
                pass

    def paint(self, text: str, color: str = "", bold: bool = False) -> str:
        if not self.use_ansi:
            return text
        style = color + (self.BOLD if bold else "")
        return f"{style}{text}{self.RESET}"

    @staticmethod
    def health_bar(health: float, width: int = 18) -> str:
        ratio = max(0.0, min(1.0, health / ElectronicPet.MAX_HEALTH))
        filled = round(width * ratio)
        return "█" * filled + "·" * (width - filled)

    def status_lines(self) -> list[str]:
        status = self.pet.snapshot()
        if status.health > 60.0:
            state, color = "正常", self.GREEN
        elif status.health > ElectronicPet.AUTO_HELP_THRESHOLD:
            state, color = "需要关注", self.YELLOW
        else:
            state, color = "危险", self.RED

        rations = "◆" * status.emergency_rations
        if status.emergency_rations == 0:
            rations = "已耗尽"

        return [
            self.paint(
                f"{status.name}  ·  {state}",
                self.WHITE,
                True,
            ),
            self.paint("生命值  ", self.MUTED)
            + self.paint(self.health_bar(status.health), color, True)
            + self.paint(f"  {status.health:5.1f} / 100", self.WHITE),
            self.paint("紧急口粮  ", self.MUTED)
            + self.paint(rations, self.YELLOW, True)
            + self.paint(
                f"   距上次喂食 {status.seconds_since_feed:.1f} 秒",
                self.MUTED,
            ),
        ]

    def print_banner(self) -> None:
        status = self.pet.snapshot()
        print()
        print(self.paint("NEO PET CARE", self.GREEN, True))
        print(self.paint("电子宠物 · 连续对话模式", self.MUTED))
        print(self.paint(f"出生时间：{status.birth_time}", self.MUTED))
        for line in self.status_lines():
            print("  " + line)
        print(self.paint("─" * 62, self.MUTED))
        print(
            self.paint("1", self.GREEN, True)
            + " 问名字   "
            + self.paint("2", self.GREEN, True)
            + " 喂食   "
            + self.paint("3", self.GREEN, True)
            + " 问生日   "
            + self.paint("4", self.GREEN, True)
            + " 打招呼"
        )
        print(
            self.paint("5", self.GREEN, True)
            + " 我爱你   "
            + self.paint("6", self.GREEN, True)
            + " 唱歌     "
            + self.paint("7", self.GREEN, True)
            + " 查看状态   "
            + self.paint("0", self.GREEN, True)
            + " 退出"
        )
        print(self.paint("─" * 62, self.MUTED))

    def print_message(
        self,
        speaker: str,
        lines: list[str],
        color: str = "",
    ) -> None:
        if not lines:
            return
        prefix = self.paint(f"{speaker} > ", color, True)
        print(prefix + lines[0])
        for line in lines[1:]:
            print(" " * (len(speaker) + 3) + line)

    def print_status(self) -> None:
        self.print_message("系统", self.status_lines(), self.GREEN)

    def print_prompt(self) -> None:
        print(self.paint("你 > ", self.GREEN, True), end="", flush=True)

    def clear_prompt_line(self) -> None:
        if self.use_ansi:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
        else:
            print()

    def _read_input(self) -> None:
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                self.commands.put(None)
                return
            self.commands.put(line)

    def print_auto_events(self) -> None:
        for event in self.pet.take_auto_events():
            self.print_message("系统", [event], self.YELLOW)

    def print_death(self) -> None:
        self.clear_prompt_line()
        self.print_auto_events()
        self.print_message(
            "系统",
            [
                f"{self.pet.name} 的生命值已归零。",
                "对不起，由于长时间没有喂食，您的宠物已经饿死了……",
            ],
            self.RED,
        )

    def run(self) -> None:
        self.pet.start_auto_care()
        reader = threading.Thread(
            target=self._read_input,
            name="pet-keyboard",
            daemon=True,
        )
        reader.start()

        self.print_banner()
        self.print_message(
            "系统",
            ["宠物已上线。自动求助处于开启状态。"],
            self.GREEN,
        )
        self.print_prompt()

        try:
            while True:
                if not self.pet.is_alive:
                    self.print_death()
                    break

                try:
                    command = self.commands.get(timeout=0.2)
                except queue.Empty:
                    continue

                if command is None:
                    self.clear_prompt_line()
                    self.print_message("系统", ["输入已关闭，游戏结束。"])
                    break

                command = command.strip()
                self.print_auto_events()

                if command == "0":
                    self.print_message("系统", ["游戏已退出。"])
                    break
                if command == "7":
                    self.print_status()
                elif command == "":
                    self.print_message("系统", ["请输入一个数字命令。"])
                else:
                    response = self.pet.respond(command)
                    self.print_message(self.pet.name, response)

                if not self.pet.is_alive:
                    self.print_death()
                    break

                self.print_prompt()
        except KeyboardInterrupt:
            self.clear_prompt_line()
            self.print_message("系统", ["游戏结束。"])
        finally:
            self.pet.stop_auto_care()


def create_pet_from_keyboard() -> ElectronicPet:
    print("请给你的宠物起个名字：")
    try:
        name = input().strip()
    except (EOFError, KeyboardInterrupt):
        name = ""

    if not name:
        name = "莉莉"
        print(f"没有输入名字，使用默认名字：{name}")
    return ElectronicPet(name)


if __name__ == "__main__":
    pet = create_pet_from_keyboard()
    PetConsoleUI(pet).run()
