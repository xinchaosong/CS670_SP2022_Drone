#!/usr/bin/env python3

from tello.tello import Tello


def main():
    commands = ["command",
                "takeoff",
                "up 50",
                "delay 4",
                "land"]

    tello = Tello()
    tello.execute_commands(commands)
    tello.close()


if __name__ == "__main__":
    main()
