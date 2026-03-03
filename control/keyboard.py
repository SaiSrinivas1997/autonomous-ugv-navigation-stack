import sys
import termios
import tty
import select
import atexit


class Keyboard:
    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        atexit.register(self.restore)

    def restore(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        key = None

        if rlist:
            c1 = sys.stdin.read(1)
            if c1 == "\x1b":  # escape sequence
                c2 = sys.stdin.read(1)
                c3 = sys.stdin.read(1)
                key = c1 + c2 + c3
            else:
                key = c1

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
