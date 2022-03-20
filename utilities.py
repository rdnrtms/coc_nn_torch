import sys
import time
import numpy as np

from torch import tensor
from functools import wraps
from collections.abc import Callable

debug = False


# This writes a tensor in a readable format to a file
def write_tensor(filename: str, T: tensor, oscNum: int, invNum: int) -> None:
    global debug

    if not debug:
        return

    with open(filename, 'w') as f:
        for i in range(T.size(0)):
            for j in range(T.size(1)):
                if i % invNum == 0 and i > 0 and j == 0:
                    f.write('-' * (2*invNum*oscNum + 2*(oscNum-1) - 1))
                    f.write('\n')
                if j % invNum == 0 and j > 0:
                    f.write('| ')

                f.write("%-6.3f" % T[i, j])
                f.write(' ')

                if j == T.size(1) - 1:
                    f.write('\n')

    return


# Decorator that reports the execution time of func
def timeit(string: str) -> None:
    def timeit_inner(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> None:
            start = time.time()
            func(*args, **kwargs)
            end = time.time()

            print(string % (end-start))

            return

        return wrapper
    return timeit_inner


# This calculate the size of a give variable
def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# Progress bar class
class ProgressBar():
    # This initiate a progress bar
    def __init__(self, tEnd: float, progressbarWidth: int = 20,
                 string: str = "Simulation: ") -> None:
        self.progressbarWidth = progressbarWidth
        self.string = string
        self.tEnd = tEnd
        self.time_ind = 0
        self.finished = False

        self.__set_progressbar_cps()

        pass

    # This is used for the progress bar to create check points for the bar
    def __set_progressbar_cps(self) -> None:
        self.time_cps = np.linspace(0.0, self.tEnd, self.progressbarWidth)

        return

    # Modifies the time checkpoints if tEnd changed
    def modify_CPs(self, tEnd: float) -> None:
        self.tEnd = tEnd
        self.__set_progressbar_cps()

        return

    # This sets the string before the progress bar
    # Note that this will work from the next reset of the progress bar
    def set_string(self, string: str) -> None:
        self.string = string

        return

    # Resets the progressbar to its initial state
    def reset(self) -> None:
        self.time_ind = 0
        self.finished = False

        return

    # This starts the progress bar
    def start(self) -> None:
        sys.stdout.write("%s[%s]" %
                         (self.string, " " * self.progressbarWidth))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.progressbarWidth + 1))

        return

    # This wraps up the progress bar
    def __end(self) -> None:
        self.finished = True

        sys.stdout.write("-")
        sys.stdout.flush()
        sys.stdout.write("] ")

        return

    # This updates the progressbar accordingly
    def update(self, t: float) -> None:
        if (self.time_ind < self.progressbarWidth-1
                and t >= self.time_cps[self.time_ind]):
            sys.stdout.write("-")
            sys.stdout.flush()
            self.time_ind += 1

        if t >= self.tEnd and not self.finished:
            self.__end()
