from __future__ import annotations

import threading
import time
from typing import Any


class Ticker(threading.Thread):  # pragma nocover
    def __init__(self, period: Any) -> None:
        super().__init__()

        self.sem = threading.Semaphore(0)
        self.done = False
        self.period = period
        print("in init", period)

    def wait(self) -> None:
        self.sem.acquire()

    def run(self) -> None:
        print("in run", self.period)
        start = time.time()
        stop = start
        while not self.done:
            stop += self.period
            # print('sleeping for', stop-start)
            start = time.time()
            time.sleep(stop - start)
            self.sem.release()

    def stop(self) -> None:
        self.done = True
        self.join()


if __name__ == "__main__":  # pragma nocover
    t = Ticker(0.1)

    tprev = time.time()
    tmax: float = 0
    tsum: float = 0
    N = 100

    t.start()
    for i in range(100):
        t.wait()
        tnow = time.time()
        dt = tnow - tprev
        tprev = tnow
        terr = abs(dt - 0.1) * 1000  # error in ms
        if i > 0:
            tsum += terr
            tmax = max(tmax, terr)
        print(".", end="", flush=True)
    t.stop()
    print(f"\nmean = {tsum / (N - 1):.2f}, max = {tmax:.2f}")
