import time


class StopWatch:
    """ very basic stop-watch class for benchmarking """

    def __init__(self):
        self._running = False
        self._start = 0.0
        self._elapsed = 0.0

    def start(self):
        assert self._running is False
        self._start = time.perf_counter()
        self._running = True

    def stop(self):
        assert self._running is True
        self._elapsed += time.perf_counter() - self._start
        self._running = False

    def elapsed(self):
        assert self._running is False
        return self._elapsed

    def reset(self):
        assert self._running is False
        self._elapsed = 0.0
