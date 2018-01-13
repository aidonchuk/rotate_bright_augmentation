import time
from threading import BoundedSemaphore, Timer


class RatedSemaphore(BoundedSemaphore):

    def __init__(self, value=1, period=1):
        BoundedSemaphore.__init__(self, value)
        t = Timer(period, self._add_token_loop,
                  kwargs=dict(time_delta=float(period) / value))
        t.daemon = True
        t.start()

    def _add_token_loop(self, time_delta):
        while True:
            try:
                BoundedSemaphore.release(self)
            except ValueError:
                pass
            time.sleep(time_delta)

    def release(self):
        pass
