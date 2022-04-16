import time


class SecInterval:
    # --- Timer object ---
    def __init__(self):
        self.last_time = None
        self.on = False

    def start(self):
        # start counting
        self.last_time = time.time()
        self.on = True

    def sec_loop(self):
        # --- return current value ---
        if self.on:
            if time.time() - self.last_time >= 1:
                return True

    def stop(self):
        # --- stop counting ---
        self.__init__()