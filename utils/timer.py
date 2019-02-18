import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name
        self.t_start = time.time()

    def __enter__(self):
        self.tic()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.t_start))

    def tic(self):
        self.t_start = time.time()

    def toc(self):
        return time.time() - self.t_start
