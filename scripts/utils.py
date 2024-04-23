import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

def N_gpu_util_timer(self):
    for n in range(10):
        GPUs = GPUtil.getGPUs()
        gpu_load = GPUs[0].load
        Graph.gpu_y.append(gpu_load)
        time.sleep(1)
    print(Graph.gpu_y)
    print('N gpu done')

if __name__ == '__main__':
    # Instantiate monitor with a 10-second delay between updates
    monitor = Monitor(10)

    # Train, etc.
    time.sleep(60)

    monitor.stop()

