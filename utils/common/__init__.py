import time
import tracemalloc


# def measure_memory(func):
#     # Taken, but modified: from: https://medium.com/survata-engineering-blog/monitoring-memory-usage-of-a-running-python-program-49f027e3d1ba
#     def inner(*args, **kwargs):
#         tracemalloc.start()
#         ret = func(*args, **kwargs)
#         current, peak = tracemalloc.get_traced_memory()
#         print(f"\n\033[37mFunction Name       :\033[35;1m {func.__name__}\033[0m")
#         print(f"\033[37mCurrent memory usage:\033[36m {current / 10 ** 6}MB\033[0m")
#         print(f"\033[37mPeak                :\033[36m {peak / 10 ** 6}MB\033[0m")
#         tracemalloc.stop()
#         return ret
#
#     return inner


class RuntimeStats:
    def __init__(self, runtime_stats):
        self.start = time.time()
        self.runtime_stats = runtime_stats

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = 'The function took {time} seconds to complete'
        self.runtime_stats.update({"start": self.start,
                                   "end": end,
                                   "runtime": runtime})
        print(msg.format(time=runtime))


class StatTrak:

    def __init__(self, list_of_partials):
        self.partials = list_of_partials

    def benchmark(self):
        pass
