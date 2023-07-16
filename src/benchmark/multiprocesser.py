import os
import time
from multiprocessing import Manager, Process, cpu_count


class ListMultiProcessor:
    """
    Input: List, Function
    Output: List of outputs
    """

    def __init__(self, input_list, function, num_threads=cpu_count(), verbose=True):
        self.verbose = verbose
        self.input_list = (
            input_list if not self.verbose else list(enumerate(input_list, start=1))
        )
        self.function = function
        self.num_threads = num_threads

    # TODO: Deprecated
    # @staticmethod
    # def _split(a, n):
    #    k, m = divmod(len(a), n)
    #    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def _process(self, input, output, verbose=True, initial_length=None):
        def _pop(array, index=0, default=None):
            try:
                return array.pop(index)
            except IndexError:
                return default

        if verbose:  # Verbose case
            print(f"Process: {os.getpid()} starting.", flush=True)
            while index_element := _pop(
                input
            ):  # Note: Walrus operator (:=) requires Python >= 3.8
                i, element = index_element
                output.append(self.function(element))
                if (
                    initial_length
                    and i
                    % (
                        int(initial_length * 0.1)
                        if (int(initial_length * 0.1)) > 0
                        else 1
                    )
                    == 0
                ):
                    print(
                        f"Process: {os.getpid()} reached {int(i / initial_length * 100)}%",
                        flush=True,
                    )
            print(f"Process: {os.getpid()} finished.", flush=True)
        else:
            while element := _pop(
                input
            ):  # Note: Walrus operator (:=) requires Python >= 3.8
                output.append(self.function(element))

    def process(self):
        with Manager() as manager:
            global input_list
            input_list = self.input_list
            INPUT = manager.list(input_list)
            OUTPUT = manager.list()  # <-- can be shared between processes.

            pr = []
            for _ in range(self.num_threads):
                p = Process(
                    target=self._process,
                    args=(INPUT, OUTPUT, self.verbose, len(self.input_list)),
                )
                p.start()
                pr.append(p)
            for pp in pr:
                pp.join()
            o = list(OUTPUT)
            return o
