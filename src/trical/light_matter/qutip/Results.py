# Object returned by the time_evolve QuTiP wrapper

import matplotlib.pyplot as plt

########################################################################################


class Results:
    """Class for packaging results obtained from Qutip's solver

    Args:
        qutip_results (qutip.results): result object from qutip
        ops (list): list of operators for which expectation values were taken
        times (iterable): list of times to evaluate these expectation values
        timescale (float): time unit (e.g. 1e-6 for microseconds)

    Attributes:
        expectation_values(dict): Dictionary mapping name of QuantumOperator object to expectation values from Qutip
    """

    def __init__(self, qutip_results, ops, times, timescale):
        self.qutip_results = qutip_results
        self.ops = ops
        self.times = times
        self.timescale = timescale

        self.expectation_values = {}
        for n, ion_proj in enumerate(self.ops):
            self.expectation_values[ion_proj.name] = self.qutip_results.expect[n]

    def quick_plot(self):
        """Convenience method for plotting expectation values for a Results object"""

        for name in self.expectation_values:
            plt.plot(self.times, self.expectation_values[name], label=name)

        plt.xlim(self.times[0], self.times[-1])
        y_min = min(
            [min(self.expectation_values[name]) for name in self.expectation_values]
        )
        y_max = max(
            [max(self.expectation_values[name]) for name in self.expectation_values]
        )
        plt.ylim(y_min, y_max)
        prefixes = {
            1: r"s",
            1e-3: r"ms",
            1e-6: r"$\mu s$",
            1e-9: r"ns",
        }

        if self.timescale in prefixes:
            prefix = prefixes[self.timescale]
        else:
            prefix = f"{self.timescale} s"

        plt.xlabel(f"t ({prefix})")
        plt.grid()
        plt.legend()
        plt.show()
