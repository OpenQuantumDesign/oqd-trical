from oqd_compiler_infrastructure import RewriteRule

########################################################################################


class AnalyseHilbertSpace(RewriteRule):
    def __init__(self):
        super().__init__()

        self.hilbert_space = {}

    def map_System(self, model):

        for n, ion in enumerate(model.ions):
            self.hilbert_space[f"E{n}"] = len(ion.levels)

        for m, mode in enumerate(model.modes):
            self.hilbert_space[f"P{m}"] = None
