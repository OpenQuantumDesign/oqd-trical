from oqd_compiler_infrastructure import RewriteRule

########################################################################################


class AnalyseHilbertSpace(RewriteRule):
    def __init__(self):
        super().__init__()

        self.hilbert_space = []

    def map_Ion(self, model):

        self.hilbert_space.append(len(model.levels))

    def map_Phonon(self, model):

        self.hilbert_space.append("f")
