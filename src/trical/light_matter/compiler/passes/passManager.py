class PassManager:

    def __init__(self):
        self.passes = []

    def add_pass(self, pass_instance):
        self.passes.append(pass_instance)

    def run(self, target):
        for p in self.passes:
            target = p.run(target)
        return target
