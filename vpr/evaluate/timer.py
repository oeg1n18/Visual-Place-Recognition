
class Timer:
    def __init__(self, dataset, vpr):
        self.dataset = dataset
        self.vpr = vpr
        self.Q = dataset.get_query_