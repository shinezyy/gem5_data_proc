from matplotlib import pyplot as plt


class ColorRotater():
    def __init__(self):
        self.colors = plt.get_cmap('Dark2').colors
        self.index = 0
        self.last_index = 0

    def get(self, inc=1):
        self.last_index = (self.index + inc - 1) % len(self.colors)
        self.index = (self.index + inc) % len(self.colors)
        return self.colors[self.last_index]

    def last(self):
        return self.colors[self.last_index]