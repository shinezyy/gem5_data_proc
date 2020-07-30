import os
from os.path import join as pjoin
from paths import st_stat_dir


class Env(object):
    def __init__(self):
        self.prefix = st_stat_dir

    def get_stat_dir(self):
        return os.path.abspath(self.prefix)

    def data(self, s):
        return os.path.abspath(pjoin(self.prefix, s))