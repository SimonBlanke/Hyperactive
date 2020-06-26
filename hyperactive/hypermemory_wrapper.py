# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import os

from hypermemory import Hypermemory


def meta_data_path():
    current_path = os.path.realpath(__file__)
    return current_path.rsplit("/", 1)[0] + "/meta_data/"


class HyperactiveMemory(Hypermemory):
    def __init__(self, X, y, model, search_space):
        super().__init__(X, y, search_space)
        self.meta_data_path = meta_data_path()

    def load(self):
        df = self.read_csv()
        df = self.hash2objects(df)
        df = self.df_para2pos(df)
        memory = self.df2dict(df)

        return memory

    def dump(self, memory):
        df = self.dict2df(memory)
        df = self.df_pos2para(df)
        df = self.objects2hash(df)
        self.save_to_csv(df)
