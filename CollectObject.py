import numpy as np
import awkward as ak

class CollectObject:

    def __init__(self, events):
        self.events = events

    def CreateMask(self, variable, cut):
        loaded = self.objects[variable]
        sign = cut[0]
        value = cut[1]
        if sign == "==":
            self.mask = self.mask & (loaded == value)
        elif sign == ">=":
            self.mask = self.mask & (loaded >= value)
        elif sign == ">":
            self.mask = self.mask & (loaded > value)
        elif sign == "<=":
            self.mask = self.mask & (loaded <= value)
        elif sign == "<":
            self.mask = self.mask & (loaded < value)
        elif sign == "!=":
            self.mask = self.mask & (loaded != value)
        else:
            pass

    def Electron(self, cuts):
        self.objects = self.events["Electron"]
        self.mask = ak.ones_like(self.objects["pt"], dtype=bool)
        for variable, cut in cuts.items():
            self.CreateMask(variable, cut)
        return self.objects[self.mask]

