from util.model import superglue, superpoint

"""
Super Model:
- Initialize the SuperPoint and SuperGlue Model
"""


class SuperModel:
    def __init__(self, gpu=True):
        self.processer = 'cuda:0' if gpu else 'cpu:0'
        self.superpoint_model = superpoint.SuperPoint({}).eval().to(self.processer)
        self.superglue_model = superglue.SuperGlue({}).eval().to(self.processer)
