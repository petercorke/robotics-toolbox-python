import sys
import zerorpc
import numpy as np
from subprocess import call, Popen


class RopyApi(object):

    def __init__(self):
        self.c = zerorpc.Client()
        self.c.connect("tcp://127.0.0.1:4242")
        arr = np.eye(4).tolist()
        a = self.c.hello(arr)
        print(a)

    def draw_robot(self, poses):
        """based on the input text, return the int result"""
        self.c.draw_robot(pose)


if __name__ == '__main__':
    RopyApi()
