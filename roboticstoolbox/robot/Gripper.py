#!/usr/bin/env python
"""
@author Jesse Haviland
"""


class Gripper():

    def __init__(
            self,
            elinks
            ):

        self._n = 0

        for link in elinks:
            if link.isjoint:
                self._n += 1

    @property
    def n(self):
        return self._n
