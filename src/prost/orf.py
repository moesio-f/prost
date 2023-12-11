"""Online Random Forest.

This implementation differs completely from
the original paper, and instead implements a
progressive Cascading Random Forest for object
detection.

Positive and negative samples are accumulated
over the online training phase and used to
train a new model every time the method train
is called.

https://doi.org/10.1109/ICCVW.2009.5457447
http://imrid.net/?p=4367
"""
from __future__ import annotations

import numpy as np

from .core import Matcher, MatchRect


class ORF(Matcher):
    def __init__(self,
                 start: np.ndarray,
                 roi: MatchRect,
                 random_state=42) -> None:
        ...

    def match(self, image: np.ndarray) -> MatchRect:
        ...

    def prob(self) -> float:
        ...

    def train(self, image: np.ndarray, rect: MatchRect):
        ...
