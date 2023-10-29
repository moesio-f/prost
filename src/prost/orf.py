"""Online Random Forest.

https://doi.org/10.1109/ICCVW.2009.5457447
http://imrid.net/?p=4367
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import Matcher, MatchRect

class ORF(Matcher):
    def match(self, image: np.ndarray) -> MatchRect:
        return None
    
    def prob(self) -> float:
        return 1.0
    
    def train(self, image: np.ndarray, rect: MatchRect):
        ...
