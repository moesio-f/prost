"""Normalized Cross-Correlation for
Template Matching.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import Matcher, MatchRect


class NCC(Matcher):
    def __init__(self, template: np.ndarray) -> None:
        assert len(template.shape) == 2
        self._template = template

    def match(self, image: np.ndarray) -> MatchRect:
        res = cv2.matchTemplate(image, self._template, cv2.TM_CCORR_NORMED)
        w, h = self._template.shape[::-1]
        _, _, _, top_left = cv2.minMaxLoc(res)
        return MatchRect(x=top_left[0], 
                         y=top_left[1],
                         w=w, 
                         h=h)
