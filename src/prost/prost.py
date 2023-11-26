"""PROST.
"""
from __future__ import annotations

import numpy as np

from .core import Matcher, MatchRect
from .flow import FLOW
from .ncc import NCC
from .orf import ORF


class PROST(Matcher):
    def __init__(self,
                 roi: MatchRect,
                 image: np.ndarray,
                 orf_th: float = 0.75):
        template = image[roi.y:roi.y + roi.h,
                                 roi.x:roi.x + roi.w]
        self._ncc = NCC(template)
        self._flow = FLOW(image, roi)
        self._orf = ORF()
        self._orf_th = orf_th

    def match(self, image: np.ndarray) -> MatchRect:
        # Perform all matching
        flow = self._flow.match(image)
        ncc = self._ncc.match(image)
        orf = self._orf.match(image)

        # The default output is the one from FLOW
        res = flow

        # Whether to use ORF or FLOW
        no_overlap = not self._overlap(flow, orf)
        over_th = self._orf.prob() >= self._orf_th
        if no_overlap and over_th:
            res = orf
            # Update FLOW with new ROI
            self._flow.set_new_roi(orf)

        # Whether to train the ORF or not
        if not no_overlap or self._overlap(ncc, orf):
            self._orf.train(image, ncc)

        return res

    def _overlap(self, a: MatchRect, b: MatchRect) -> bool:
        ...