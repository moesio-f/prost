"""Mean Shift Optical Flow.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import Matcher, MatchRect


class FLOW(Matcher):
    def __init__(self,
                 start: np.ndarray,
                 roi: MatchRect,
                 ms_imp: str = 'cv2',
                 ms_iter: int = 20,
                 ms_eps: int = 20) -> None:
        assert len(start.shape) == 2
        self._prev = start
        self._roi = roi
        self._ms_iter = ms_iter
        self._ms_eps = ms_eps
        self._cv2_ms = ms_imp == 'cv2'
        self._flow = (0.5, 3, 15, 3, 5, 1.2, 0)

    def match(self, image: np.ndarray) -> MatchRect:
        assert len(image.shape) == 2

        # Obtain the flow
        flow = cv2.calcOpticalFlowFarneback(
            self._prev, image, None, *self._flow)

        # Select which implementation to use
        if self._cv2_ms:
            self._roi = self._cv2_mean_shift(flow)
        else:
            self._roi = self._2d_mean_shift(flow)

        # Update
        self._prev = image

        # Return current roi
        return self._roi

    def _cv2_mean_shift(self, flow: np.ndarray) -> MatchRect:
        # CV2 Mean Shift configuration
        conf = (cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                self._ms_iter,
                self._ms_eps)

        # Obtain the magnitude and angle of the field vectors.
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Use the mean-shift starting at the ROI
        # For simplicity, we use the magnitude and angle
        #   as an estimate for the motion. Maybe implement
        #   the Mean Shift from scratch and apply it directly
        #   to the dense optical flow.
        _, w = cv2.meanShift(mag * ang,
                             self._roi.as_bounding_rect(),
                             conf)

        return MatchRect(*w)

    def _2d_mean_shift(self, flow: np.ndarray) -> MatchRect:
        pass
