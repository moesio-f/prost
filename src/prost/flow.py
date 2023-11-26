"""Mean Shift Optical Flow.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import Matcher, MatchRect


class FLOW(Matcher):
    def __init__(self,
                 start: np.ndarray,
                 roi: MatchRect) -> None:
        assert len(start.shape) == 2
        self._prev = start
        self._roi = roi
        self._flow = (0.5, 3, 15, 3, 5, 1.2, 0)

    def match(self, image: np.ndarray) -> MatchRect:
        assert len(image.shape) == 2

        # Obtain the flow
        flow = cv2.calcOpticalFlowFarneback(self._prev,
                                            image,
                                            None,
                                            *self._flow)

        # Find new ROI
        self._roi = self._mag_mean_shift(flow)

        # Update
        self._prev = image

        # Return current roi
        return self._roi

    def set_new_roi(self,
                    new_roi: MatchRect,
                    reset_flow: bool = True):
        self._roi = new_roi
        if reset_flow:
            self._displacement = np.array([0.0, 0.0])

    def _mag_mean_shift(self, flow: np.ndarray) -> MatchRect:
        # CV2 Mean Shift configuration
        conf = (cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                50,
                0.001)

        # Obtain the magnitude and angle of the field vectors.
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Use the mean-shift starting at the ROI
        # and the magnitude of the flow vectors
        # to obtain new ROI
        _, w = cv2.meanShift(mag,
                             self._roi.as_bounding_rect(),
                             conf)

        return MatchRect(*w)
