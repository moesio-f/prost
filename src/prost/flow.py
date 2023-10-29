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
                 ms_iter: int = 20,
                 ms_eps: int = 20) -> None:
        assert len(start.shape) == 2
        self._prev = start
        self._roi = roi
        self._ms = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ms_iter, ms_eps)
        self._flow = (0.5, 3, 15, 3, 5, 1.2, 0)

    def match(self, image: np.ndarray) -> MatchRect:
        assert len(image.shape) == 2

        # Obtain the flow
        flow = cv2.calcOpticalFlowFarneback(self._prev, image, None, *self._flow)
        
        # Obtain the magnitude and angle of the field vectors.
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Use the mean-shift starting at the ROI
        # For simplicity, we use the magnitude and angle
        #   as an estimate for the motion. Maybe implement
        #   the Mean Shift from scratch and apply it directly
        #   to the dense optical flow.
        _, w = cv2.meanShift(mag * ang, 
                             self._roi.as_bounding_rect(), 
                             self._ms)
        
        # Update
        self._prev = image
        self._roi = MatchRect(*w)

        # Return current roi
        return self._roi