"""Mean Shift Optical Flow.
"""
from __future__ import annotations

import cv2
import numpy as np
import numpy.linalg as LA

from .core import Matcher, MatchRect


class FLOW(Matcher):
    def __init__(self,
                 start: np.ndarray,
                 roi: MatchRect,
                 mode: str = 'mag') -> None:
        assert len(start.shape) == 2
        self._prev = start
        self._roi = roi
        self._displacement: np.ndarray = np.array([0.0, 0.0])
        self._flow = (0.5, 3, 15, 3, 5, 1.2, 0)
        self._mag = mode == 'mag'

    def match(self, image: np.ndarray) -> MatchRect:
        assert len(image.shape) == 2

        # Obtain the flow
        flow = cv2.calcOpticalFlowFarneback(self._prev,
                                            image,
                                            None,
                                            *self._flow)

        # Find new ROI
        if self._mag:
            self._roi = self._mag_mean_shift(flow)
        else:
            self._roi = self._displacement_shift(flow)

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


    def _displacement_shift(self, flow: np.ndarray) -> MatchRect:
        # Obtain current ROI
        x, y, w, h = self._roi.as_bounding_rect()

        # Obtain flow vectors in ROI
        flow_vectors = flow[y:y+h, x:x+w, :]

        # Obtain average displacement and collect
        self._displacement += flow_vectors.mean(axis=(0, 1))

        # Round to integer
        d = self._displacement.round().astype(np.int32)

        # Update x and y
        y += d[0]
        x += d[1]

        # Reset displacement if moved a pixel or more
        self._displacement[d > 0] = 0.0

        # Guarantee that x and y are within bounds
        x = min(max(0, x), flow.shape[1])
        y = min(max(0, y), flow.shape[0])

        # Return new ROI
        return MatchRect(x, y, w, h)
