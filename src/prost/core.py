"""Entities.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchRect:
    x: int
    y: int
    w: int
    h: int

    def as_bounding_rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class Matcher(ABC):
    def match(self, image: np.ndarray) -> MatchRect:
        ...


class Dataset:
    IMG_TEMPLATE = 'img{:05d}.png'

    def __init__(self, name: str):
        self._name = name
        self._dir = Path(__file__).parent
        self._dir = self._dir.joinpath('..', '..', 'data')
        self._dir = self._dir.joinpath(self._name)

        if not self._dir.exists():
            self._download(self._dir.parent)

        with self._dir.joinpath(f'{self._name}_frames.txt').open() as f:
            entry = f.readlines()[0]
            self._min, self._max = map(int, entry.split(','))

        assert self._min == 0

    def start_idx(self) -> int:
        return self._min
    
    def end_idx(self) -> int:
        return self._max - 1

    def image(self, index: int, gray: bool = True) -> np.ndarray:
        assert index >= self._min and index < self._max
        flag = cv2.IMREAD_GRAYSCALE
        if not gray:
            flag = cv2.IMREAD_COLOR
        p = self._dir.joinpath('imgs',
                               self.IMG_TEMPLATE.format(index)).resolve()
        return cv2.imread(str(p), flag)

    def gt(self, index: int) -> MatchRect:
        assert index >= self._min and index < self._max
        p = self._dir.joinpath(f'{self._name}_gt.txt')
        c = p.read_text().split('\n')
        row = map(lambda v: int(float(v)), 
                  c[index].split(','))
        return MatchRect(*row)

    def get(self, index: int, gray: bool = True) -> tuple[np.ndarray, MatchRect]:
        return self.image(index, gray), self.gt(index)
    
    def _download(self, data_dir: Path):
        ...


def compare_match(ds: Dataset,
                  matcher: Matcher,
                  index: int) -> tuple[np.ndarray, MatchRect, MatchRect]:
    img, gt = ds.get(index)
    rect = matcher.match(img)
    return img, gt, rect
