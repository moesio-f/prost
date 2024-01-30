"""Entities.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import cv2
import gdown
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


class TrainableMatcher(Matcher):
    def prob(self) -> float:
        ...

    def train(self, image: np.ndarray, rect: MatchRect):
        ...


class Dataset:
    _URL = ('https://drive.google.com/uc?id='
            '1hePLR3aU-8371Z3idmUK7cDtZp_xbEDk')
    _DIR = Path(__file__).parent.joinpath('data')
    IMG_TEMPLATE = 'img{:05d}.png'

    def __init__(self, name: str):
        self._name = name
        self._dir = self._DIR.joinpath(self._name).resolve()

        if not self._dir.exists():
            self._download(self._dir.parent)

        assert self._dir.exists()
        with self._dir.joinpath(f'{self._name}_frames.txt').open() as f:
            entry = f.readlines()[0]
            self._min, self._max = map(int, entry.split(','))

        assert self._min == 0

    @property
    def name(self) -> str:
        return self._name

    def start_idx(self) -> int:
        return self._min

    def end_idx(self) -> int:
        return self._max

    def n_images(self) -> int:
        return self._max + 1

    def image(self, index: int, gray: bool = True) -> np.ndarray:
        assert index >= self._min and index <= self._max
        flag = cv2.IMREAD_GRAYSCALE
        if not gray:
            flag = cv2.IMREAD_COLOR
        p = self._dir.joinpath('imgs',
                               self.IMG_TEMPLATE.format(index)).resolve()
        img = cv2.imread(str(p), flag)
        mean = np.mean(img)
        assert mean > 0 and mean < 255
        return img

    def gt(self, index: int) -> MatchRect:
        assert index >= self._min and index <= self._max
        p = self._dir.joinpath(f'{self._name}_gt.txt')
        c = p.read_text().split('\n')
        row = map(lambda v: int(float(v)),
                  c[index].split(','))
        return MatchRect(*row)

    def get(self, index: int, gray: bool = True) -> tuple[np.ndarray,
                                                          MatchRect]:
        return self.image(index, gray), self.gt(index)

    def _download(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=False)
        zip_path = data_dir.joinpath('data.zip')

        # Fazendo download do zip
        gdown.cached_download(self._URL,
                              str(zip_path),
                              postprocess=gdown.extractall)

        # Apagando o zip após download e extraçaõ
        zip_path.unlink()

    @classmethod
    def available_datasets(cls) -> list[str]:
        if not cls._DIR.exists():
            print('[WARNING] No datasets found,'
                  ' instantiate the class at least once '
                  'to download all datasets.')
            return []

        return [d.name
                for d in cls._DIR.iterdir()
                if d.is_dir()]


def compare_match(ds: Dataset,
                  matcher: Matcher,
                  index: int) -> tuple[np.ndarray, MatchRect, MatchRect]:
    img, gt = ds.get(index)
    rect = matcher.match(img)
    return img, gt, rect
