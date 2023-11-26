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

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .core import Matcher, MatchRect
from .external import mergevec


class ORF(Matcher):
    def __init__(self,
                 start: np.ndarray,
                 roi: MatchRect,
                 random_state=42,
                 quiet: bool = True,
                 bin_prefix: str = '',
                 pos_per_img=100,
                 neg_per_img=500) -> None:
        # Initialize temporary directories
        #   to train the cascade classifier
        self._tmp = tempfile.TemporaryDirectory()
        self._root_dir = Path(self._tmp.name).resolve()
        self._pos_dir = self._root_dir.joinpath('positive')
        self._neg_dir = self._root_dir.joinpath('negative')
        self._model_dir = self._root_dir.joinpath('model')
        self._train_dir = self._root_dir.joinpath('train')
        self._train_bg = self._train_dir.joinpath('bg.txt')
        self._train_vec = self._train_dir.joinpath('input.vec')
        self._pos_dir.mkdir(parents=False, exist_ok=False)
        self._neg_dir.mkdir(parents=False, exist_ok=False)
        self._model_dir.mkdir(parents=False, exist_ok=False)
        self._train_dir.mkdir(parents=False, exist_ok=False)

        # Initialize other variables
        self._clf = None
        self._rng = np.random.default_rng(random_state)
        self._roi = roi
        self._last_prob = 0.0
        self._idx_pos = 1
        self._idx_neg = 1
        self._idx_train_vec = 1
        self._quiet = quiet
        self._bin = bin_prefix
        self._pos_p_img = pos_per_img
        self._neg_p_img = neg_per_img

        # Train with start image
        self.train(start, roi)

    def match(self, image: np.ndarray) -> MatchRect:
        # Detect the object in the image
        min_w, max_w = int(0.75*self._roi.w), int(1.25*self._roi.w)
        min_h, max_h = int(0.75*self._roi.h), int(1.25*self._roi.h)
        matches, _, w = self._clf.detectMultiScale3(
            image,
            outputRejectLevels=True,
            minSize=(min_w, min_h),
            maxSize=(max_w, max_h))

        # If no match is found, return
        #   the first ROI
        if len(matches) <= 0:
            self._last_prob = 0.0
            return self._roi

        # Return the object with the closest ratio
        def _diff(m):
            r = self._roi.w / self._roi.h
            r_ = m[0][2] / m[0][3]
            return abs(r - r_) - m[1]

        matches = sorted(zip(matches, w), key=_diff)
        self._last_prob = matches[0][1]
        return MatchRect(*matches[0][0])

    def prob(self) -> float:
        return self._last_prob

    def train(self, image: np.ndarray, rect: MatchRect):
        # Create negative samples
        self._create_n_negatives(image, rect)

        # Create positive samples
        self._create_n_positives(image, rect)

        # Then, we can finally train and store the model
        self._train_cascade()

        # Lastly, we load the new model
        self._clf = cv2.CascadeClassifier()
        self._clf.load(str(self._model_dir.joinpath('cascade.xml')))

    def _create_n_negatives(self,
                            image: np.ndarray,
                            rect: MatchRect):
        # Create a copy of the image
        image = image.copy()
        mean_color = image.mean()
        img_h, img_w = image.shape

        # Replace the object with random noise
        noise_min = max(image.min(), mean_color - 10)
        noise_max = min(image.max(), mean_color + 10)
        noise = self._rng.integers(noise_min, noise_max,
                                   size=(rect.h, rect.w))
        image[rect.y:rect.y+rect.h,
              rect.x:rect.x+rect.w] = noise

        # Generate negative examples
        for _ in range(self._neg_p_img):
            # Obtain filename
            fpath = f'img{self._idx_neg}.jpg'
            fpath = str(self._neg_dir.joinpath(fpath))

            # Generate random sample
            kind = self._rng.choice(['same', 'greater'])
            w, h = rect.w, rect.h
            if kind == 'greater':
                h = self._rng.integers(h, image.shape[0])
                w = self._rng.integers(w, image.shape[1])
            x = self._rng.integers(0, img_w - w)
            y = self._rng.integers(0, img_h - h)
            img = image[y:y+h, x:x+w].copy()

            # Update intensity
            if self._rng.random() > 0.65:
                k = self._rng.integers(-2, 2)
                img = cv2.convertScaleAbs(img, None, beta=k)

            # Save sample
            cv2.imwrite(fpath, img)
            with self._train_bg.open('a') as f:
                f.write(f"{fpath}\n")

            # Increase index
            self._idx_neg += 1

    def _create_n_positives(self,
                            image: np.ndarray,
                            rect: MatchRect):
        # Obtain ROI
        image = image[rect.y:rect.y+rect.h,
                      rect.x:rect.x+rect.w].copy()

        # Add positive image to directory
        fpath = f'img{self._idx_pos}.jpg'
        fpath = str(self._pos_dir.joinpath(fpath))
        cv2.imwrite(fpath, image)

        # Increase index
        out_vec = self._pos_dir.joinpath(f'vec{self._idx_train_vec}.vec')
        self._idx_pos += 1
        self._idx_train_vec += 1

        # Generate random samples
        rel_bg = str(self._train_bg)
        rel_bg = rel_bg.replace(f'{self._root_dir}', '..')
        extras = ' > /dev/null' if self._quiet else ''
        code = os.system(f'{self._bin}opencv_createsamples '
                         f'-vec {str(out_vec)} '
                         f'-img {fpath} '
                         f'-bg {rel_bg} '
                         f'-w {self._roi.w} -h {self._roi.h} '
                         f'-num {self._pos_p_img}{extras}')
        assert not code

    def _train_cascade(self):
        # Delete current model if it exists
        if self._model_dir.exists():
            shutil.rmtree(self._model_dir)
            self._model_dir.mkdir(parents=False, exist_ok=False)

        # Merge all .vec files
        out_vec = str(self._train_vec)
        mergevec.merge_vec_files(str(self._pos_dir), out_vec)

        # Automatically set n_stages and number os samples
        total_pos = self._pos_p_img * (self._idx_pos - 1)
        total_neg = self._idx_neg - 1
        n_stages = min(2 * (self._idx_pos - 1), 10)
        num_pos = total_pos // n_stages
        num_neg = total_neg // n_stages

        # Train cascade classifier
        extras = ' > /dev/null' if self._quiet else ''
        code = os.system(f'{self._bin}opencv_traincascade '
                         f'-data {str(self._model_dir)} '
                         f'-vec {out_vec} '
                         f'-bg {str(self._train_bg)} '
                         f'-numPos {num_pos} '
                         f'-numNeg {num_neg} '
                         f'-w {self._roi.w} -h {self._roi.h} '
                         f'-minHitRate 0.8 '
                         f'-numStages {n_stages}{extras}')
        assert not code
