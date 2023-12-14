"""Multilayer Perceptron.

This module implements a MLP
for object detection.

The MLP is trained in an online
manner.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from .core import MatchRect, TrainableMatcher


class _MLPDetector(nn.Module):
    def __init__(self,
                 image_shape: tuple,
                 n_hidden: list[int]):
        super().__init__()

        # Input layer
        extractor = [nn.Flatten()]

        # Hidden layers
        for i in range(len(n_hidden)):
            n_prev = np.prod(image_shape) if i <= 0 else n_hidden[i - 1]
            hidden_layer = [nn.Linear(n_prev,
                                      n_hidden[i],
                                      bias=True),
                            nn.ReLU()]

            # Extend layers with hidden layer
            extractor.extend(hidden_layer)

        # Feature extractor
        self._extractor = nn.Sequential(*extractor)

        # Classificator Layer
        # Output the probability of each class (background,
        # object)
        self._clf = nn.Sequential(nn.Linear(n_hidden[-1],
                                            2,
                                            bias=True),
                                  nn.Softmax(dim=1))

        # Regression layer to output (x, y, w, h)
        # All values are between (0, 1) and should be multiplied
        #   by the image shape to obtain absolute values
        self._reg = nn.Sequential(nn.Linear(n_hidden[-1],
                                            4,
                                            bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extractor(x)
        logits = self._clf(features)
        rect = self._reg(features)
        return logits, rect


class MLP(TrainableMatcher):
    def __init__(self,
                 start: np.ndarray,
                 roi: MatchRect,
                 n_hidden: list[int] = None,
                 steps: int = 100,
                 lr: float = 0.001,
                 device=None) -> None:
        # Automatically select device
        if device is None:
            device = torch.device('cuda:0'
                                  if torch.cuda.is_available()
                                  else 'cpu')

        # Default hidden layers
        if n_hidden is None:
            n_hidden = [500]

        # Store variables
        self._device = device
        self._n_hidden = n_hidden
        self._steps = steps
        self._lr = lr
        self._clf_loss = nn.CrossEntropyLoss()
        self._reg_loss = nn.MSELoss()
        self._mlp = _MLPDetector(start.shape, n_hidden)
        self._optim = optim.AdamW(self._mlp.parameters(), lr=self._lr)
        self._last_prob = 0.0
        self._start_roi = roi

        # Send MLP to device
        self._mlp.to(device)

        # Train with start image
        self.train(start, roi)

    def match(self, image: np.ndarray) -> MatchRect:
        # Obtain image width and height
        img_h, img_w = image.shape[:2]

        # Predict
        X = np.expand_dims(image, axis=0).astype(np.float32)
        X = torch.from_numpy(X)
        X = X.to(self._device)
        logits, rect = self._mlp(X)

        # Ensure that tensors are available for
        #   CPU
        logits, rect = logits.cpu(), rect.cpu()

        # Clip rect to [0, 1]
        rect = torch.clip(rect, 0, 1)

        # Obtain predicted class
        _, pred = torch.max(logits, 1)

        # If the object wasn't found
        if pred == 0:
            self._last_prob = 0.0
            return self._start_roi

        self._last_prob = logits[0][1].item()
        x, y, w, h = torch.squeeze(rect).tolist()
        x = round(x * img_w)
        y = round(y * img_h)
        w = round(w * img_w)
        h = round(h * img_h)
        return MatchRect(x, y, w, h)

    def prob(self) -> float:
        return self._last_prob

    def train(self, image: np.ndarray, rect: MatchRect):
        # Obtain image width and height
        img_h, img_w = image.shape[:2]

        # Obtain targets
        y_clf = torch.ones(1, dtype=torch.long, device=self._device)
        y_reg = torch.from_numpy(np.array([[
            rect.x / img_w,
            rect.y / img_h,
            rect.w / img_w,
            rect.h / img_h]],
            dtype=np.float32)).to(self._device)

        # Obtain input
        X = np.expand_dims(image, axis=0).astype(np.float32)
        X = torch.from_numpy(X)
        X = X.to(self._device)

        for _ in range(self._steps):
            self._optim.zero_grad()
            logits, rect = self._mlp(X)
            loss_clf = self._clf_loss(logits, y_clf)
            loss_reg = self._reg_loss(rect, y_reg)
            loss = loss_clf + loss_reg
            loss.backward()
            self._optim.step()
