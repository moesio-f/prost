"""Convolutional Neural Network.

This module implements a MLP
for object detection.

The MLP is trained in an online
manner.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn, optim

from .core import MatchRect, TrainableMatcher


class _CNNDetector(nn.Module):
    def __init__(self,
                 image_shape: tuple,
                 kernels: list[int],
                 channels: list[int],
                 poolings: list[Callable[[], nn.Module]],
                 paddings: list[int],
                 strides: list[int] = None,
                 activation_fn: Callable[[], nn.Module] = None):
        super().__init__()

        assert len(kernels) > 0
        assert len(kernels) == len(channels)
        assert len(kernels) == len(poolings)
        assert len(kernels) == len(paddings)

        # Set defaults
        if strides is None:
            strides = [1] * len(kernels)

        if activation_fn is None:
            activation_fn = nn.ReLU

        # Construct layers
        layers = []
        for i in range(len(kernels)):
            n_prev = image_shape[0] if i <= 0 else channels[i - 1]
            padding = 0 if paddings[i] is None else paddings[i]
            hidden_layer = [nn.Conv2d(in_channels=n_prev,
                                      out_channels=channels[i],
                                      kernel_size=kernels[i],
                                      stride=strides[i],
                                      padding=padding,
                                      bias=True),
                            activation_fn()]

            # Maybe add pooling
            pooling = poolings[i]
            if pooling is not None:
                hidden_layer.append(pooling())

            # Extend layers with hidden layer
            layers.extend(hidden_layer)

        # Create feature extractor
        layers.append(nn.Flatten())
        self._extractor = nn.Sequential(*layers)

        # Classificator Layer
        # Output the probability of each class (background,
        # object)
        self._clf = nn.Sequential(nn.LazyLinear(out_features=2,
                                                bias=True))

        # Regression layer to output (x, y, w, h)
        # All values are between (0, 1) and should be multiplied
        #   by the image shape to obtain absolute values
        self._reg = nn.Sequential(nn.LazyLinear(out_features=4,
                                                bias=True))

        # Initialize lazy layers
        input_shape = (1, *image_shape)
        out = self._extractor(torch.zeros(input_shape))
        self._clf(out)
        self._reg(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extractor(x)
        logits = self._clf(features)
        rect = self._reg(features)
        return logits, rect


class CNN(TrainableMatcher):
    def __init__(self,
                 start: np.ndarray,
                 roi: MatchRect,
                 kernels: list[int] = None,
                 channels: list[int] = None,
                 poolings: list[Callable[[], nn.Module]] = None,
                 paddings: list[int] = None,
                 strides: list[int] = None,
                 activation_fn: Callable[[], nn.Module] = None,
                 steps: int = 100,
                 lr: float = 0.001,
                 device=None) -> None:
        # Automatically select device
        if device is None:
            device = torch.device('cuda:0'
                                  if torch.cuda.is_available()
                                  else 'cpu')

        # Default parameters
        if kernels is None:
            kernels = [5, 3, 3, 3]

        if channels is None:
            channels = [128, 64, 64, 48]

        if poolings is None:
            poolings = [lambda: nn.MaxPool2d(2),
                        lambda: nn.MaxPool2d(2),
                        lambda: nn.MaxPool2d(2),
                        lambda: nn.MaxPool2d(2)]

        if paddings is None:
            paddings = [0, 0, 0, 0]

        image_shape = start.shape
        if len(image_shape) == 2:
            image_shape = (1, *image_shape)

        # Store variables
        self._device = device
        self._steps = steps
        self._lr = lr
        self._clf_loss = nn.CrossEntropyLoss()
        self._reg_loss = nn.MSELoss()
        self._mlp = _CNNDetector(image_shape,
                                 kernels,
                                 channels,
                                 poolings,
                                 paddings,
                                 strides,
                                 activation_fn)
        self._optim = optim.AdamW(self._mlp.parameters(), lr=self._lr)
        self._last_prob = 0.0
        self._start_roi = roi
        self._min_w = round(self._start_roi.w * 0.8)
        self._max_w = round(self._start_roi.w * 1.2)
        self._min_h = round(self._start_roi.h * 0.8)
        self._max_h = round(self._start_roi.h * 1.2)

        # Send MLP to device
        self._mlp.to(device)

        # Train with start image
        self.train(start, roi)

    def match(self, image: np.ndarray) -> MatchRect:
        # Obtain image width and height
        img_h, img_w = image.shape[:2]

        # Predict
        X = np.expand_dims(image, axis=(0, 1)).astype(np.float32)
        X = torch.from_numpy(X)
        X = X.to(self._device)
        logits, rect = self._mlp(X)

        # Ensure that tensors are available for
        #   CPU
        logits, rect = logits.cpu(), rect.cpu()
        probs = logits.softmax(dim=1)

        # Clip rect to [0, 1]
        rect = torch.clip(rect, 0, 1)

        # Obtain predicted class
        _, pred = torch.max(probs, 1)

        # If the object wasn't found
        if pred == 0:
            self._last_prob = 0.0
            return self._start_roi

        self._last_prob = probs[0][1].item()
        x, y, w, h = torch.squeeze(rect).tolist()
        x = round(x * img_w)
        y = round(y * img_h)
        w = min(self._max_w, max(self._min_w, round(w * img_w)))
        h = min(self._max_h, max(self._min_h, round(h * img_w)))
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
        X = np.expand_dims(image, axis=(0, 1)).astype(np.float32)
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
