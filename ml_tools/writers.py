# Copyright 2023 Vincent Dutordoir.
#
# Modifications:
#   - TensorBoardX support
#   - Local file writer

# Copyright 2022 The CLU Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library for unify reporting model metrics across various logging formats.

This library provides a MetricWriter for each logging format (SummyWriter,
LoggingWriter, etc.) and composing MetricWriter to add support for asynchronous
logging or writing to multiple formats.
"""

import abc
import atexit
import os
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd
from jaxtyping import Array
from tensorboardX import SummaryWriter
from tensorboardX.utils import figure_to_image, make_grid

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("please install matplotlib")


Scalar = Union[int, float, Array]


class _MetricWriter(abc.ABC):
    """MetricWriter inferface."""

    def __init__(self):
        atexit.register(self.close)

    @abc.abstractmethod
    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        """Write scalar values for the step.

        Consecutive calls to this method can provide different sets of scalars.
        Repeated writes for the same metric at the same step are not allowed.

        Args:
          step: Step at which the scalar values occurred.
          scalars: Mapping from metric name to value.
        """

    @abc.abstractmethod
    def write_images(self, step: int, images: Mapping[str, Array]):
        """Write images for the step.

        Consecutive calls to this method can provide different sets of images.
        Repeated writes for the same image key at the same step are not allowed.

        Warning: Not all MetricWriter implementation support writing images!

        Args:
          step: Step at which the images occurred.
          images: Mapping from image key to images. Images should have the shape [N,
            H, W, C] or [H, W, C], where H is the height, W is the width and C the
            number of channels (1 or 3). N is the number of images that will be
            written. Image dimensions can differ between different image keys but
            not between different steps for the same image key.
        """

    @abc.abstractmethod
    def write_figures(self, step: int, figures: Mapping[str, plt.Figure]):
        """Writes matplotlib figures.

        Note that this requires the ``matplotlib`` package.

        Args:
            figure (matplotlib.pyplot.figure)
        """

    @abc.abstractmethod
    def flush(self):
        """Tells the MetricWriter to write out any cached values."""

    @abc.abstractmethod
    def close(self):
        """Flushes and closes the MetricWriter.

        Calling any method on MetricWriter after MetricWriter.close()
        is undefined behavior.
        """


class MultiWriter(_MetricWriter):
    """MetricWriter that writes to multiple writers at once."""

    def __init__(self, writers: Sequence[_MetricWriter]):
        self._writers = tuple(writers)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        for w in self._writers:
            w.write_scalars(step, scalars)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for w in self._writers:
            w.write_images(step, images)

    def write_figures(self, step: int, figures: Mapping[str, plt.Figure]):
        for w in self._writers:
            w.write_figures(step, figures)

    def flush(self):
        for w in self._writers:
            w.flush()

    def close(self):
        for w in self._writers:
            w.close()


class TensorBoardWriter(_MetricWriter):
    """MetricWriter that writes TF summary files."""

    def __init__(self, logdir: str, export_scalars: bool = True):
        """
        export_scalars: If `True` exports the scalars to json to `logdir/scalars.json`
            when writer is closed.
        """
        super().__init__()
        self._export_scalars = export_scalars
        self._logdir = logdir
        self._summary_writer = SummaryWriter(logdir)

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        for key, value in scalars.items():
            self._summary_writer.add_scalar(key, value, global_step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        """format: (N)CHW

        img_tensor: An `uint8` or `float` Tensor of shape `
            [channel, height, width]` where `channel` is 1, 3, or 4.
            The elements in img_tensor can either have values
            in [0, 1] (float32) or [0, 255] (uint8).
            Users are responsible to scale the data in the correct range/type.
        """
        for key, value in images.items():
            if len(value.shape) == 3:
                self._summary_writer.add_image(key, value, global_step=step)
            if len(value.shape) == 4:
                self._summary_writer.add_images(key, value, global_step=step)

    def write_figures(self, step: int, figures: Mapping[str, plt.Figure]):
        """Writes matplotlib figures.

        Note that this requires the ``matplotlib`` package.

        Args:
            figure (matplotlib.pyplot.figure)
        """
        self.write_images(step, {k: figure_to_image(fig) for k, fig in figures.items()})

    def flush(self):
        self._summary_writer.flush()

    def close(self):
        if self._export_scalars:
            self._summary_writer.export_scalars_to_json(f"{self._logdir}/scalars.json")
        self._summary_writer.close()


def _cond_mkdir(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


_IMAGE_PATH_TEMPLATE = "%s/images"


class LocalWriter(_MetricWriter):
    """MetricWriter that writes files to local disk."""

    def __init__(self, logdir: str, flush_every_n: int = 100):
        """
        export_scalars: If `True` exports the scalars to json to `logdir/scalars.json`
            when writer is closed.
        """
        super().__init__()
        _cond_mkdir(logdir)
        self._count = 0
        self._flush_every_n = flush_every_n
        self._logdir = logdir
        self._metrics_path = f"{self._logdir}/metrics.csv"
        self._metrics = []

    def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
        metrics = {"step": step, **scalars}
        self._metrics.append(metrics)
        if (self._count + 1) % self._flush_every_n == 0:
            self.flush()
        self._count += 1

    def write_images(self, step: int, images: Mapping[str, Array]):
        """format: (N)CHW, C=1 or 3

        img_tensor: An `uint8` or `float` Tensor of shape `
            [channel, height, width]` where `channel` is 1, 3, or 4.
            The elements in img_tensor can either have values
            in [0, 1] (float32) or [0, 255] (uint8).
            Users are responsible to scale the data in the correct range/type.
        """
        path = _IMAGE_PATH_TEMPLATE % self._logdir
        _cond_mkdir(path)

        for key, value in images.items():
            if len(value.shape) == 4:
                value = make_grid(value)

            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.imshow(np.transpose(value, [1, 2, 0]))
            fig.savefig(path + f"/{key}_{step}.png", bbox_inches="tight", dpi=300)
            plt.close(fig)

    def write_figures(self, step: int, figures: Mapping[str, plt.Figure]):
        """Writes matplotlib figures.

        Note that this requires the ``matplotlib`` package.

        Args:
            figure (matplotlib.pyplot.figure)
        """
        path = _IMAGE_PATH_TEMPLATE % self._logdir
        _cond_mkdir(path)

        for key, fig in figures.items():
            fig.savefig(path + f"/{key}_{step}.png", bbox_inches="tight")
            plt.close(fig)

    def flush(self):
        if len(self._metrics) == 0:
            return

        if os.path.exists(self._metrics_path):
            # append
            pd.DataFrame(self._metrics).to_csv(
                self._metrics_path, mode="a", header=False, index=False
            )
        else:
            # create new one
            pd.DataFrame(self._metrics).to_csv(self._metrics_path, header=False, index=False)

        self._count = 0
        self._metrics = []

    def close(self):
        self.flush()
