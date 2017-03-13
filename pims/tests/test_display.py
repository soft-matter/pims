from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import nose
import os
import functools
import numpy as np
import pims
from numpy.testing import assert_array_equal
from pims import plot_to_frame, plots_to_frame
from pims.display import export_moviepy, export_pyav
from nose.tools import assert_true, assert_equal, assert_less
from .test_common import _skip_if_no_MoviePy, _skip_if_no_PyAV, path

import unittest
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    mpl = None
    plt = None

def _skip_if_no_mpl():
    if plt is None:
        raise nose.SkipTest('Matplotlib not installed. Skipping.')

class TestPlotToFrame(unittest.TestCase):
    def setUp(self):
        _skip_if_no_mpl()
        plt.switch_backend('Agg')  # does not plot to screen
        x = np.linspace(0, 2*np.pi, 100)
        t = np.linspace(0, 2*np.pi, 10)
        y = np.sin(x[np.newaxis, :] - t[:, np.newaxis])

        self.figures = []
        self.axes = []
        for line in y:
            fig = plt.figure(figsize=(8, 6), tight_layout=False)
            ax = fig.gca()
            ax.plot(x, line)
            self.figures.append(fig)
            self.axes.append(ax)

        self.expected_shape = (10, 384, 512, 4)

        mpl.rcParams.update({'savefig.bbox': None})

    def tearDown(self):
        for fig in self.figures:
            plt.close(fig)

    def test_ax_to_frame(self):
        frame = plot_to_frame(self.axes[0])
        assert_equal(frame.shape, (384, 512, 4))

    def test_plot_to_frame(self):
        frame = plot_to_frame(self.figures[0])
        assert_equal(frame.shape, (384, 512, 4))

    def test_axes_to_frame(self):
        frame = plots_to_frame(self.axes)
        assert_equal(frame.shape, (10, 384, 512, 4))

    def test_plots_to_frame(self):
        frame = plots_to_frame(self.figures)
        assert_equal(frame.shape, (10, 384, 512, 4))

    def test_plot_width(self):
        width = np.random.randint(100, 1000)
        frame = plot_to_frame(self.figures[0], width)
        assert_equal(frame.shape[1], width)

    def test_plot_tight(self):
        fig = self.figures[0]
        fig.set_tight_layout(False)  # default to standard
        assert_equal(plot_to_frame(fig).shape[:2], (384, 512))
        assert_less(plot_to_frame(fig, bbox_inches='tight').shape[:2], (384, 512))
        assert_equal(plot_to_frame(fig).shape[:2], (384, 512))

        fig.set_tight_layout(True)   # default to tight
        assert_less(plot_to_frame(fig).shape[:2], (384, 512))
        assert_equal(plot_to_frame(fig, bbox_inches='standard').shape[:2],
                    (384, 512))
        assert_less(plot_to_frame(fig).shape[:2], (384, 512))

    def test_plots_tight(self):
        frame = plots_to_frame(self.figures, bbox_inches='tight')
        assert_less(frame.shape[1:3], (384, 512))

    def test_plot_resize(self):
        frame = plot_to_frame(self.figures[0], fig_size_inches=(4, 4))
        assert_equal(frame.shape, (512, 512, 4))

    def test_plots_resize(self):
        frame = plots_to_frame(self.figures, fig_size_inches=(4, 4))
        assert_equal(frame.shape, (10, 512, 512, 4))

    def test_plots_width(self):
        width = np.random.randint(100, 1000)
        frame = plots_to_frame(self.figures, width)
        assert_equal(frame.shape[2], width)

    def test_plots_from_generator(self):
        frame = plots_to_frame(iter(self.figures))
        assert_equal(frame.shape, (10, 384, 512, 4))


class ExportCommon(object):
    def setUp(self):
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        self.expected_shape = (128, 128, 3)
        self.expected_len = 20

        self.sequence = np.random.randint(0, 255, size=(self.expected_len,) +
                                                       self.expected_shape,
                                          dtype=np.uint8)
        self.tempfile = 'tempvideo.avi'  # avi containers support most codecs

    def tearDown(self):
        if os.path.isfile(self.tempfile):
            os.remove(self.tempfile)

    def test_quality_wmv2(self):
        self.export_func(self.sequence, self.tempfile, codec='wmv2',
                         quality=1)
        lossless_size = int(os.path.getsize(self.tempfile))

        self.export_func(self.sequence, self.tempfile, codec='wmv2',
                         quality=0.01)
        compressed_size = int(os.path.getsize(self.tempfile))

        assert_less(compressed_size, lossless_size)

    def test_quality_mpeg4(self):
        self.export_func(self.sequence, self.tempfile, codec='mpeg4',
                         quality=1)
        lossless_size = int(os.path.getsize(self.tempfile))

        self.export_func(self.sequence, self.tempfile, codec='mpeg4',
                         quality=5)
        compressed_size = int(os.path.getsize(self.tempfile))

        assert_less(compressed_size, lossless_size)

    def test_quality_h264(self):
        self.export_func(self.sequence, self.tempfile, codec='libx264',
                         quality=0)
        lossless_size = int(os.path.getsize(self.tempfile))

        self.export_func(self.sequence, self.tempfile, codec='libx264',
                         quality=23)
        compressed_size = int(os.path.getsize(self.tempfile))

        assert_less(compressed_size, lossless_size)

    def test_rawvideo_export(self):
        """Exported frames must equal the input exactly"""
        self.export_func(self.sequence, self.tempfile, codec='rawvideo',
                         pixel_format='bgr24')
        with pims.open(self.tempfile) as reader:
            assert_equal(len(reader), self.expected_len)
            assert_equal(reader.frame_shape, self.expected_shape)
            for a, b in zip(self.sequence, reader):
                assert_array_equal(a, b)


class TestExportMoviePy(unittest.TestCase, ExportCommon):
    def setUp(self):
        _skip_if_no_MoviePy()
        self.export_func = functools.partial(export_moviepy, verbose=False)
        ExportCommon.setUp(self)


class TestExportPyAV(unittest.TestCase, ExportCommon):
    def setUp(self):
        _skip_if_no_PyAV()
        self.export_func = export_pyav
        ExportCommon.setUp(self)
