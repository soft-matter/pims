import unittest
import os
import functools
import numpy as np
import pims
from numpy.testing import assert_array_equal
from pims import plot_to_frame, plots_to_frame
from pims.display import export_moviepy, export_pyav
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
        raise unittest.SkipTest('Matplotlib not installed. Skipping.')


class TestPlotToFrame(unittest.TestCase):
    def setUp(self):
        _skip_if_no_mpl()
        plt.switch_backend('Agg')  # does not plot to screen
        x = np.linspace(0, 2 * np.pi, 100)
        t = np.linspace(0, 2 * np.pi, 10)
        y = np.sin(x[np.newaxis, :] - t[:, np.newaxis])

        self.figures = []
        self.axes = []
        for line in y:
            fig = plt.figure(figsize=(8, 6))
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
        self.assertEqual(frame.shape, (384, 512, 4))

    def test_plot_to_frame(self):
        frame = plot_to_frame(self.figures[0])
        self.assertEqual(frame.shape, (384, 512, 4))

    def test_axes_to_frame(self):
        frame = plots_to_frame(self.axes)
        self.assertEqual(frame.shape, (10, 384, 512, 4))

    def test_plots_to_frame(self):
        frame = plots_to_frame(self.figures)
        self.assertEqual(frame.shape, (10, 384, 512, 4))

    def test_plot_width(self):
        width = np.random.randint(100, 1000)
        frame = plot_to_frame(self.figures[0], width)
        self.assertEqual(frame.shape[1], width)

    def test_plot_tight(self):
        fig = self.figures[0]
        self.assertEqual(plot_to_frame(fig).shape[:2], (384, 512))
        self.assertLess(
            plot_to_frame(fig, bbox_inches='tight').shape[:2], (384, 512))
        self.assertEqual(plot_to_frame(fig).shape[:2], (384, 512))

        # default to tight
        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine("tight")
        else:
            fig.set_tight_layout(True)
        self.assertLess(plot_to_frame(fig).shape[:2], (384, 512))
        self.assertEqual(
            plot_to_frame(fig, bbox_inches='standard').shape[:2], (384, 512))
        self.assertLess(plot_to_frame(fig).shape[:2], (384, 512))

    def test_plots_tight(self):
        frame = plots_to_frame(self.figures, bbox_inches='tight')
        self.assertLess(frame.shape[1:3], (384, 512))

    def test_plot_resize(self):
        frame = plot_to_frame(self.figures[0], fig_size_inches=(4, 4))
        self.assertEqual(frame.shape, (512, 512, 4))

    def test_plots_resize(self):
        frame = plots_to_frame(self.figures, fig_size_inches=(4, 4))
        self.assertEqual(frame.shape, (10, 512, 512, 4))

    def test_plots_width(self):
        width = np.random.randint(100, 1000)
        frame = plots_to_frame(self.figures, width)
        self.assertEqual(frame.shape[2], width)

    def test_plots_from_generator(self):
        frame = plots_to_frame(iter(self.figures))
        self.assertEqual(frame.shape, (10, 384, 512, 4))


class ExportCommon(object):
    def setUp(self):
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
        h, w = 128, 128
        self.expected_shape = (h, w, 3)
        self.expected_shape_rgba = (h, w, 4)
        self.expected_len = 20

        self.sequence = np.random.randint(0, 255,
                                          size=(self.expected_len,) +
                                               self.expected_shape,
                                          ).astype(np.uint8)
        self.sequence_rgba = np.random.randint(0, 255,
                                               size=(self.expected_len,) +
                                                    self.expected_shape_rgba,
                                               ).astype(np.uint8)
        self.tempfile = 'tempvideo.avi'  # avi containers support most codecs

    def tearDown(self):
        if hasattr(self, 'tempfile') and os.path.isfile(self.tempfile):
            os.remove(self.tempfile)

    def test_quality_wmv2(self):
        self.export_func(self.sequence, self.tempfile, codec='wmv2',
                         quality=1)
        lossless_size = int(os.path.getsize(self.tempfile))

        self.export_func(self.sequence, self.tempfile, codec='wmv2',
                         quality=0.01)
        compressed_size = int(os.path.getsize(self.tempfile))

        self.assertLess(compressed_size, lossless_size)

    def test_quality_mpeg4(self):
        self.export_func(self.sequence, self.tempfile, codec='mpeg4',
                         quality=1)
        lossless_size = int(os.path.getsize(self.tempfile))

        self.export_func(self.sequence, self.tempfile, codec='mpeg4',
                         quality=5)
        compressed_size = int(os.path.getsize(self.tempfile))

        self.assertLess(compressed_size, lossless_size)

    def test_quality_h264(self):
        self.export_func(self.sequence, self.tempfile, codec='libx264',
                         quality=0)
        lossless_size = int(os.path.getsize(self.tempfile))

        self.export_func(self.sequence, self.tempfile, codec='libx264',
                         quality=23)
        compressed_size = int(os.path.getsize(self.tempfile))

        self.assertLess(compressed_size, lossless_size)

    def test_rgba_h264(self):
        """Remove alpha channel."""
        # Start with smoke test for H.264
        self.export_func(self.sequence_rgba, self.tempfile, codec='libx264')
        # Check that RGB channels are preserved
        self.export_func(self.sequence_rgba, self.tempfile, codec='rawvideo',
                         pixel_format='bgr24')
        sequence_rgba_stripped = self.sequence_rgba[:, :, :, :3]
        with pims.open(self.tempfile) as reader:
            self.assertEqual(len(reader), self.expected_len)
            self.assertEqual(reader.frame_shape, self.expected_shape)
            for a, b in zip(sequence_rgba_stripped, reader):
                assert_array_equal(a, b)

    def test_rawvideo_export(self):
        """Exported frames must equal the input exactly"""
        self.export_func(self.sequence, self.tempfile, codec='rawvideo',
                         pixel_format='bgr24')
        with pims.open(self.tempfile) as reader:
            self.assertEqual(len(reader), self.expected_len)
            self.assertEqual(reader.frame_shape, self.expected_shape)
            for a, b in zip(self.sequence, reader):
                assert_array_equal(a, b)

    def test_rgb_memorder(self):
        """Fortran memory order must be converted."""
        self.export_func(self.sequence.astype(self.sequence.dtype,
                                              order='F'),
                         self.tempfile, codec='rawvideo',
                         pixel_format='bgr24')
        with pims.open(self.tempfile) as reader:
            self.assertEqual(len(reader), self.expected_len)
            self.assertEqual(reader.frame_shape, self.expected_shape)
            for a, b in zip(self.sequence, reader):
                assert_array_equal(a, b)


class TestExportMoviePy(ExportCommon, unittest.TestCase):
    def setUp(self):
        _skip_if_no_MoviePy()
        self.export_func = functools.partial(export_moviepy, verbose=False)
        ExportCommon.setUp(self)

    def tearDown(self):
        super(TestExportMoviePy, self).tearDown()


class TestExportPyAV(ExportCommon, unittest.TestCase):
    def setUp(self):
        _skip_if_no_PyAV()
        self.export_func = export_pyav
        ExportCommon.setUp(self)

    def tearDown(self):
        super(TestExportPyAV, self).tearDown()
