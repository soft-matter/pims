from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six import with_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from skimage.viewer.qt import QtGui, QtCore, QtWidgets, FigureCanvasQTAgg

try:
    from qimage2ndarray import array2qimage
    has_qimage2ndarray = True
except:
    has_qimage2ndarray = False

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

try:
    import pyqtgraph
    if ((QtCore.PYQT_VERSION_STR == pyqtgraph.Qt.QtCore.PYQT_VERSION_STR) and
        (QtCore.QT_VERSION_STR == pyqtgraph.Qt.QtCore.QT_VERSION_STR)):
        from pyqtgraph.opengl import GLViewWidget, GLVolumeItem, GLBoxItem
        has_pyqtgraph = True
    else:
        has_pyqtgraph = False
except ImportError:
    has_pyqtgraph = False


class Display(with_metaclass(ABCMeta, object)):
    name = 'base_class'
    ndim = 2
    available = False
    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def ndim(self):
        pass

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self.update_image(value)
        self._image = value

    @abstractmethod
    def update_image(self, value):
        pass


class DisplayQt(Display):
    name = 'Qt'
    ndim = 2
    available = has_qimage2ndarray
    def __init__(self, shape, mpp=1.):
        self._widget = QtWidgets.QLabel()
        self._widget.setBackgroundRole(QtGui.QPalette.Base)
        self._widget.setScaledContents(True)
        self._widget.updateGeometry()

    def update_image(self, value):
        self._widget.setPixmap(QtGui.QPixmap.fromImage(array2qimage(value)))

    @property
    def widget(self):
        return self._widget

    def close(self):
        self._widget.close()


class FigureCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images."""
    def __init__(self, figure, **kwargs):
        self.fig = figure
        super(FigureCanvas, self).__init__(self.fig)

    def resizeEvent(self, event):
        super(FigureCanvas, self).resizeEvent(event)
        # Call to `resize_event` missing in FigureManagerQT.
        # See https://github.com/matplotlib/matplotlib/pull/1585
        self.resize_event()


class DisplayMPL(Display):
    name = 'Matplotlib'
    ndim = 2
    available = has_matplotlib

    def __init__(self, shape, mpp=1.):
        scale = 1
        dpi = mpl.rcParams['figure.dpi']
        h, w = shape[:2]

        figsize = np.array((w, h), dtype=float) / dpi * scale

        self.fig = plt.Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self.ax.set_axis_off()
        self.ax.imshow(np.zeros((h, w, 3), dtype=np.bool),
                       interpolation='nearest', cmap='gray')
        self.ax.autoscale(enable=False)

        self._image_plot = self.ax.images[0]
        self._image_plot.set_clim((0, 255))

    def connect_event(self, event, callback):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvas.mpl_disconnect(callback_id)

    def redraw(self):
        self.canvas.draw_idle()

    def update_image(self, image):
        self._image_plot.set_array(image)

        # Adjust size if new image shape doesn't match the original
        h, w = image.shape[:2]
        self._image_plot.set_extent((0, w, h, 0))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        self.redraw()

    @property
    def widget(self):
        return self.canvas

    def close(self):
        plt.close(self.fig)
        self.canvas.close()


class DisplayMPL_mip(DisplayMPL):
    name = 'Matplotlib (MIP)'
    ndim = 3
    available = has_matplotlib
    def __init__(self, shape, mpp=1.):
        super(DisplayMPL_mip, self).__init__(shape[1:], mpp)

    def update_image(self, image):
        super(DisplayMPL_mip, self).update_image(image.max(0))


class DisplayVolume(Display):
    name = 'OpenGL 3D'
    ndim = 3
    available = has_pyqtgraph

    def __init__(self, shape, mpp=None):
        if len(shape) != 3:
            raise ValueError('Invalid image dimensionality')
        try:
            if len(mpp) != 3:
                mpp = (1.,) * 3
        except:
            mpp = (1.,) * 3

        gl_widget = GLViewWidget()
        self._widget = gl_widget

        self.data_plot = np.zeros(tuple(shape[-1:-4:-1]) + (4,), dtype=np.ubyte)

        volume = GLVolumeItem(self.data_plot)
        volume.scale(mpp[2], -mpp[1], mpp[0])
        shape_um = [m * s for (s, m) in zip(shape, mpp)][::-1]
        volume.translate(*[shape_um[0] / -2, shape_um[1] / 2, shape_um[2] / -2])
        gl_widget.setCameraPosition(azimuth=-45, elevation=30,
                                    distance=max(shape_um) * 1.5)
        gl_widget.addItem(volume)

        box = GLBoxItem(color=(255, 255, 255))
        box.setSize(*shape_um)
        box.translate(*[s / -2 for s in shape_um])
        gl_widget.addItem(box)

        self.volume = volume

    def update_image(self, image):
        new_image = self.volume.data.copy()
        new_image[:, :, :, :3] = image.transpose([2, 1, 0, 3])
        new_image[:, :, :, 3] = np.mean(image, axis=3).T
        self.volume.setData(new_image)

    def close(self):
        self.widget.close()

    @property
    def widget(self):
        return self._widget
