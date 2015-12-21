import os
import numpy as np
from pims import pipeline, to_rgb, FramesSequenceND
from types import FunctionType
from skimage.viewer.widgets import Slider, CheckBox
from skimage.viewer.qt import (Qt, QtWidgets, QtGui, QtCore, Signal, has_qt,
                               FigureCanvasQTAgg)
from skimage.viewer.utils import (init_qtapp, start_qtapp)
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

def available():
    try:
        if not has_qt:
            raise ImportError
    except ImportError:
        return False
    else:
        return True


class DockWidgetCloseable(QtWidgets.QDockWidget):
    close_event_signal = Signal()
    def closeEvent(self, event):
        self.close_event_signal.emit()
        super(DockWidgetCloseable, self).closeEvent(event)


def normalize(arr):
    """This normalizes an array to values between 0 and 1.

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    ndarray of float
        normalized array
    """
    ptp = arr.max() - arr.min()
    # Handle edge case of a flat image.
    if ptp == 0:
        ptp = 1
    scaled_arr = (arr - arr.min()) / ptp
    return scaled_arr


def to_rgb_uint8(image, autoscale=True):
    ndim = image.ndim
    shape = image.shape
    try:
        colors = image.colors
    except AttributeError:
        colors = None

    # 2D, grayscale
    if ndim == 2:
        pass
    # 2D, has colors attribute
    elif ndim == 3 and colors is not None:
        image = to_rgb(image, colors, False)
    # 2D, RGB
    elif ndim == 3 and shape[2] in [3, 4]:
        pass
    # 2D, is multichannel
    elif ndim == 3 and shape[0] < 5:  # guessing; could be small z-stack
        image = to_rgb(image, None, False)
    # 3D, grayscale
    elif ndim == 3:
        pass
    # 3D, has colors attribute
    elif ndim == 4 and colors is not None:
        image = to_rgb(image, colors, False)
    # 3D, RGB
    elif ndim == 4 and shape[3] in [3, 4]:
        pass
    # 3D, is multichannel
    elif ndim == 4 and shape[0] < 5:
        image = to_rgb(image, None, False)
    else:
        raise ValueError("No display possible for frames of shape {0}".format(shape))

    if autoscale:
        image = (normalize(image) * 255).astype(np.uint8)
    elif image.dtype is not np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
            # sometimes 12-bit images are stored as unsigned 16-bit
            if max_value == 2**16 - 1 and image.max() < 2**12:
                max_value = 2**12 - 1
            image = (image / max_value * 255).astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)

    if image.shape[-1] != 3:
        image = np.repeat(image[..., np.newaxis], 3, axis=image.ndim)

    return image


class BlitManager(object):
    """Object that manages blits on an axes"""
    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('draw_event', self.on_draw_event)
        self.background = None
        self.artists = []

    def add_artists(self, artists):
        self.artists.extend(artists)
        self.redraw()

    def remove_artists(self, artists):
        for artist in artists:
            self.artists.remove(artist)

    def on_draw_event(self, event=None):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.draw_artists()

    def redraw(self):
        if self.background is not None:
            self.canvas.restore_region(self.background)
            self.draw_artists()
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def draw_artists(self):
        for artist in self.artists:
            self.ax.draw_artist(artist)


class FigureCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images."""
    def __init__(self, figure, **kwargs):
        self.fig = figure
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def resizeEvent(self, event):
        FigureCanvasQTAgg.resizeEvent(self, event)
        # Call to `resize_event` missing in FigureManagerQT.
        # See https://github.com/matplotlib/matplotlib/pull/1585
        self.resize_event()


class DisplayMPL(object):
  #  mouse_position = Signal(int, int)
    def __init__(self, shape, useblit=True):
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

        self.useblit = useblit
        if useblit:
            self._blit_manager = BlitManager(self.ax)

        self._image_plot = self.ax.images[0]
        self._image_plot.set_clim((0, 255))

        self.connect_event('motion_notify_event', self.update_mouse_position)

    def connect_event(self, event, callback):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvas.mpl_disconnect(callback_id)

    def redraw(self):
        if self.useblit:
            self._blit_manager.redraw()
        else:
            self.canvas.draw_idle()

    def update_image(self, image):
        self.image = image
        self._image_plot.set_array(image)

        # Adjust size if new image shape doesn't match the original
        h, w = image.shape[:2]
        self._image_plot.set_extent((0, w, h, 0))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        if self.useblit:
            self._blit_manager.background = None

        self.redraw()

    @property
    def size(self):
        result = self.canvas.sizeHint()
        return result.height(), result.width()

    @property
    def widget(self):
        return self.canvas

    def close(self):
        plt.close(self.fig)
        self.canvas.close()

    def update_mouse_position(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            x = int(event.xdata + 0.5)
            y = int(event.ydata + 0.5)
       #     self.mouse_position.emit(y, x)

class DisplayVolume(object):
    status_msg = Signal(str)
    def __init__(self, shape):
        widget = gl.GLViewWidget()
        widget.opts['distance'] = 50
        widget.show()

        self.data_plot = np.zeros(tuple(shape[:3]) + (4,), dtype=np.ubyte)

        volume = gl.GLVolumeItem(self.data_plot)
        center = [int(s // -2) for s in shape[:3]]
        volume.translate(*center)
        widget.addItem(volume)

        ax = gl.GLAxisItem()
        widget.addItem(ax)

        self.volume = volume
        self.widget = widget

    def update_image(self, image):
        self.image = image
        self.data_plot[:, :, :, :3] = image
        self.data_plot[:, :, :, 3] = np.mean(image, axis=3)

    @property
    def size(self):
        return (512, 512)

    def close(self):
        pass


class Viewer(QtWidgets.QMainWindow):
    """Viewer for displaying image sequences from pims readers.

    Parameters
    ----------
    reader : pims reader, optional
        Reader that loads images to be displayed.
    """
    dock_areas = {'top': Qt.TopDockWidgetArea,
                  'bottom': Qt.BottomDockWidgetArea,
                  'left': Qt.LeftDockWidgetArea,
                  'right': Qt.RightDockWidgetArea}

    def __init__(self, reader=None, useblit=True):
        self.pipelines = []
        self.plugins = []
        self.reader = None
        self.renderer = None
        self.sliders = dict()
        self.useblit = useblit
        self.is_multichannel = False
        self.is_volume = False
        self.is_ND = False

        # Start main loop
        init_qtapp()
        super(Viewer, self).__init__()

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Python IMage Sequence Viewer")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('Open file', self.open_file,
                                 Qt.CTRL + Qt.Key_O)
        self.file_menu.addAction('Save to file', self.save_to_file,
                                 Qt.CTRL + Qt.Key_S)
        self.file_menu.addAction('Quit', self.close,
                                 Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)


        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        self._add_widget_size(self.file_menu, dimension='height')

        if reader is not None:
            self.update_reader(reader)

    def __add__(self, plugin):
        """Add plugin to ImageViewer"""
        plugin.pipeline_changed.connect(self.update_pipeline)
        plugin.attach(self)

        if plugin.dock:
            location = self.dock_areas[plugin.dock]
            dock_location = Qt.DockWidgetArea(location)
            dock = DockWidgetCloseable()
            dock.setWidget(plugin)
            dock.setWindowTitle(plugin.name)
            dock.close_event_signal.connect(plugin.cancel_pipeline)
            self.addDockWidget(dock_location, dock)

            horiz = (self.dock_areas['left'], self.dock_areas['right'])
            dimension = 'width' if location in horiz else 'height'
            self._add_widget_size(dock, dimension=dimension)

        return self

    def _add_widget_size(self, widget, dimension='width'):
        widget_size = widget.sizeHint()
        viewer_size = self.frameGeometry()

        dx = 0
        dy = 72
        if dimension == 'width':
            dx += widget_size.width()
        elif dimension == 'height':
            dy += widget_size.height()
        w = viewer_size.width()
        h = viewer_size.height()
        self.resize(w + dx, h + dy)

    def open_file(self, filename=None):
        """Open image file and display in viewer."""
        if filename is None:
            try:
                cur_dir = os.path.dirname(self.reader.filename)
            except AttributeError:
                cur_dir = ''
            filename = QtGui.QFileDialog.getOpenFileName(directory=cur_dir)
            if isinstance(filename, tuple):
                # Handle discrepancy between PyQt4 and PySide APIs.
                filename = filename[0]
        if filename is None or len(filename) == 0:
            return
        import pims
        reader = pims.open(filename)
        if self.reader is not None:
            self.close_reader()
        self.update_reader(reader)

    def close_reader(self):
        self.reader.close()
        self.renderer.close()
        self.slider.close()

    def update_display(self, shape, mode='mpl'):
        if mode == 'mpl':
            self.renderer = DisplayMPL(shape)
        elif mode == 'volume':
            self.renderer = DisplayVolume(shape)
        else:
            raise ValueError('Unknown display mode')

        # resize the window
        status_bar = self.statusBar()
        statusbar_height = status_bar.sizeHint().height()
        canvas_height, canvas_width = self.renderer.size
        self.resize(canvas_width, canvas_height + statusbar_height)
        # connect the statusbar
        self.status_message = status_bar.showMessage
        #self.renderer.mouse_position.connect(self.update_status_bar)

        self.layout.addWidget(self.renderer.widget)

    def update_reader(self, reader):
        self.reader = reader
        if isinstance(reader, FramesSequenceND):
            reader.bundle_axes = 'yx'
            shape = reader.frame_shape
            reader.iter_axes = ''
            index = reader.default_coords.copy()
            self.is_ND = True
        else:
            index = 0
            shape = reader.frame_shape
            # identify shape
            if len(shape) == 2:
                pass
            elif (len(shape) == 3) and (shape[2] in [3, 4]):
                shape = shape[:2]
            elif (len(shape) == 3) and (shape[0] < 5):
                shape = shape[1:]
            else:
                raise ValueError('Unsupported image shape')
            self.is_ND = False

        self.update_display(shape, 'mpl')

        # add sliders
        slider_widget = QtWidgets.QWidget()
        slider_layout = QtWidgets.QGridLayout(slider_widget)
        self.sliders = dict()
        if self.is_ND:
            for axis in reader.sizes:
                if axis in ['x', 'y'] or reader.sizes[axis] <= 1:
                    continue
                slider = Slider(axis, low=0, high=reader.sizes[axis] - 1,
                                value=index[axis], update_on='release',
                                value_type='int',
                                callback=self.slider_callback_ND)
                slider_layout.addWidget(slider, len(self.sliders), 0)
                self.sliders[axis] = slider
                if axis == 'c':
                    checkbox = CheckBox('multichannel', True,
                                        callback=self.display_multichannel)
                    self.display_multichannel(None, True)
                    slider_layout.addWidget(checkbox, len(self.sliders)-1, 1)
                if axis == 'z':
                    checkbox = CheckBox('volume', False,
                                        callback=self.display_volume)
                    slider_layout.addWidget(checkbox, len(self.sliders)-1, 1)
        elif len(self.reader) > 1:
            slider = Slider('index', low=0, high=len(self.reader) - 1,
                            value=index, update_on='release', value_type='int',
                            callback=self.slider_callback)
            slider_layout.addWidget(slider, 0, 0)
            self.sliders['index'] = slider

        if len(self.sliders) > 0:
            dock_location = Qt.DockWidgetArea(self.dock_areas['bottom'])
            dock = DockWidgetCloseable()
            dock.setWidget(slider_widget)
            dock.setWindowTitle('Axes sliders')
            dock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                             QtGui.QDockWidget.DockWidgetMovable)
            self.addDockWidget(dock_location, dock)
            self._add_widget_size(dock, dimension='height')

        self.index = None
        self.update_index(index)

    def slider_callback(self, name, index):
        self.update_index(index)

    def slider_callback_ND(self, name, index):
        if self.index[name] == index:
            return
        new_index = self.index.copy()
        new_index[name] = index
        self.update_index(new_index)

    def display_multichannel(self, name, value):
        if value:
            self.is_multichannel = True
            self.sliders['c'].setDisabled(True)
            self.reader.bundle_axes = 'cyx'
        else:
            self.is_multichannel = False
            self.sliders['c'].setDisabled(False)
            self.reader.bundle_axes = 'yx'
        if name is not None:
            self.update_index(self.index)

    def display_volume(self, name, value):
        if value == self.is_volume:
            return
        if value:
            self.renderer.close()
            self.reader.bundle_axes = 'zyx'
            self.update_display(self.reader.frame_shape, 'volume')
            self.sliders['z'].setDisabled(True)
        else:
            self.renderer.close()
            self.reader.bundle_axes = 'yx'
            self.update_display(self.reader.frame_shape, 'mpl')
            self.sliders['z'].setDisabled(False)
        if name is not None:
            self.update_index(self.index)


    def update_index(self, index):
        """Select image on display using index into image collection."""
        self.index = index

        if self.is_ND:
            for name in self.sliders:
                self.sliders[name].val = index[name]
            self.reader.default_coords.update(index)
            image = self.reader[0]
        else:
            self.sliders['index'].val = index
            image = self.reader[index]

        self.original_image = image
        self.update_image()

    def update_image(self):
        image = self.original_image.copy()
        for func in self.pipelines:
            image = func(image)
        self.image = image

    def update_pipeline(self, index, func):
        self.pipelines[index] = func
        self.update_image()

    def save_to_file(self, filename=None):
        raise NotImplementedError()

    def closeEvent(self, event):
        self.close()

    def show(self, main_window=True):
        """Show ImageViewer and attached plugins.

        This behaves much like `matplotlib.pyplot.show` and `QWidget.show`.
        """
        self.move(0, 0)
        for p in self.plugins:
            p.show()
        super(Viewer, self).show()
        self.activateWindow()
        self.raise_()
        if main_window:
            start_qtapp()

        result = [None] * (len(self.pipelines) + 1)
        result[0] = self.reader
        for i, func in enumerate(self.pipelines):
            result[i + 1] = pipeline(func)(result[i])

        return result

    def redraw(self):
        self.renderer.redraw()

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        self.renderer.update_image(to_rgb_uint8(image, autoscale=True))

    def update_status_bar(self, y, x):
        try:
            if self.is_multichannel:
                val = self.image[:, y, x]
            else:
                val = self.image[y, x]
            msg = "%4s @ [%4s, %4s]" % (val, x, y)
        except IndexError:
            msg = ""
        self.status_message(msg)


class PipelinePlugin(QtWidgets.QDialog):
    """Base class for plugins that interact with an ImageViewer.

    A plugin connects an image filter (or another function) to an image viewer.
    Note that a Plugin is initialized *without* an image viewer and attached in
    a later step. See example below for details.

    Parameters
    ----------
    image_viewer : ImageViewer
        Window containing image used in measurement/manipulation.
    image_filter : function
        Function that gets called to update image in image viewer. This value
        can be `None` if, for example, you have a plugin that extracts
        information from an image and doesn't manipulate it. Alternatively,
        this function can be defined as a method in a Plugin subclass.
    height, width : int
        Size of plugin window in pixels. Note that Qt will automatically resize
        a window to fit components. So if you're adding rows of components, you
        can leave `height = 0` and just let Qt determine the final height.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        Matplotlib backends. If None, set to True when using Agg backend.
        This only has an effect if you draw on top of an image viewer.

    Attributes
    ----------
    image_viewer : ImageViewer
        Window containing image used in measurement.
    name : str
        Name of plugin. This is displayed as the window title.
    artist : list
        List of Matplotlib artists and canvastools. Any artists created by the
        plugin should be added to this list so that it gets cleaned up on
        close.

    Examples
    --------
    >>> from skimage.viewer import ImageViewer
    >>> from skimage.viewer.widgets import Slider
    >>> from skimage import data
    >>>
    >>> plugin = Plugin(image_filter=lambda img,
    ...                 threshold: img > threshold) # doctest: +SKIP
    >>> plugin += Slider('threshold', 0, 255)       # doctest: +SKIP
    >>>
    >>> image = data.coins()
    >>> viewer = ImageViewer(image)       # doctest: +SKIP
    >>> viewer += plugin                  # doctest: +SKIP
    >>> thresholded = viewer.show()[0][0] # doctest: +SKIP

    The plugin will automatically delegate parameters to `image_filter` based
    on its parameter type, i.e., `ptype` (widgets for required arguments must
    be added in the order they appear in the function). The image attached
    to the viewer is **automatically passed as the first argument** to the
    filter function.

    #TODO: Add flag so image is not passed to filter function by default.

    `ptype = 'kwarg'` is the default for most widgets so it's unnecessary here.

    """
    # Signals used when viewers are linked to the Plugin output.
    pipeline_changed = Signal(int, FunctionType)
    _started = Signal(int)

    def __init__(self, image_filter, name=None,
                 height=0, width=400, useblit=True,
                 dock='bottom'):
        init_qtapp()
        super(PipelinePlugin, self).__init__()

        self.dock = dock

        self.image_viewer = None
        self.image_filter = image_filter

        if name is None:
            self.name = image_filter.__name__
        else:
            self.name = name

        self.setWindowTitle(self.name)
        self.layout = QtWidgets.QGridLayout(self)
        self.resize(width, height)
        self.row = 0

        self.arguments = []
        self.keyword_arguments = {}

        self.useblit = useblit

    def attach(self, image_viewer):
        """Attach the plugin to an ImageViewer.

        Note that the ImageViewer will automatically call this method when the
        plugin is added to the ImageViewer. For example::

            viewer += Plugin(...)

        Also note that `attach` automatically calls the filter function so that
        the image matches the filtered value specified by attached widgets.
        """
        self.setParent(image_viewer)
        self.setWindowFlags(QtCore.Qt.Dialog)

        self.image_viewer = image_viewer
        self.image_viewer.plugins.append(self)

        self.pipeline_index = len(self.image_viewer.pipelines)
        self.image_viewer.pipelines += [None]

        self.rejected.connect(lambda: self.close())

        self.filter_image()

    def add_widget(self, widget):
        """Add widget to plugin.

        Alternatively, Plugin's `__add__` method is overloaded to add widgets::

            plugin += Widget(...)

        Widgets can adjust required or optional arguments of filter function or
        parameters for the plugin. This is specified by the Widget's `ptype`.
        """
        if widget.ptype == 'kwarg':
            name = widget.name.replace(' ', '_')
            self.keyword_arguments[name] = widget
            widget.callback = self.filter_image
        elif widget.ptype == 'arg':
            self.arguments.append(widget)
            widget.callback = self.filter_image
        elif widget.ptype == 'plugin':
            widget.callback = self.update_plugin
        widget.plugin = self
        self.layout.addWidget(widget, self.row, 0)
        self.row += 1

    def __add__(self, widget):
        self.add_widget(widget)
        return self

    def filter_image(self, *widget_arg):
        """Call `image_filter` with widget args and kwargs

        Note: `display_filtered_image` is automatically called.
        """
        # `widget_arg` is passed by the active widget but is unused since all
        # filter arguments are pulled directly from attached the widgets.
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.items()])
        func = lambda x: self.image_filter(x, *self.arguments, **kwargs)
        self.pipeline_changed.emit(self.pipeline_index, func)

    def _get_value(self, param):
        # If param is a widget, return its `val` attribute.
        return param if not hasattr(param, 'val') else param.val

    def update_plugin(self, name, value):
        """Update keyword parameters of the plugin itself.

        These parameters will typically be implemented as class properties so
        that they update the image or some other component.
        """
        setattr(self, name, value)

    def show(self, main_window=True):
        """Show plugin."""
        super(PipelinePlugin, self).show()
        self.activateWindow()
        self.raise_()

        # Emit signal with x-hint so new windows can be displayed w/o overlap.
        size = self.frameGeometry()
        x_hint = size.x() + size.width()
        self._started.emit(x_hint)

    def cancel_pipeline(self):
        """Close the plugin and clean up."""
        if self in self.image_viewer.plugins:
            self.image_viewer.plugins.remove(self)
        if self.pipeline_index is not None:
            del self.image_viewer.pipelines[self.pipeline_index]
        self.image_viewer.update_image()
        super(PipelinePlugin, self).close()
