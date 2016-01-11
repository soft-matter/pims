from __future__ import (division, unicode_literals)
import os
from types import FunctionType
from functools import partial
from itertools import chain

from slicerator import pipeline
from pims.frame import Frame
from pims.base_frames import FramesSequence, FramesSequenceND
from pims.display import to_rgb_uint8

from skimage.viewer.widgets import Slider, CheckBox
from skimage.viewer.qt import Qt, QtWidgets, QtGui, QtCore, Signal, _qt_version
from skimage.viewer.utils import init_qtapp, start_qtapp

from pims.viewer.display import Display, DisplayMPL

if _qt_version == 5:
    from matplotlib.backends.backend_qt5 import TimerQT
elif _qt_version == 4:
    from matplotlib.backends.backend_qt4 import TimerQT


class QDockWidgetCloseable(QtWidgets.QDockWidget):
    """A QDockWidget that emits a signal when closed."""
    close_event_signal = Signal()

    def closeEvent(self, event):
        self.close_event_signal.emit()
        super(QDockWidgetCloseable, self).closeEvent(event)


def _recursive_subclasses(cls):
    "Return all subclasses (and their subclasses, etc.)."
    # Source: http://stackoverflow.com/a/3862957/1221924
    return (cls.__subclasses__() +
        [g for s in cls.__subclasses__() for g in _recursive_subclasses(s)])


class FramesSequence_Wrapper(FramesSequenceND):
    """This class wraps a FramesSequence so that it behaves as a
    FramesSequenceND. All attributes are forwarded to the containing reader."""
    colors_RGB = dict(colors=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])

    def __init__(self, frames_sequence):
        self._reader = frames_sequence
        self._last_i = None
        self._last_frame = None
        shape = frames_sequence.frame_shape
        ndim = len(shape)

        try:
            colors = self._reader.metadata['colors']
            if len(colors) != shape[0]:
                colors = None
        except (KeyError, AttributeError):
            colors = None

        c = None
        z = None
        self.is_RGB = False
        # 2D, grayscale
        if ndim == 2:
            y, x = shape
        # 2D, has colors attribute
        elif ndim == 3 and colors is not None:
            c, y, x = shape
        # 2D, RGB
        elif ndim == 3 and shape[2] in [3, 4]:
            y, x, c = shape
            self.is_RGB = True
        # 2D, is multichannel
        elif ndim == 3 and shape[0] < 5:  # guessing; could be small z-stack
            c, y, x = shape
        # 3D, grayscale
        elif ndim == 3:
            z, y, x = shape
        # 3D, has colors attribute
        elif ndim == 4 and colors is not None:
            c, z, y, x = shape
        # 3D, RGB
        elif ndim == 4 and shape[3] in [3, 4]:
            z, y, x, c = shape
            self.is_RGB = True
        # 3D, is multichannel
        elif ndim == 4 and shape[0] < 5:
            c, z, y, x = shape
        else:
            raise ValueError("Cannot interpret dimensions for a reader of "
                             "shape {0}".format(shape))

        self._init_axis('y', y)
        self._init_axis('x', x)
        self._init_axis('t', len(self._reader))
        if z is not None:
            self._init_axis('z', z)
        if c is not None:
            self._init_axis('c', c)

    def get_frame_2D(self, **ind):
        # do some cacheing
        if self._last_i != ind['t']:
            self._last_i = ind['t']
            self._last_frame = self._reader[ind['t']]
        frame = self._last_frame

        # hack to force colors to RGB:
        if self.is_RGB:
            try:
                frame.metadata.update(self.colors_RGB)
            except AttributeError:
                try:
                    frame_no = frame.frame_no
                except AttributeError:
                    frame_no = None
                frame = Frame(frame, frame_no=frame_no,
                              metadata=self.colors_RGB)

        if 'z' in ind and 'c' in ind and self.is_RGB:
            return frame[ind['z'], :, :, ind['c']]
        elif 'z' in ind and 'c' in ind:
            return frame[ind['c'], ind['z'], :, :]
        elif 'z' in ind:
            return frame[ind['z'], :, :]
        elif 'c' in ind and self.is_RGB:
            return frame[:, :, ind['c']]
        elif 'c' in ind:
            return frame[ind['c'], :, :]
        else:
            return frame

    @property
    def pixel_type(self):
        return self._reader.pixel_type

    def __getattr__(self, attr):
        return self._reader.__getattr__(attr)


class Viewer(QtWidgets.QMainWindow):
    """Viewer for displaying image sequences from pims readers.

    Parameters
    ----------
    reader : pims reader, optional
        Reader that loads images to be displayed.

    Notes
    -----
    The Viewer was partly based on `skimage.viewer.CollectionViewer`
    """
    dock_areas = {'top': Qt.TopDockWidgetArea,
                  'bottom': Qt.BottomDockWidgetArea,
                  'left': Qt.LeftDockWidgetArea,
                  'right': Qt.RightDockWidgetArea}
    _dropped = Signal(list)

    def __init__(self, reader=None, width=800, height=600):
        self.pipelines = []
        self.plugins = []
        self.reader = None
        self.original_image = None
        self.renderer = None
        self.sliders = dict()
        self.channel_tabs = None
        self.slider_dock = None
        self.is_multichannel = False
        self.is_playing = False
        self.mpp = None
        self._timer = None

        # Start main loop
        init_qtapp()
        super(Viewer, self).__init__()

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Python IMage Sequence Viewer")

        open_with_menu = QtWidgets.QMenu('Open with', self)
        for cls in set(chain(_recursive_subclasses(FramesSequence),
                             _recursive_subclasses(FramesSequenceND))):
            open_with_menu.addAction(cls.__name__,
                                     partial(self.open_file, reader_cls=cls))

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('Open file', self.open_file,
                                 Qt.CTRL + Qt.Key_O)
        self.file_menu.addMenu(open_with_menu)
        self.file_menu.addAction('Quit', self.close,
                                 Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.view_menu = QtWidgets.QMenu('&View', self)
        for cls in _recursive_subclasses(Display):
            if cls.available:
                self.view_menu.addAction(cls.name,
                                         partial(self.update_display,
                                                 display_class=cls))
        self.menuBar().addMenu(self.view_menu)

        self.pipeline_menu = QtWidgets.QMenu('&Pipelines', self)
        for pipeline_obj in ViewerPipeline.instances:
            self.pipeline_menu.addAction(pipeline_obj.name,
                                         partial(self.add_pipeline,
                                                 pipeline_obj))
        self.menuBar().addMenu(self.pipeline_menu)

        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QGridLayout(self.main_widget)

        self.show_status_message = self.statusBar().showMessage

        self.resize(width, height)

        self.setAcceptDrops(True)
        self._dropped.connect(self._open_dropped)

        if reader is not None:
            self.update_reader(reader)

    def open_file(self, filename=None, reader_cls=None):
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
        if reader_cls is None:
            reader = pims.open(filename)
        else:
            reader = reader_cls(filename)
        if self.reader is not None:
            self.close_reader()
        self.update_reader(reader)
        self.show_status_message('Opened {}'.format(filename))

    def update_reader(self, reader):
        """Load a new reader into the Viewer."""
        if not isinstance(reader, FramesSequenceND):
            reader = FramesSequence_Wrapper(reader)
        self.reader = reader
        self.reader.iter_axes = ''
        index = reader.default_coords.copy()

        # try to readout calibration
        try:
            mpp = reader.calibration
        except AttributeError:
            mpp = 1.
        if mpp is None:
            mpp = 1.
        if 'z' in self.reader.sizes:
            try:
                mppZ = reader.calibrationZ
            except AttributeError:
                mppZ = mpp
            self.mpp = (mppZ, mpp, mpp)
        else:
            self.mpp = (mpp, mpp)

        # add color tabs
        if 'c' in reader.sizes:
            self.is_multichannel = True
            self.channel_tabs = QtWidgets.QTabBar(self)
            self.main_layout.addWidget(self.channel_tabs, 1, 0)
            self.channel_tabs.addTab('all')
            for c in range(reader.sizes['c']):
                self.channel_tabs.addTab(str(c))
                self.channel_tabs.setShape(QtWidgets.QTabBar.RoundedSouth)
                self.channel_tabs.currentChanged.connect(self.channel_tab_callback)
        else:
            self.is_multichannel = False

        # add sliders
        self.sliders = dict()
        for axis in reader.sizes:
            if axis in ['x', 'y', 'c'] or reader.sizes[axis] <= 1:
                continue
            self.sliders[axis] = Slider(axis, low=0, high=reader.sizes[axis] - 1,
                                        value=index[axis], update_on='release',
                                        value_type='int',
                                        callback=self.slider_callback)

        if len(self.sliders) > 0:
            slider_widget = QtWidgets.QWidget()
            slider_layout = QtWidgets.QGridLayout(slider_widget)
            for i, axis in enumerate(self.sliders):
                slider_layout.addWidget(self.sliders[axis], i, 0)
                if axis == 't':
                    checkbox = CheckBox('play', self.is_playing,
                                        callback=self.play_callback)
                    slider_layout.addWidget(checkbox, i, 1)

            self.slider_dock = QtWidgets.QDockWidget()
            self.slider_dock.setWidget(slider_widget)
            self.slider_dock.setWindowTitle('Axes sliders')
            self.slider_dock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                                         QtGui.QDockWidget.DockWidgetMovable)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.slider_dock)

        self.index = index
        self.update_display()

    def close_reader(self):
        self.reader.close()
        self.renderer.close()
        if self.slider_dock is not None:
            self.slider_dock.close()
        if self.channel_tabs is not None:
            self.channel_tabs.close()

    def update_display(self, display_class=None):
        """Change display mode."""
        if display_class is None:
            display_class = DisplayMPL

        shape = [self.reader.sizes['y'], self.reader.sizes['x']]
        if display_class.ndim == 3:
            try:
                shape = [self.reader.sizes['z']] + shape
            except KeyError:
                raise KeyError('z axis does not exist: cannot display in 3D')

        if self.renderer is not None:
            self.renderer.close()
        self.renderer = display_class(shape, self.mpp)

        # connect the statusbar, only for matplotlib display mode
        if hasattr(self.renderer, 'connect_event'):
            self.renderer.connect_event('motion_notify_event',
                                        self._update_mouse_position)

        self.renderer.widget.setSizePolicy(QtGui.QSizePolicy.Ignored,
                                           QtGui.QSizePolicy.Ignored)
        self.renderer.widget.updateGeometry()
        self.main_layout.addWidget(self.renderer.widget, 0, 0)
        self.update_index()

    def update_reader_axes(self):
        """Make sure that the reader bundle_axes settings are correct"""
        try:
            ndim = self.renderer.ndim
        except AttributeError:
            ndim = 2
        if ndim == 3 and 'z' not in self.reader.sizes:
            raise ValueError('z axis does not exist: cannot display in 3D')
        if self.is_multichannel and 'c' not in self.reader.sizes:
            raise ValueError('c axis does not exist: cannot display multicolor')

        bundle_axes = ''
        if self.is_multichannel:
            bundle_axes += 'c'
        if ndim == 3:
            bundle_axes += 'z'
        self.reader.bundle_axes = bundle_axes + 'yx'

    def update_index(self, index=None):
        """Select image on display using index into reader."""
        if index == self.index:
            return
        elif index is None:
            index = self.index
        else:
            self.index = index

        self.update_reader_axes()

        for name in self.sliders:
            self.sliders[name].val = index[name]
        self.reader.default_coords.update(index)
        image = self.reader[0]

        self.original_image = image
        self.update_image()

    def update_image(self):
        """Update the image that is being viewed."""
        if self.original_image is None:
            return
        image = self.original_image.copy()
        for func in self.pipelines:
            image = func(image)
        self.image = image

    def update_pipeline(self, index, func):
        """This is called by the ViewerPipeline to update its effect."""
        self.pipelines[index] = func
        self.update_image()

    @property
    def image(self):
        """The image that is being displayed"""
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        self.renderer.image = to_rgb_uint8(image, autoscale=True)

    def slider_callback(self, name, index):
        """Callback function for axes sliders."""
        if self.index[name] == index:
            return
        new_index = self.index.copy()
        new_index[name] = index
        self.update_index(new_index)

    def channel_tab_callback(self, index):
        """Callback function for channel tabs."""
        self.is_multichannel = index == 0
        if index > 0:  # monochannel: update channel field
            self.index['c'] = index - 1  # because 0 is multichannel

        self.update_index()

    def play_callback(self, name, value):
        """Callback function for play checkbox."""
        if value == self.is_playing:
            return
        if value:
            self.play()
        else:
            self.stop()

    def play(self, fps=None):
        """Start the movie."""
        if fps is None:
            try:
                fps = self.reader.frame_rate
            except AttributeError:
                fps = 25.
        self._timer = TimerQT(interval=1000/fps)
        self._timer.add_callback(self.next_index)
        self.is_playing = True
        self._timer.start()

    def stop(self):
        """Stop the movie."""
        if self._timer is None:
            return
        self.is_playing = False
        self._timer.stop()
        self._timer = None

    def next_index(self):
        """Increase time index by one. At the end, restart."""
        if self.index['t'] + 1 < self.reader.sizes['t']:
            self.index['t'] += 1
        else:
            self.index['t'] = 0
        self.update_index()

    def add_pipeline(self, pipeline_obj):
        """Add ViewerPipeline to the Viewer"""
        pipeline_obj.pipeline_changed.connect(self.update_pipeline)
        pipeline_obj.attach(self)

        if pipeline_obj.dock:
            location = self.dock_areas[pipeline_obj.dock]
            dock_location = Qt.DockWidgetArea(location)
            dock = QDockWidgetCloseable()
            dock.setWidget(pipeline_obj)
            dock.setWindowTitle(pipeline_obj.name)
            dock.close_event_signal.connect(pipeline_obj.close_pipeline)
            dock.setSizePolicy(QtGui.QSizePolicy.Fixed,
                               QtGui.QSizePolicy.MinimumExpanding)
            self.addDockWidget(dock_location, dock)

    def __add__(self, pipeline_obj):
        self.add_pipeline(pipeline_obj)
        return self

    def closeEvent(self, event):
        # obtain the result values before everything is closed
        result = [None] * (len(self.pipelines) + 1)
        result[0] = self.reader
        for i, func in enumerate(self.pipelines):
            result[i + 1] = pipeline(func)(result[i])
        self._result_value = result

        super(Viewer, self).closeEvent(event)

    def show(self, main_window=True):
        """Show Viewer and attached ViewerPipelines."""
        self.move(0, 0)
        for p in self.plugins:
            p.show()
        super(Viewer, self).show()
        self.activateWindow()
        self.raise_()
        if main_window:
            start_qtapp()

        return self._result_value

    def _update_mouse_position(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            x = int(event.xdata + 0.5)
            y = int(event.ydata + 0.5)
            try:
                if self.is_multichannel:
                    val = self.image[:, y, x]
                else:
                    val = self.image[y, x]
                msg = "%4s @ [%4s, %4s]" % (val, x, y)
            except IndexError:
                msg = ""
            self.show_status_message(msg)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self._dropped.emit(links)
        else:
            event.ignore()

    def _open_dropped(self, links):
        for fn in links:
            if os.path.exists(fn):
                self.open_file(fn)
                break


class ViewerPipeline(QtWidgets.QDialog):
    """Base class for viewing the result of image processing inside the Viewer.

    The ViewerPipeline class connects an image filter (or another function) to
    the Viewer. The Viewer returns a reader object that has the function applied
    with parameters set inside the Viewer.

    Parameters
    ----------
    pipeline_func : function
        Function that processes the image. It should not change the image shape.
    name : string
        Name of pipeline. This is displayed as the window title.
    height, width : int
        Size of plugin window in pixels. Note that Qt will automatically resize
        a window to fit components. So if you're adding rows of components, you
        can leave `height = 0` and just let Qt determine the final height.
    dock : {bottom, top, left, right}
        Default docking area

    Examples
    --------
    >>> import numpy as np
    >>> from pims.viewer import Viewer, ViewerPipeline, Slider
    >>>
    >>> def add_noise(img, noise_level):
    >>>     return img + np.random.random(img.shape) * noise_level
    >>>
    >>> image_reader = np.zeros((1, 512, 512), dtype=np.uint8)
    >>>
    >>> AddNoise = ViewerPipeline(add_noise) + Slider('noise_level', 0, 100, 0)
    >>> viewer = Viewer(image_reader)
    >>> viewer += AddNoise
    >>> original, noise_added = viewer.show()

    Notes
    -----
    ViewerPipeline was partly based on `skimage.viewer.plugins.Plugin`
    """
    # Signals used when viewers are linked to the Plugin output.
    pipeline_changed = Signal(int, FunctionType)
    instances = []
    def __init__(self, pipeline_func, name=None, height=0, width=400,
                 dock='bottom'):
        init_qtapp()
        super(ViewerPipeline, self).__init__()

        self.dock = dock

        self.image_viewer = None
        self.pipeline_func = pipeline_func

        if name is None:
            self.name = pipeline_func.__name__
        else:
            self.name = name

        self.setWindowTitle(self.name)
        self.layout = QtWidgets.QGridLayout(self)
        self.resize(width, height)
        self.row = 0

        self.arguments = []
        self.keyword_arguments = {}

        # the class keeps a list of its instances. this means that the objects
        # will not get garbage collected.
        ViewerPipeline.instances.append(self)

    def attach(self, image_viewer):
        """Attach the pipeline to an ImageViewer.

        Note that the ImageViewer will automatically call this method when the
        plugin is added to the ImageViewer. For example:

            viewer += Plugin(pipeline_func)

        Also note that `attach` automatically calls the filter function so that
        the image matches the filtered value specified by attached widgets.
        """
        self.setParent(image_viewer)
        self.setWindowFlags(QtCore.Qt.Dialog)

        self.image_viewer = image_viewer
        self.image_viewer.plugins.append(self)

        self.pipeline_index = len(self.image_viewer.pipelines)
        self.image_viewer.pipelines += [None]

        #self.rejected.connect(lambda: self.close())

        self.filter_image()

    def add_widget(self, widget):
        """Add widget to pipeline.

        Alternatively, you can use simple addition to add widgets:

            plugin += Slider('param_name', low=0, high=100)

        Widgets can adjust arguments of the pipeline function, as specified by
        the Widget's `ptype`.
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
        """Send the changed pipeline function to the Viewer. """
        # `widget_arg` is passed by the active widget but is unused since all
        # filter arguments are pulled directly from attached the widgets.
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.items()])
        func = lambda x: self.pipeline_func(x, *self.arguments, **kwargs)
        self.pipeline_changed.emit(self.pipeline_index, func)

    def _get_value(self, param):
        # If param is a widget, return its `val` attribute.
        return param if not hasattr(param, 'val') else param.val

    def show(self, main_window=True):
        """Show plugin."""
        super(ViewerPipeline, self).show()
        self.activateWindow()
        self.raise_()

    def close_pipeline(self):
        """Close the plugin and clean up."""
        if self in self.image_viewer.plugins:
            self.image_viewer.plugins.remove(self)

        # delete the pipeline
        if self.pipeline_index is not None:
            del self.image_viewer.pipelines[self.pipeline_index]
        # decrease pipeline_index for the other pipelines
        for plugin in self.image_viewer.plugins:
            try:
                if plugin.pipeline_index > self.pipeline_index:
                    plugin.pipeline_index -= 1
            except AttributeError:
                pass  # no pipeline_index

        self.image_viewer.update_image()
        self.close()
