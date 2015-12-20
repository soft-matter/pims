import os
import numpy as np
from pims import pipeline
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.qt import QtGui, QtCore, Signal
from skimage.viewer.plugins import Plugin


def available():
    try:
        from skimage.viewer.qt import has_qt
        if not has_qt:
            raise ImportError
    except ImportError:
        return False
    else:
        return True


class PimsViewer(ImageViewer):
    """Viewer for displaying image sequences from pims readers.

    Parameters
    ----------
    reader : pims reader, optional
        Reader that loads images to be displayed.
    """
    def __init__(self, reader=None):
        self.index = 0
        self._pipeline_func = []

        if reader is None:
            self.reader = np.zeros((1, 128, 128))
        else:
            self.reader = reader

        super(PimsViewer, self).__init__(self.reader[0])

        slider_kws = dict(value=0, low=0, high=len(self.reader) - 1)
        slider_kws['update_on'] = 'release'
        slider_kws['callback'] = self.update_index
        slider_kws['value_type'] = 'int'
        self.slider = Slider('index', **slider_kws)
        self.layout.addWidget(self.slider)
    
    def __add__(self, plugin):
        plugin.pipeline_changed.connect(self._update_pipeline)
        super(PimsViewer, self).__add__(plugin)
        return self
        
    def _update_pipeline(self, plugin_index, kwargs):
        plugin = self.plugins[plugin_index]
        self._pipeline_func[plugin_index] = lambda x: plugin.image_filter(x, **kwargs)
        image = self.original_image.copy()
        for func in self._pipeline_func:
            image = func(image)
        self.image = image

    def save_to_file(self, filename=None):
        raise NotImplementedError()

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
        try:   # attempt to close current reader
            self.reader.close()
        except:
            pass
        self.reader = reader
        self.index = 0
        self.slider.slider.setRange(0, self.num_images - 1)
        self.slider.val = 0
        self.slider.editbox.setText('0')
        self.update_image(self.reader[self.index])

    def update_index(self, name, index):
        """Select image on display using index into image collection."""
        index = min(max(int(round(index)), 0), len(self.reader) - 1)
        if index == self.index:
            return

        self.index = index
        self.slider.val = index
        self.update_image(self.reader[self.index])

    def show(self, main_window=True):
        super(PimsViewer, self).show(main_window)
        
        result = [None] * (len(self._pipeline_func) + 1)
        result[0] = self.reader
        for i, func in enumerate(self._pipeline_func):
            result[i + 1] = pipeline(func)(result[i])

        return result


class PipelinePlugin(Plugin):
    pipeline_changed = Signal(int, dict)

    def attach(self, image_viewer):
        self.setParent(image_viewer)
        self.setWindowFlags(QtCore.Qt.Dialog)

        self.image_viewer = image_viewer
        self.image_viewer.plugins.append(self)

        self._index = len(self.image_viewer._pipeline_func)
        self.image_viewer._pipeline_func += [lambda x: x]

        self.filter_image()

    def output(self):
        return

    def filter_image(self, *widget_arg):
        if self.image_filter is None:
            return

        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.items()])

        self.pipeline_changed.emit(self._index, kwargs)

    def _update_original_image(self, image):
        self._on_new_image(image)
        self.filter_image()
