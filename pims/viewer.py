import os
import numpy as np
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.qt import QtWidgets, QtGui, QtCore
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

    Select the displayed frame of the image sequence using the slider or
    with the following keyboard shortcuts:

        left/right arrows
            Previous/next image in sequence.
        number keys, 0--9
            0% to 90% of sequence. For example, "5" goes to the image in the
            middle (i.e. 50%) of the sequence.
        home/end keys
            First/last image in sequence.

    Parameters
    ----------
    reader : pims reader, optional
        Reader that loads images to be displayed.
    """
    def __init__(self, reader=None):
        self.index = 0

        if reader is None:
            self.reader = np.zeros((1, 128, 128))
            self.num_images = 1
        else:
            self.reader = reader
            self.num_images = len(self.reader)

        super(PimsViewer, self).__init__(self.reader[0])

        slider_kws = dict(value=0, low=0, high=self.num_images - 1)
        slider_kws['update_on'] = 'release'
        slider_kws['callback'] = self.update_index
        slider_kws['value_type'] = 'int'
        self.slider = Slider('index', **slider_kws)
        self.layout.addWidget(self.slider)

        self.original_reader = reader

    def update_image(self):
        self.image = self.reader[self.index]

    def update_reader(self, reader):
        try:   # attempt to close current reader
            self.reader.close()
        except:
            pass
        self.original_reader = reader
        self.reader = reader
        self.index = 0
        self.num_images = len(self.reader)
        self.slider.slider.setRange(0, self.num_images - 1)
        self.slider.val = 0
        self.slider.editbox.setText('0')
        self.update_image()

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
        self.update_reader(pims.open(filename))

    def update_index(self, name, index):
        """Select image on display using index into image collection."""
        index = int(round(index))

        if index == self.index:
            return

        # clip index value to collection limits
        index = max(index, 0)
        index = min(index, self.num_images - 1)

        self.index = index
        self.slider.val = index
        self.update_image()

    def keyPressEvent(self, event):
        if type(event) == QtWidgets.QKeyEvent:
            key = event.key()
            # Number keys (code: 0 = key 48, 9 = key 57) move to deciles
            if 48 <= key < 58:
                index = 0.1 * int(key - 48) * self.num_images
                self.update_index('', index)
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def closeEvent(self, event):
        try:   # attempt to close current reader
            self.reader.close()
        except:
            pass
        self.close()


class PimsPlugin(Plugin):
    """Example:
    @pims.pipeline
    def convert_to_grey(img, r, g, b):
        grey = (img * [r, g, b]).sum(2)
        return grey.astype(img.dtype)

    viewer = PimsViewer(pims.open(\path\to\video'))
    viewer += (PimsPlugin(convert_to_grey) + Slider('r', 0, 1, 0.2125) +
               Slider('g', 0, 1, 0.7154) + Slider('b', 0, 1, 0.0721))
    """
    def attach(self, image_viewer):
        self.setParent(image_viewer)
        self.setWindowFlags(QtCore.Qt.Dialog)

        self.image_viewer = image_viewer
        self.image_viewer.plugins.append(self)

        self.arguments = []

        # Call filter so that filtered image matches widget values
        self.filter_image()

    def filter_image(self, *widget_arg):
        """Call `image_filter` with widget args and kwargs

        Note: `display_filtered_image` is automatically called.
        """
        # `widget_arg` is passed by the active widget but is unused since all
        # filter arguments are pulled directly from attached the widgets.

        if self.image_filter is None:
            return
        arguments = [self._get_value(a) for a in self.arguments]
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.items()])
        filtered = self.image_filter(self.image_viewer.original_reader,
                                     *arguments, **kwargs)

        self.image_viewer.reader = filtered
        self.image_viewer.update_image()

    def display_filtered_image(self, reader):
        """Display the filtered image on image viewer.

        If you don't want to simply replace the displayed image with the
        filtered image (e.g., you want to display a transparent overlay),
        you can override this method.
        """
        self.image_viewer.reader = reader

    def _update_original_image(self, image):
        self.filter_image()

    def show(self, main_window=True):
        """Show plugin."""
        super(Plugin, self).show()
        self.activateWindow()
        self.raise_()

        # Emit signal with x-hint so new windows can be displayed w/o overlap.
        size = self.frameGeometry()
        x_hint = size.x() + size.width()
        self._started.emit(x_hint)

    def output(self):
        return self.image_viewer.reader
