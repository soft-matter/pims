try:
    from skimage.viewer.qt import has_qt
    viewer_available = has_qt
except ImportError:
    viewer_available = False

if viewer_available:
    from pims.viewer.viewer import Viewer, ViewerPipeline, ViewerPlotting
    from skimage.viewer.widgets import Text, ComboBox, Slider, CheckBox
else:
    def viewer_not_available(*args, **kwargs):
        raise ImportError(
            "The PIMS viewer requires scikit-image, matplotlib, and PyQt.")
    Viewer = ViewerPipeline = ViewerPlotting = viewer_not_available
