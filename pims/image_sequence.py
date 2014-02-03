import os
from scipy.ndimage import imread
from pims.base_frames import FramesSequence


class ImageSequence(FramesSequence):
    """Iterable object that returns frames of video as numpy arrays.

    Parameters
    ----------
    directory : string
    gray : Convert color image to grayscale. True by default.
    invert : Invert black and white. True by default.

    Examples
    --------
    >>> video = ImageSequence('directory_name')
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[1][0:10][0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video
    """

    def __init__(self, directory, process_func=None, dtype=None):
        if not os.path.isdir(directory):
            raise ValueError("%s is not a directory." % directory)
        filenames = os.listdir(directory)
        filenames.sort()  # listdir returns arbitrary order
        make_full_path = lambda filename: os.path.join(directory, filename)
        self._filepaths = map(make_full_path, filenames)
        self._count = len(self._filepaths)

        if process_func is None:
            process_func = lambda x: x
        if not callable(process_func):
            raise ValueError("process_func must be a function, or None")
        self.process_func = process_func

        tmp = imread(self._filepaths[0])
        self._first_frame_shape = tmp.shape

        if dtype is None:
            self._dtype = tmp.dtype
        else:
            self._dtype = dtype

    def get_frame(self, j):
        if j > self._count:
            raise ValueError("File does not contain this many frames")
        res = imread(self._filepaths[j])
        if res.dtype != self._dtype:
            res = res.astype(self._dtype)
        res = self.process_func(res)
        return res

    def __len__(self):
        return self._count

    @property
    def frame_shape(self):
        return self._first_frame_shape

    @property
    def pixel_type(self):
        return self._dtype
