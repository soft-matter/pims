from pims.base_frames import FramesSequenceND
from pims.frame import Frame

try:
    import imageio
except ImportError:
    imageio = None
try:
    import imageio_ffmpeg
    try:
        imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError:
        imageio_ffmpeg = None
except ImportError:
    imageio_ffmpeg = None


def available():
    return imageio is not None


def ffmpeg_available():
    return (imageio is not None) and (imageio_ffmpeg is not None)


class ImageIOReader(FramesSequenceND):
    class_priority = 6

    propagate_attrs = ['frame_shape', 'pixel_type', 'metadata',
                       'get_metadata_raw', 'reader_class_name']

    @classmethod
    def class_exts(cls):
        exts = {'tiff', 'bmp', 'cut', 'dds', 'exr', 'g3', 'hdr', 'iff', 'j2k',
                'jng', 'jp2', 'jpeg', 'jpg', 'koala', 'pbm', 'pbmraw', 'pcd',
                'pcx', 'pfm', 'pgm', 'pgmraw', 'pict', 'png', 'ppm', 'ppmraw',
                'psd', 'ras', 'raw', 'sgi', 'targa', 'fi_tiff', 'wbmp', 'webp',
                'xbm', 'xpm', 'ico', 'gif', 'dicom', 'npz', 'fits', 'itk',
                'gdal', 'dummy', 'gif', 'ffmpeg', 'avbin', 'swf', 'fits',
                'gdal', 'ts', 'tif'}
        return exts.union(cls.additional_class_exts())

    @classmethod
    def additional_class_exts(cls):
        """If imageio-ffmpeg is available, more filetypes are supported."""
        movie_exts = set()
        if imageio_ffmpeg is not None:
            movie_exts = movie_exts.union(
                {'mov', 'avi', 'mpg', 'mpeg', 'mp4', 'mkv', 'wmv'}
            )
        return movie_exts

    def __init__(self, filename, **kwargs):
        if imageio is None:
            raise ImportError('The ImageIOReader requires imageio and '
                              '(for imageio >= 2.5) imageio-ffmpeg to work.')

        super(self.__class__, self).__init__()

        self.reader = imageio.get_reader(filename, **kwargs)
        self.filename = filename
        self._len = self.reader.get_length()
        # fallback to count_frames, for newer imageio versions
        if self._len == float("inf"):
            self._len = self.reader.count_frames()
        try:
            int(self._len)
        except OverflowError:
            self.reader.close()
            raise NotImplementedError(
                "Do not know how to deal with infinite readers"
                )

        first_frame = self.get_frame_2D(t=0)
        self._shape = first_frame.shape
        self._dtype = first_frame.dtype

        self._setup_axes()
        self._register_get_frame(self.get_frame_2D, 'yx')

    def _setup_axes(self):
        """Setup the xyctz axes, iterate over t axis by default

        """
        if self._shape[1] > 0:
            self._init_axis('x', self._shape[1])
        if self._shape[0] > 0:
            self._init_axis('y', self._shape[0])
        if self._len > 0:
            self._init_axis('t', self._len)

        if len(self.sizes) == 0:
            raise EmptyFileError("No axes were found for this file.")

        # provide the default
        self.iter_axes = self._guess_default_iter_axis()


    def _guess_default_iter_axis(self):
        """
        Guesses the default axis to iterate over based on axis sizes.
        Returns:
            the axis to iterate over
        """
        priority = ['t', 'z', 'c', 'v']
        found_axes = []
        for axis in priority:
            try:
                current_size = self.sizes[axis]
            except KeyError:
                continue

            if current_size > 1:
                return axis

            found_axes.append(axis)

        return found_axes[0]

    def get_frame_2D(self, **coords):
        i = coords['t'] if 't' in coords else 0
        frame = self.reader.get_data(i)
        return Frame(frame, frame_no=i, metadata=frame.meta)

    def get_metadata(self):
        return self.reader.get_meta_data(None)

    def __len__(self):
        return self._len

    def __iter__(self):
        iterable = self.reader.iter_data()
        for i in range(len(self)):
            frame = next(iterable)
            yield Frame(frame, frame_no=i, metadata=frame.meta)

    @property
    def frame_rate(self):
        return self.get_metadata()['fps']

    @property
    def frame_shape(self):
        return self._shape

    @property
    def pixel_type(self):
        return self._dtype

    def close(self):
        self.reader.close()
        super().close()

    def __del__(self):
        if hasattr(self, 'reader'):
            self.reader.close()
