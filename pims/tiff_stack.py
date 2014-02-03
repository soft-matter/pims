import os
import itertools
import numpy as np

try:
    from PIL import Image  # should work with PIL or PILLOW
except ImportError:
    pass  # api.py will handle

try:
    from libtiff import TIFF
except ImportError:
    pass  # api.py will handle

from pims.base_frames import FramesSequence

_dtype_map = {4: np.uint8,
              8: np.uint8,
              16: np.uint16}


class TiffStack_libtiff(FramesSequence):
    """Iterable object that returns frames of video as numpy arrays.

    Parameters
    ----------
    filename : string
    gray : Convert color image to grayscale. True by default.
    invert : Invert black and white. True by default.

    Examples
    --------
    >>> video = TiffStack('filename')
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[1][0:10][0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = video.count # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video
    """
    def __init__(self, filename, dtype=None):
        self._filename = filename
        self._tiff = TIFF.open(filename)

        self._count = 1
        while not self._tiff.LastDirectory():
            self._count += 1
            self._tiff.ReadDirectory()

        # reset to 0
        self._tiff.SetDirectory(0)

        tmp = self._tiff.read_image()
        if dtype is None:
            self._dtype = tmp.dtype
        else:
            self._dtype = dtype

        self._im_sz = tmp.shape

        self._byte_swap = bool(self._tiff.IsByteSwapped())

    def get_frame(self, j):
        if j > self._count:
            raise ValueError("File does not contain this many frames")
        self._tiff.SetDirectory(j)
        res = self._tiff.read_image().byteswap(self._byte_swap)
        if res.dtype != self._dtype:
            res = res.astype(self._dtype)

        return res

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        return self._im_sz

    def __len__(self):
        return self._count


class TiffStack_pil(FramesSequence):
    '''
    Class for wrapping tiff stacks (that is single file with
    many frames) that depends on PIL/PILLOW

    Parameters
    ----------
    fname : str
        Fully qualified file name

    dtype : `None` or `numpy.dtype`
        If `None`, use the native type of the image,
        other wise coerce into the specified dtype.
    '''
    def __init__(self, fname, dtype=None):

        self.im = Image.open(fname)

        self.im.seek(0)
        # this will need some work to deal with color
        if dtype is None:
            res = self.im.tag[0x102][0]
            print res
            self._dtype = _dtype_map.get(res, np.int16)
        else:
            self._dtype = dtype

        try:
            samples_per_pixel = self.im.tag[0x115][0]
            if samples_per_pixel != 1:
                raise ValueError("support for color not implemented")
        except:
            pass
        # get image dimensions from the meta data the order is flipped
        # due to row major v col major ordering in tiffs and numpy
        self._im_sz = (self.im.tag[0x101][0],
                      self.im.tag[0x100][0])
        self.cur = self.im.tell()
        # walk through stack to get length, there has to
        # be a better way to do this
        for j in itertools.count():
            try:
                self.im.seek(j)
            except EOFError:
                break

        self._count = j
        self.im.seek(0)

    def get_frame(self, j):
        '''Extracts the jth frame from the image sequence.
        if the frame does not exist return None'''
        try:
            self.im.seek(j)
        except EOFError:
            return None
        self.cur = self.im.tell()
        return np.reshape(self.im.getdata(),
                          self._im_sz).astype(self._dtype).T[::-1]

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        return self._im_sz

    def __len__(self):
        return self._count


class MM_TiffStack(TiffStack_pil):
    """
    Specialized class for dealing with meta-morph tiffs.

    A function `get_meta` is added which extracts and parses
    the xml meta-data field.
    """
    def get_meta(self, j):
        cur = self.im.tell()
        if cur != j:
            self.im.seek(j)
            xml_str = im_wrap.im.tag[270]
            self.im.seek(cur)
        else:
            xml_str = im_wrap.im.tag[270]
        return _parse_mm_xml_string(xml_str)


class TiffSeries(FramesSequence):
    '''
    Class for dealing with a series of tiffs which are systematically
    named.

    Parameters
    ----------

    name_template : string
      input string for `format` to generate the file names keyed on `ind`.  For
      example for the set of files ('/home/user/data/f_001.tif',
       '/home/user/data/f_002.tif', ...) the correct input would be
        `name_template=/home/user/data/f_{ind:03d}.tif'

    offset : int, default 1
        The file number for frame 0.

    dtype : None or numpy.dtype
        If `None`, use the native type of the image,
        other wise coerce into the specified dtype.

    '''
    def __init__(self, name_template, offset=1, dtype=None):

        self._name_template = name_template
        self._offset = offset

        im = Image.open(self._name_template.format(ind=self._offset))
        # get image dimensions from the meta data the order is flipped
        # due to row major v col major ordering in tiffs and numpy
        self._im_sz = (im.tag[0x101][0],
                       im.tag[0x100][0])
        if dtype is None:
            res = im.tag[0x102][0]
            self._dtype = _dtype_map.get(res, np.int16)
        else:
            self._dtype = dtype

        try:
            samples_per_pixel = im.tag[0x115][0]
            if samples_per_pixel != 1:
                raise ValueError("support for color not implemented")
        except:
            pass

        # sort out how many there are
        while os.path.isfile(name_template.format(j + offset)):
            j += 1

        self._count = j

    def get_frame(self, j):
        '''Extracts the jth frame from the image sequence.
        if the frame does not exist return None'''

        im = Image.open(self._name_template.format(ind=j + self._offset))

        return np.reshape(im.getdata(),
                          self._im_sz).astype(self._dtype)

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        return self._im_sz

    def __len__(self):
        return self._count


# needed for the wrapper classes
def _parse_mm_xml_string(xml_str):
    """
    Parses Meta-Morph xml meta-data strings to a dictionary

    Parameters
    ----------
    xml_str : string
        A properly formed xml string

    Returns
    -------
    f : dict
        A dictionary object containing the meta-data
    """
    def _write(md_dict, name, val):
        if (name == "acquisition-time-local"
             or name == "modification-time-local"):
            tmp = int(val[18:])
            val = val[:18] + "%(#)03d" % {"#": tmp}
            val = datetime.datetime.strptime(val,
                                             '%Y%m%d %H:%M:%S.%f')
        md_dict[name] = val

    def _parse_attr(file_obj, dom_obj):
        if dom_obj.getAttribute("id") == "Description":
            _parse_des(file_obj, dom_obj)
        elif dom_obj.getAttribute("type") == "int":
            _write(file_obj, dom_obj.getAttribute("id"),
                   int(dom_obj.getAttribute("value")))
        elif dom_obj.getAttribute("type") == "float":
            _write(file_obj, dom_obj.getAttribute("id"),
                   float(dom_obj.getAttribute("value")))
        else:
            _write(file_obj, dom_obj.getAttribute("id"),
                   dom_obj.getAttribute("value").encode('ascii'))

    def _parse_des(file_obj, des_obj):
        des_string = des_obj.getAttribute("value")
        des_split = des_string.split("&#13;&#10;")

        for x in des_split:
            tmp_split = x.split(":")
            if len(tmp_split) == 2:
                _write(file_obj, tmp_split[0],
                       tmp_split[1].encode('ascii'))

    dom = xml.dom.minidom.parseString(xml_str)

    props = dom.getElementsByTagName("prop")
    f = dict()
    for p in props:
        _parse_attr(f, p)

    return f
