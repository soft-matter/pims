from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
from datetime import datetime
import itertools
import numpy as np
from pims.frame import Frame
from imageio.core import Format

try:
    from PIL import Image  # should work with PIL or PILLOW
except ImportError:
    Image = None

try:
    from libtiff import TIFF
except ImportError:
    TIFF = None

try:
    import tifffile
except ImportError:
    try:
        from skimage.external import tifffile
    except ImportError:
        tifffile = None


def libtiff_available():
    return TIFF is not None


def PIL_available():
    return Image is not None


def tifffile_available():
    return tifffile is not None


from pims.base_frames import FramesSequence

_dtype_map = {4: np.uint8,
              8: np.uint8,
              16: np.uint16}

def _tiff_datetime(dt_str):
    """Convert the DateTime string of TIFF files to a datetime object"""
    return datetime(year=int(dt_str[0:4]), month=int(dt_str[5:7]),
                    day=int(dt_str[8:10]), hour=int(dt_str[11:13]),
                    minute=int(dt_str[14:16]), second=int(dt_str[17:19]))


class FormatTiffStack_tifffile(Format):
    """Read TIFF stacks (single files containing many images) into an
    iterable object that returns images as numpy arrays.

    This reader, based on tiffile.py, should read standard TIFF
    files and sundry derivatives of the format used in microscopy.

    Parameters
    ----------
    filename : string

    Examples
    --------
    >>> video = TiffStack('many_images.tif')  # or .tiff
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[-1]) # Show the last frame.
    >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video

    Note
    ----
    This wraps tifffile.py. It should deal with a range of
    tiff files and sundry microscope related tiff derivatives.
    The obvious thing to do here is to extend tifffile.TiffFile;
    however that would over-ride our nice slicing sematics. The
    way that TiffFile deals with __getitem__ is to just pass it
    through to an underlying list.  Further, it return TiffPages,
    not arrays as we desire.

    See Also
    --------
    TiffStack_pil, TiffStack_libtiff, ImageSequence
    """
    def _can_read(self, request):
        """Determine whether `request.filename` can be read using this
        Format.Reader, judging from the imageio.core.Request object."""
        if request.mode[1] in (self.modes + '?'):
            if request.filename.lower().endswith(self.extensions):
                return True

    def _can_write(self, request):
        """Determine whether file type `request.filename` can be written using
        this Format.Writer, judging from the imageio.core.Request object."""
        return False

    class Reader(Format.Reader):
        def _open(self, **kwargs):
            # Specify kwargs here. Optionally, the user-specified kwargs
            # can also be accessed via the request.kwargs object.
            #
            # The request object provides two ways to get access to the
            # data. Use just one:
            #  - Use request.get_file() for a file object (preferred)
            #  - Use request.get_local_filename() for a file on the system
            handle = self.request.get_file()
            record = tifffile.TiffFile(handle, **kwargs).series[0]
            if hasattr(record, 'pages'):
                self._tiff = record.pages
            else:
                self._tiff = record['pages']

        def _close(self):
            # Close the reader.
            # Note that the request object will close self._fp
            pass

        def _get_length(self):
            # Return the number of images. Can be np.inf
            return len(self._tiff)

        def _get_data(self, index):
            tiff = self._tiff[index]
            return tiff.asarray(), self._read_metadata(tiff)

        def _get_meta_data(self, index):
            # Get the meta data for the given index. If index is None, it
            # should return the global meta data.
            tiff = self._tiff[index]
            return self._read_metadata(tiff)

        def _read_metadata(self, tiff):
            """Read metadata for current frame and return as dict"""
            md = {}
            try:
                md["ImageDescription"] = (
                    tiff.tags["image_description"].value.decode())
            except:
                pass
            try:
                dt = tiff.tags["datetime"].value.decode()
                md["DateTime"] = _tiff_datetime(dt)
            except:
                pass
            try:
                md["Software"] = tiff.tags["software"].value.decode()
            except:
                pass
            try:
                md["DocumentName"] = tiff.tags["document_name"].value.decode()
            except:
                pass
            return md


class TiffStack_libtiff(FramesSequence):
    """Read TIFF stacks (single files containing many images) into an
    iterable object that returns images as numpy arrays.

    This reader, based on libtiff, should read standard TIFF files.

    Parameters
    ----------
    filename : string

    Examples
    --------
    >>> video = TiffStack('many_images.tif')  # or .tiff
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[-1]) # Show the last frame.
    >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video

    See Also
    --------
    TiffStack_pil, TiffStack_tiffile, ImageSequence
    """
    def __init__(self, filename):
        self._filename = filename
        self._tiff = TIFF.open(filename)

        self._count = 1
        while not self._tiff.LastDirectory():
            self._count += 1
            self._tiff.ReadDirectory()

        # reset to 0
        self._tiff.SetDirectory(0)

        tmp = self._tiff.read_image()
        self._dtype = tmp.dtype

        self._im_sz = tmp.shape

        self._byte_swap = bool(self._tiff.IsByteSwapped())

    def get_frame(self, j):
        if j > self._count:
            raise ValueError("File does not contain this many frames")
        self._tiff.SetDirectory(j)
        res = self._tiff.read_image()
        if self._byte_swap:
            res = res.newbyteorder()
        return Frame(res, frame_no=j, metadata=self._read_metadata())

    def _read_metadata(self):
        """Read metadata for current frame and return as dict"""
        md = {}
        #explicit str() is needed for python2, otherwise the type would be
        #unicode and then it won't work anymore
        dt = self._tiff.GetField(str("ImageDescription"))
        if dt is not None:
            try:
                md["ImageDescription"] = dt.decode()
            except:
                pass
        dt = self._tiff.GetField(str("DateTime"))
        if dt is not None:
            try:
                md["DateTime"] = _tiff_datetime(dt.decode())
            except:
                pass
        dt = self._tiff.GetField(str("Software"))
        if dt is not None:
            try:
                md["Software"] = dt.decode()
            except:
                pass
        dt = self._tiff.GetField(str("DocumentName"))
        if dt is not None:
            try:
                md["DocumentName"] = dt.decode()
            except:
                pass
        return md

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        return self._im_sz

    def __len__(self):
        return self._count

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count=len(self),
                                  filename=self._filename,
                                  dtype=self.pixel_type)


class TiffStack_pil(FramesSequence):
    """Read TIFF stacks (single files containing many images) into an
    iterable object that returns images as numpy arrays.

    This reader, based on PIL/Pillow, should read standard TIFF stacks.

    Parameters
    ----------
    filename : string

    Examples
    --------
    >>> video = TiffStack('many_images.tif')  # or .tiff
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[-1]) # Show the last frame.
    >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video

    See Also
    --------
    TiffStack_libtiff, TiffStack_tiffile, ImageSequence
    """
    def __init__(self, fname):

        self.im = Image.open(fname)
        self._filename = fname  # used by __repr__

        self.im.seek(0)
        # this will need some work to deal with color
        res = self.im.tag[0x102][0]
        self._dtype = _dtype_map.get(res, np.int16)
        try:
            samples_per_pixel = self.im.tag[0x115][0]
            if samples_per_pixel != 1:
                raise ValueError("support for color not implemented")
        except:
            pass
        # get image dimensions from the meta data the order is flipped
        # due to row major v col major ordering in tiffs and numpy
        w = self.im.tag[0x101][0]
        h = self.im.tag[0x100][0]
        samples_per_px = self.im.tag[0x115][0]
        if samples_per_px != 1:
            self._im_sz = (w, h, samples_per_px)
        else:
            self._im_sz = (w, h)
        # walk through stack to get length, there has to
        # be a better way to do this
        for j in itertools.count():
            try:
                self.im.seek(j)
            except EOFError:
                break

        self._count = j
        self.cur = self.im.tell()

    def get_frame(self, j):
        '''Extracts the jth frame from the image sequence.
        if the frame does not exist return None'''
        # PIL does not support random access. If we need to rewind, re-open
        # the file.
        if j < self.cur:
            self.im = Image.open(self._filename)
            self.im.seek(j)
        elif j > self.cur:
            self.im.seek(j)
        elif j > len(self):
            raise IndexError("out of bounds; length is {0}".format(len(self)))
        # If j == self.cur, do nothing.
        self.cur = self.im.tell()
        res = np.reshape(self.im.getdata(), self._im_sz)
        return Frame(res, frame_no=j, metadata=self._read_metadata())

    def _read_metadata(self):
        """Read metadata for current frame and return as dict"""
        try:
            tags = self.im.tag_v2  # for Pillow >= v3.0.0
        except AttributeError:
            tags = self.im.tag  # for Pillow < v3.0.0
        md = {}
        try:
            md["ImageDescription"] = tags[270]
        except KeyError:
            pass
        try:
            md["DateTime"] = _tiff_datetime(tags[306])
        except KeyError:
            pass
        try:
            md["Software"] = tags[305]
        except KeyError:
            pass
        try:
            md["DocumentName"] = tags[269]
        except KeyError:
            pass
        return md

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        return self._im_sz

    def __len__(self):
        return self._count

    def close(self):
        self._tiff.close()

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count=len(self),
                                  filename=self._filename,
                                  dtype=self.pixel_type)


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

    '''
    def __init__(self, name_template, offset=1):

        self._name_template = name_template
        self._offset = offset

        im = Image.open(self._name_template.format(ind=self._offset))
        # get image dimensions from the meta data the order is flipped
        # due to row major v col major ordering in tiffs and numpy
        self._im_sz = (im.tag[0x101][0],
                       im.tag[0x100][0])

        res = im.tag[0x102][0]
        self._dtype = _dtype_map.get(res, np.int16)

        try:
            samples_per_pixel = im.tag[0x115][0]
            if samples_per_pixel != 1:
                raise ValueError("support for color not implemented")
        except:
            pass

        j = 0
        # sort out how many there are
        while os.path.isfile(name_template.format(j + offset)):
            j += 1

        self._count = j

    def get_frame(self, j):
        '''Extracts the jth frame from the image sequence.
        if the frame does not exist return None'''

        im = Image.open(self._name_template.format(ind=j + self._offset))

        return np.reshape(im.getdata(), self._im_sz)

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        return self._im_sz

    def __len__(self):
        return self._count

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {name_template}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self.frame_shape[0],
                                  h=self.frame_shape[1],
                                  count=len(self),
                                  name_template=self._name_template,
                                  dtype=self.pixel_type)


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
