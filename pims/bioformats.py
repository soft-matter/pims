from __future__ import (absolute_import, division, print_function)

import numpy as np

from pims.base_frames import FramesSequence
from pims.frame import Frame
from warnings import warn
from os.path import isfile

try:
    import javabridge
except ImportError:
    javabridge = None

try:
    import bioformats
except ImportError:
    bioformats = None

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None


def available():
    try:
        import javabridge
        import bioformats
    except ImportError:
        return False
    else:
        return True


class MetadataRetrieve(object):
    """This class is an interface to loci.formats.meta.MetadataRetrieve. At
    initialization, it tests all the MetadataRetrieve functions and it only
    binds the ones that do not raise a java exception.

    Parameters
    ----------
    jmd: _javabridge.JB_Object
        java MetadataStore, retrieved with reader.rdr.getMetadataStore()
    log: _javabridge.JB_Object, optional
        java OutputStream to which java system.err and system.out are printing.

    Methods
    ----------
    <loci.formats.meta.MetadataRetrieve.function>(*args) : float or int or str
        see http://downloads.openmicroscopy.org/bio-formats/5.0.6/api/loci/
                                             formats/meta/MetadataRetrieve.html
    """
    def __init__(self, jmd, log=None):
        jmd = javabridge.JWrapper(jmd)

        def wrap_md(fn, name=None, paramcount=None, *args):
            if len(args) != paramcount:
                # raise sensible error for wrong number of arguments
                raise TypeError(('{0}() takes exactly {1} arguments ({2} ' +
                                 'given)').format(name, paramcount, len(args)))
            try:
                jw = fn(*args)
            except javabridge.JavaException as e:
                if log is not None:
                    print(javabridge.to_string(log))
                    javabridge.call(log, 'reset', '()V')
                raise e
            if jw is None or jw == '':
                return None
            # convert value to int, float, or string
            jw = str(jw)
            try:  # deal with values hidden inside 'value[number]'
                temp = jw[jw.index('value[') + 6:]
                temp = temp[:temp.index(']')]
            except ValueError:
                pass  # do nothing when 'value[' or ']' were not found
            else:
                jw = temp
            try:
                return int(jw)
            except ValueError:
                try:
                    return float(jw)
                except ValueError:
                    return jw

        env = javabridge.get_env()
        for name, method in jmd.methods.iteritems():
            if name[:3] == 'get':
                if name in ['getRoot', 'getClass']:
                    continue
                params = env.get_object_array_elements(method[0].getParameterTypes())
                try:
                    fn = getattr(jmd, name)
                    field = fn(*((0,) * len(params)))
                    if field is not None:
                        # If there is no exception, wrap the function and bind.
                        def fnw(fn1=fn, naame=name, paramcount=len(params)):
                            return (lambda *args: wrap_md(fn1, naame,
                                                          paramcount, *args))
                        fnw = fnw()
                        fnw.__doc__ = fn.__doc__
                        setattr(self, name, fnw)
                except javabridge.JavaException:
                    # function is not supported by this specific reader
                    pass

        if log is not None:
            javabridge.call(log, 'reset', '()V')

    def __repr__(self):
        listing = list(filter(lambda x: x[:2] != '__', dir(self)))
        return '<MetadataRetrieve> Available loci.formats.meta.' + \
               'MetadataRetrieve functions: ' + ', '.join(listing)


class BioformatsReaderRaw(FramesSequence):
    """Reads 2D images from the frames of a file supported by bioformats into
    an iterable object that returns images as numpy arrays.

    Parameters
    ----------
    filename : str
    process_func : function, optional
        callable with signature `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    dtype : numpy datatype, optional
        Image arrays will be converted to this datatype.
    as_grey : boolean, optional
        Convert color images to greyscale. False by default.
        May not be used in conjunction with process_func.
    meta : bool, optional
        When true, the metadata object is generated. Takes time to build.
    java_memory : str, optional
        The max heap size of the java virtual machine, default 512m. As soon as
        the virtual machine is started, python has to be restarted to change
        the max heap size.
    series: int, optional
        Active image series index, defaults to 0. Changeable via the `series`
        property.


    Attributes
    ----------
    __len__ : int
        Number of planes in active series (= size Z*C*T)
    metadata : MetadataRetrieve object
        This object contains loci.formats.meta.MetadataRetrieve functions for
        metadata reading. Not available when meta == False.
    frame_metadata : dict
        This dictionary sets which metadata fields are read and passed into the
        Frame.metadata field obtained by get_frame. This will only work if
        meta=True. Only MetadataRetrieve methods with signature (series, plane)
        will be accepted.
    sizes : dict of int
        Number of series and for active series: X, Y, Z, C, T sizes
    frame_shape : tuple of int
        Shape of the image (y, x) or (y, x, 3) or (y, x, 4)
    series : int
        active series that is read by get_frame. Writeable.
    channel : int or list of int
        channel(s) that are read by get_frame. Writeable.
    crop : tuple of int (y offset, x offset, height, width) or None
        when set, reader only reads part of the image
    pixel_type : numpy.dtype
        numpy datatype of pixels
    reader_class_name : string
        classname of bioformats imagereader (loci.formats.in.*)
    java_log : string
        contains everything printed to java system.out and system.err
    isRGB : boolean
        True if the image is an RGB image

    Methods
    ----------
    get_frame(plane) : pims.frame object
        returns 2D image in active series. See notes for metadata content.
    get_index(z, c, t) : int
        returns the imageindex in the current series with given coordinates
    get_metadata_raw(form) : dict or list or string
        returns the raw metadata from the file. Form defaults to 'dict', other
        options are 'list' and 'string'.
    get_metadata_xml() : string
        returns the metadata in xml format
    get_metadata_omexml() : bioformats.OMEXML object
        parses the xml metadata to an omexml object
    close(is_last) :
        closes the reader. When is_last is true, java VM is stopped. Be sure
        to do that only at the last image, because the VM cannot be restarted
        unless you restart python console. The same as pims.kill_vm()

    Examples
    ----------
    >>> frames.metadata.getPlaneDeltaT(0, 50)
    ...    # evaluates loci.formats.meta.MetadataRetrieve.getPlaneDeltaT(0, 50)

    Notes
    ----------
    Be sure to kill the java VM with pims.kill_vm() the end of the script. It
    cannot be restarted from the same python console, however. You can also
    kill the vm by calling frame.close(is_last=True).

    Dependencies:
    https://github.com/CellProfiler/python-bioformats
    https://github.com/CellProfiler/python-javabridge
    or (windows compiled) http://www.lfd.uci.edu/~gohlke/pythonlibs/#javabridge

    Tested with files from http://loci.wisc.edu/software/sample-data
    Working for:
        Zeiss Laser Scanning Microscopy, IPLab, Gatan Digital Micrograph,
        Image-Pro sequence, Leica, Image-Pro workspace, Nikon NIS-Elements ND2,
        Image Cytometry Standard, QuickTime movie
    Not (fully) working for:
        Olympus Fluoview TIFF, Bio-Rad PIC, Openlab LIFF, PerkinElmer,
        Andor Bio-imaging Division TIFF, Leica LIF, BIo-Rad PIC

    For files larger than 4GB, 64 bits Python is required

    Metadata automatically provided by get_frame, as dictionary:
        plane: index of image in series
        series: series index
        indexC, indexZ, indexT: indexes of C, Z, T
        X, Y, Z: physical location of the image in microns
        T: timestamp of the image in seconds
    """

    @classmethod
    def class_exts(cls):
        try:
            return set(bioformats.READABLE_FORMATS)
        except AttributeError:
            return {}

    class_priority = 2

    def __init__(self, filename, process_func=None, dtype=None,
                 as_grey=False, meta=True, java_memory='512m', series=0):
        global _java_log

        # Make sure that file exists before starting java
        if not isfile(filename):
            raise IOError('The file "{}" does not exist.'.format(filename))

        # Start java VM and initialize logger (globally)
        if not javabridge._javabridge.get_vm().is_active():
            javabridge.start_vm(class_path=bioformats.JARS, run_headless=True,
                                max_heap_size=java_memory)
            _java_log = javabridge.run_script("""
                org.apache.log4j.BasicConfigurator.configure();
                log4j_logger = org.apache.log4j.Logger.getRootLogger();
                log4j_logger.setLevel(org.apache.log4j.Level.WARN);
                java_out = new java.io.ByteArrayOutputStream();
                out_printstream = new java.io.PrintStream(java_out);
                java.lang.System.setOut(out_printstream);
                java.lang.System.setErr(out_printstream);
                java_out;""")
        # If there are no JVMs attached to this thread, attach one
        if javabridge._javabridge.get_thread_local("attach_count", 0) == 0:
            javabridge.attach()

        # Initialize reader and metadata
        self.filename = str(filename)
        if meta:
            self.reader = bioformats.ImageReader(path=self.filename)
            self.metadata = MetadataRetrieve(self.reader.rdr.getMetadataStore(),
                                             log=_java_log)
        else:  # skip built-in metadata initialization
            self.reader = bioformats.ImageReader(path=self.filename,
                                                 perform_init=False)
            self.reader.rdr.setId(self.filename)
        self.rdr = self.reader.rdr  # define shorthand

        # Set the correct series and initialize the sizes
        self._size_series = self.rdr.getSeriesCount()
        if series >= self._size_series or series < 0:
            self.close()
            raise IndexError('Series index out of bounds.')
        self._series = series
        self._forced_dtype = dtype
        self._crop = None        
        self._change_series()

        # Define a process func, if applicable
        # TODO: check if as grey works with series with different dimensions
        self._validate_process_func(process_func)
        self._as_grey(as_grey, process_func)

        # Define the names of the standard per frame metadata.
        self.frame_metadata = {}
        if meta:
            if hasattr(self.metadata, 'getPlaneTheT'):
                self.frame_metadata['indexT'] = 'getPlaneTheT'
            if hasattr(self.metadata, 'getPlaneTheZ'):
                self.frame_metadata['indexZ'] = 'getPlaneTheZ'
            if hasattr(self.metadata, 'getPlaneTheC'):
                self.frame_metadata['indexC'] = 'getPlaneTheC'
            if hasattr(self.metadata, 'getPlaneDeltaT'):
                self.frame_metadata['T'] = 'getPlaneDeltaT'
            if hasattr(self.metadata, 'getPlanePositionX'):
                self.frame_metadata['X'] = 'getPlanePositionX'
            if hasattr(self.metadata, 'getPlanePositionY'):
                self.frame_metadata['Y'] = 'getPlanePositionY'
            if hasattr(self.metadata, 'getPlanePositionZ'):
                self.frame_metadata['Z'] = 'getPlanePositionZ'

    def _change_series(self):
        """Changes series and rereads dtype, sizes and pixelsizes.
        When pixelsize Y is not found, pixels are assumed to be square.
        """
        series = self._series
        self.rdr.setSeries(series)
        self.isRGB = self.rdr.isRGB()
        self._isInterleaved = self.rdr.isInterleaved()
        self._sizeT = self.rdr.getSizeT()
        self._sizeZ = self.rdr.getSizeZ()
        self._sizeY = self.rdr.getSizeY()
        self._sizeX = self.rdr.getSizeX()
        self._planes = self.rdr.getImageCount()

        # determine pixel type using bioformats
        pixel_type = self.rdr.getPixelType()
        little_endian = self.rdr.isLittleEndian()
        FormatTools = bioformats.formatreader.make_format_tools_class()
        if pixel_type == FormatTools.INT8:
            self._source_dtype = np.int8
        elif pixel_type == FormatTools.UINT8:
            self._source_dtype = np.uint8
        elif pixel_type == FormatTools.UINT16:
            self._source_dtype = '<u2' if little_endian else '>u2'
        elif pixel_type == FormatTools.INT16:
            self._source_dtype = '<i2' if little_endian else '>i2'
        elif pixel_type == FormatTools.UINT32:
            self._source_dtype = '<u4' if little_endian else '>u4'
        elif pixel_type == FormatTools.INT32:
            self._source_dtype = '<i4' if little_endian else '>i4'
        elif pixel_type == FormatTools.FLOAT:
            self._source_dtype = '<f4' if little_endian else '>f4'
        elif pixel_type == FormatTools.DOUBLE:
            self._source_dtype = '<f8' if little_endian else '>f8'

        if self._forced_dtype is None:
            self._pixel_type = self._source_dtype
        else:
            self._pixel_type = self._forced_dtype

        # Set image shape
        if self.isRGB:
            image = np.frombuffer(self.rdr.openBytes(0), self._source_dtype)
            self._sizeRGB = int(len(image) / (self._sizeX * self._sizeY))
            self._first_frame_shape = (self._sizeY, self._sizeX, self._sizeRGB)
            self._sizeC = 1
        else:
            self._first_frame_shape = (self._sizeY, self._sizeX)
            self._sizeC = self.rdr.getSizeC()
            
        # Reassert crop area
        crop_area = self.crop
        try:
            self.crop = crop_area
        except ValueError:
            self._crop = None

    def __len__(self):
        return self._planes

    def close(self, is_last=False):
        self.reader.close()
        if is_last:
            try:
                javabridge.kill_vm()
            except AttributeError:  # java_vm was already killed
                pass

    def __del__(self):
        self.close()

    @property
    def sizes(self):
        return {'series': self._size_series, 'X': self._sizeX,
                'Y': self._sizeY, 'Z': self._sizeZ, 'C': self._sizeC,
                'T': self._sizeT}

    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, value):
        if value >= self._size_series or value < 0:
            raise IndexError('Series index out of bounds.')
        else:
            if value != self._series:
                self._series = value
                self._change_series()

    @property
    def frame_shape(self):
        if self.crop is None:
            return (self._sizeY, self._sizeX)
        else:
            return (self._crop[2], self._crop[3])

    @property
    def crop(self):
        return self._crop  # YXHW order

    @crop.setter
    def crop(self, value):
        if value is None:
            self._crop = None
        else:
            Y, X, H, W = value
            if X < 0 or Y < 0 or X + W > self._sizeX or W < 0 or \
               Y + H > self._sizeY or H < 0:
                raise ValueError('Crop range out of bounds')
            self._crop = (Y, X, H, W)

    def get_frame(self, j):
        """Returns image in current series specific as a Frame object with
        specified frame_no and metadata attributes.
        """
        im, metadata = self._get_frame_2D(self.series, j)
        return Frame(self.process_func(im), frame_no=j, metadata=metadata)

    def _get_frame(self, series, j):
        """Actual reader, returns image as 2D numpy array and metadata as
        dict. It changes the series property if necessary.
        """
        self.series = series  # use property setter & error reporting

        if self._crop is None:
            im = np.frombuffer(self.rdr.openBytes(j), self._source_dtype)
        else:
            Y, X, H, W = self._crop
            im = np.frombuffer(self.rdr.openBytesXYWH(j, X, Y, W, H),
                               self._source_dtype)
        if self.isRGB:
            if self._isInterleaved:
                im.shape = self.frame_shape + (self._sizeRGB,)
            else:
                im.shape = (self._sizeRGB,) + self.frame_shape
                im = im.transpose(1, 2, 0)  # put RGB in inner dimension
        else:
            im.shape = self.frame_shape

        if im.dtype != self._pixel_type:
            im = im.astype(self._pixel_type)

        metadata = {'frame': j, 'series': series}
        for key, method in self.frame_metadata.iteritems():
            metadata[key] = getattr(self.metadata, method)(series, j)

        return im, metadata

    def get_metadata_xml(self):
        # bioformats.get_omexml_metadata opens and closes a new reader
        return bioformats.get_omexml_metadata(self.filename)

    def get_metadata_omexml(self):
        return bioformats.OMEXML(self.get_metadata_xml())

    def get_metadata_raw(self, form='dict'):
        # code based on javabridge.jutil.to_string,
        # .jdictionary_to_string_dictionary and .jenumeration_to_string_list
        # addition is that it deals with UnicodeErrors
        def to_string(jobject):
            if not isinstance(jobject, javabridge.jutil._javabridge.JB_Object):
                try:
                    return str(jobject)
                except UnicodeError:
                    return jobject
            return javabridge.jutil.call(jobject, 'toString',
                                         '()Ljava/lang/String;')
        hashtable = self.rdr.getMetadata()
        jhashtable = javabridge.jutil.get_dictionary_wrapper(hashtable)
        jenumeration = javabridge.jutil.get_enumeration_wrapper(jhashtable.keys())
        keys = []
        while jenumeration.hasMoreElements():
            keys.append(jenumeration.nextElement())
        if form == 'dict':
            result = {}
            for key in keys:
                result[key] = to_string(jhashtable.get(key))
        elif form == 'list':
            result = []
            for key in keys:
                result.append(key + ': ' + to_string(jhashtable.get(key)))
        elif form == 'string':
            result = ''
            for key in keys:
                result += key + ': ' + to_string(jhashtable.get(key)) + '\n'
        return result

    def get_index(self, z, c, t):
        return self.rdr.getIndex(z, c, t)

    def java_log(self):
        return javabridge.to_string(_java_log)

    @property
    def reader_class_name(self):
        return self.rdr.get_class_name()

    @property
    def pixel_type(self):
        return self._pixel_type

    def __repr__(self):
        result = """<Frames>
Source: {filename}
Series: {mp}, active: {mpa}
Framecount: {count} frames
Colordepth: {c}
Zstack depth: {z}
Time frames: {t}
Frame Shape: {w} x {h}""".format(w=self._sizeX,
                                 h=self._sizeY,
                                 mp=self._size_series,
                                 mpa=self._series,
                                 count=self._planes,
                                 z=self._sizeZ,
                                 t=self._sizeT,
                                 c=self._sizeC,
                                 filename=self.filename)
        return result


class BioformatsReader(BioformatsReaderRaw):
    """Reads multidimensional images from the frames of a file supported by
    bioformats into an iterable object that returns images as numpy arrays
    indexed by t. The numpy array dimensions are CZYX, ZYX, CYX or YX,
    depending on the contents of the file and the setting of the channel
    property.

    Parameters
    ----------
    filename: str
    process_func : function, optional
        callable with signature `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    dtype : numpy datatype, optional
        Image arrays will be converted to this datatype.
    as_grey : boolean, optional
        Convert color images to greyscale. False by default.
        May not be used in conjunction with process_func.
    meta: bool, optional
        When true, the metadata object is generated. Takes time to build.
    java_memory : str, optional
        The max heap size of the java virtual machine, default 512m. As soon as
        the virtual machine is started, python has to be restarted to change
        the max heap size.
    series: int, optional
        Active image series index, defaults to 0. Changeable via the `series`
        property.
    C : int or list of int
        Channel(s) that are read by get_frame. Changeable via the `channel`
        property. Defaults to all channels.

    Attributes
    ----------
    __len__ : int
        Number of timepoints in active series (equal to sizes['T'])
    metadata : MetadataRetrieve object
        This object contains loci.formats.meta.MetadataRetrieve functions for
        metadata reading. Not available when meta == False.
    frame_metadata : dict
        This dictionary sets which metadata fields are read and passed into the
        Frame.metadata field obtained by get_frame. This will only work if
        meta=True. Only MetadataRetrieve methods with signature (series, plane)
        will be accepted.
    sizes : dict of int
        Number of series and for active series: X, Y, Z, C, T sizes
    frame_shape : tuple of int
        Shape of the image (y, x) or (y, x, 3) or (y, x, 4)
    channel : int or iterable of int
        channel(s) that are read by get_frame. Writeable.
    series : int
        active series that is read by get_frame. Writeable.
    channel : int or list of int
        channel(s) that are read by get_frame. Writeable.
    crop : tuple of int (y offset, x offset, height, width) or None
        when set, reader only reads part of the image
    pixel_type : numpy.dtype
        numpy datatype of pixels
    reader_class_name : string
        classname of bioformats imagereader (loci.formats.in.*)
    java_log : string
        contains everything printed to java system.out and system.err
    isRGB : boolean
        True if the image is an RGB image
    channel_RGB : list of rgb values (floats)
        The rgb values of all active channels set by the channels property. If
        not supported by the underlying reader, this returns an empty list

    Methods
    ----------
    get_frame(plane) : pims.frame object
        returns 3D image in active series. See notes for metadata content.
    get_index(z, c, t) : int
        returns the imageindex in the current series with given coordinates
    get_metadata_raw(form) : dict or list or string
        returns the raw metadata from the file. Form defaults to 'dict', other
        options are 'list' and 'string'.
    get_metadata_xml() : string
        returns the metadata in xml format
    get_metadata_omexml() : bioformats.OMEXML object
        parses the xml metadata to an omexml object
    close(is_last) :
        closes the reader. When is_last is true, java VM is stopped. Be sure
        to do that only at the last image, because the VM cannot be restarted
        unless you restart python console. The same as pims.kill_vm()

    Examples
    ----------
    >>> frames.metadata.getPlaneDeltaT(0, 50)
    ...    # evaluates loci.formats.meta.MetadataRetrieve.getPlaneDeltaT(0, 50)

    Notes
    ----------
    Be sure to kill the java VM with pims.kill_vm() the end of the script. It
    cannot be restarted from the same python console, however. You can also
    kill the vm by calling frame.close(is_last=True).

    Dependencies:
    https://github.com/CellProfiler/python-bioformats
    https://github.com/CellProfiler/python-javabridge
    or (windows compiled) http://www.lfd.uci.edu/~gohlke/pythonlibs/#javabridge

    Tested with files from http://loci.wisc.edu/software/sample-data
    Working for:
        Zeiss Laser Scanning Microscopy, IPLab, Gatan Digital Micrograph,
        Image-Pro sequence, Leica, Image-Pro workspace, Nikon NIS-Elements ND2,
        Image Cytometry Standard, QuickTime movie
    Not (fully) working for:
        Olympus Fluoview TIFF, Bio-Rad PIC, Openlab LIFF, PerkinElmer,
        Andor Bio-imaging Division TIFF, Leica LIF, BIo-Rad PIC

    For files larger than 4GB, 64 bits Python is required

    Metadata automatically provided by get_frame, as dictionary:
        plane: index of image in series
        series: series index
        indexC, indexZ, indexT: indexes of C, Z, T
        X, Y, Z: physical location of the image in microns
        T: timestamp of the image in seconds
    """
    class_priority = 5

    def __init__(self, filename, process_func=None, dtype=None, as_grey=False,
                 meta=True, java_memory='512m', series=0, C=None):
        self._channel = C
        super(BioformatsReader, self).__init__(filename, process_func, dtype,
                                               as_grey, meta, java_memory,
                                               series)

    def __len__(self):
        return self._sizeT

    @property
    def channel(self):
        if self.isRGB:
            raise AttributeError('Channel index not applicable to RGB files.')
        return self._channel

    @channel.setter
    def channel(self, value):
        if self.isRGB:
            raise AttributeError('Channel index not applicable to RGB files.')
        try:
            channel = tuple(value)
        except TypeError:
            channel = (value,)
        if np.any(np.greater_equal(channel, self._sizeC)) or \
           np.any(np.less(channel, 0)):
            raise IndexError('Channel index should be positive and less ' +
                             'than the number of channels ' +
                             '({})'.format(self._sizeC + 1))
        self._channel = channel
        rgbvalues = []
        try:
            for c in channel:
                rgba = self.metadata.getChannelColor(self.series, c)
                rgbvalues.append([(rgba >> 24 & 255) / 255,
                                  (rgba >> 16 & 255) / 255,
                                  (rgba >> 8 & 255) / 255])
        except:  # a lot could happen, use catch all here
            self.channel_RGB = []
        else:
            self.channel_RGB = rgbvalues

    def _change_series(self):
        super(BioformatsReader, self)._change_series()

        if self.isRGB:
            self._channel = (0,)
            self.channel_RGB = []
        elif self._channel is None:
            self.channel = tuple(range(self._sizeC))
        else:
            try:
                self.channel = self._channel
            except IndexError:
                warn("Channel index out of range. Resetting to all channels.",
                     UserWarning)
                self.channel = tuple(range(self._sizeC))

    def get_frame(self, t):
        """Returns image in current series at specified T index. The image is
        wrapped in a Frame object with specified frame_no and metadata."""
        shape = (len(self._channel), self._sizeZ) + self.frame_shape
        if self.isRGB:
            shape = shape + (self._sizeRGB,)
        imlist = np.zeros(shape, dtype=self.pixel_type)

        mdlist = []
        for (Nc, c) in enumerate(self._channel):
            for z in range(self._sizeZ):
                index = self.get_index(z, c, t)
                imlist[Nc, z], md = self._get_frame(self.series, index)
                mdlist.append(md)

        keys = mdlist[0].keys()
        metadata = {}
        for k in keys:
            metadata[k] = [row[k] for row in mdlist]
            if metadata[k][1:] == metadata[k][:-1]:
                metadata[k] = metadata[k][0]

        if len(self.channel_RGB) > 0:
            metadata['colors'] = self.channel_RGB

        return Frame(self.process_func(imlist.squeeze()), frame_no=t,
                     metadata=metadata)
