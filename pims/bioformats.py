import numpy as np
import packaging.version

from pims.base_frames import FramesSequence, FramesSequenceND
from pims.frame import Frame
from warnings import warn
import os

try:
    import jpype
except ImportError:
    jpype = None


def available():
    return jpype is not None


def _gen_jar_locations():
    """
    Generator that yields optional locations of loci_tools.jar.
    The precedence order is (highest priority first):

    1. pims package location
    2. PROGRAMDATA/pims/loci_tools.jar
    3. LOCALAPPDATA/pims/loci_tools.jar
    4. APPDATA/pims/loci_tools.jar
    5. /etc/loci_tools.jar
    6. ~/.config/pims/loci_tools.jar
    """
    yield os.path.dirname(__file__)
    if 'PROGRAMDATA' in os.environ:
        yield os.path.join(os.environ['PROGRAMDATA'], 'pims')
    if 'LOCALAPPDATA' in os.environ:
        yield os.path.join(os.environ['LOCALAPPDATA'], 'pims')
    if 'APPDATA' in os.environ:
        yield os.path.join(os.environ['APPDATA'], 'pims')
    yield '/etc'
    yield os.path.join(os.path.expanduser('~'), '.config', 'pims')


def _find_jar():
    """
    Finds the location of loci_tools.jar, if necessary download it to a
    writeable location.
    """
    for loc in _gen_jar_locations():
        if os.path.isfile(os.path.join(loc, 'loci_tools.jar')):
            return os.path.join(loc, 'loci_tools.jar')

    warn('loci_tools.jar not found, downloading')
    return download_jar()


def download_jar(version='6.7.0'):
    """ Downloads the bioformats distribution of given version. """
    from urllib.request import urlopen
    import hashlib

    for loc in _gen_jar_locations():
        # check if dir exists and has write access:
        if os.path.exists(loc) and os.access(loc, os.W_OK):
            break
        # if directory is pims and it does not exist, so make it (if allowed)
        if os.path.basename(loc) == 'pims' and \
           os.access(os.path.dirname(loc), os.W_OK):
            os.mkdir(loc)
            break
    else:
        raise IOError('No writeable location found. In order to use the '
                      'Bioformats reader, please download '
                      'loci_tools.jar to the pims program folder or one of '
                      'the locations provided by _gen_jar_locations().')

    url = ('http://downloads.openmicroscopy.org/bio-formats/' + version +
           '/artifacts/loci_tools.jar')

    path = os.path.join(loc, 'loci_tools.jar')
    loci_tools = urlopen(url).read()
    sha1_checksum = urlopen(url + '.sha1').read().split(b' ')[0].decode()

    downloaded = hashlib.sha1(loci_tools).hexdigest()
    if downloaded != sha1_checksum:
        raise IOError("Downloaded loci_tools.jar has invalid checksum. "
                      "Please try again.")

    with open(path, 'wb') as output:
        output.write(loci_tools)

    return path


def _jbytearr_stringbuffer(arr, dtype):
    # see https://github.com/originell/jpype/issues/71 and
    # https://github.com/originell/jpype/pull/73
    Jstr = jpype.java.lang.String(arr, 'ISO-8859-1').toString().encode('UTF-16LE')
    bytearr = np.array(np.frombuffer(Jstr, dtype='<u2'), dtype=np.byte)
    return np.frombuffer(bytearr, dtype=dtype)


def _jbytearr_javacasting(arr, dtype, bpp, fp, little_endian):
    # let java do the type conversion
    Jconv = loci.common.DataTools.makeDataArray(arr, bpp, fp, little_endian)
    return np.array(Jconv[:], dtype=dtype)


def _jrgba_to_rgb(rgba):
    return ((rgba >> 24 & 255) / 255.,
            (rgba >> 16 & 255) / 255.,
            (rgba >> 8 & 255) / 255.)


class MetadataRetrieve(object):
    """This class is an interface to loci.formats.meta.MetadataRetrieve. At
    initialization, it tests all the MetadataRetrieve functions and it only
    binds the ones that do not raise a java exception.

    Parameters
    ----------
    jmd: jpype._jclass.loci.formats.ome.OMEXMLMetadataImpl
        java MetadataStore, instanciated with:
            jmd = loci.formats.MetadataTools.createOMEXMLMetadata()
        and coupled to reader with `rdr.setMetadataStore(metadata)`

    Methods
    ----------
    <loci.formats.meta.MetadataRetrieve.function>(*args) : float or int or str
        see loci.formats.meta.MetadataRetrieve API on openmicroscopy.org
    """
    def __init__(self, md):
        def wrap_md(fn, name=None, paramcount=None, *args):
            if len(args) != paramcount:
                # raise sensible error for wrong number of arguments
                raise TypeError(('{0}() takes exactly {1} arguments ({2} ' +
                                 'given)').format(name, paramcount, len(args)))
            field = fn(*args)

            # deal with fields wrapped in a custom metadata type
            if hasattr(field, 'value'):
                field = field.value
            try:  # some fields have to be called
                field = field()
            except TypeError:
                pass

            # Convert this Java value to a Python type
            field = str(field)

            # convert to int or float if possible
            try:
                return int(field)
            except ValueError:
                try:
                    return float(field)
                except ValueError:
                    return field

        self.fields = []

        for name in dir(md):
            if (name[:3] != 'get') or (name in ['getRoot', 'getClass']):
                continue
            fn = getattr(md, name)
            for paramcount in range(5):
                try:
                    field = fn(*((0,) * paramcount))
                    if field is None:
                        continue
                    # If there is no exception, wrap the function and bind.
                    def fnw(fn1=fn, naame=name, paramcount=paramcount):
                        return (lambda *args: wrap_md(fn1, naame,
                                                      paramcount, *args))
                    fnw = fnw()
                    fnw.__doc__ = ('loci.formats.meta.MetadataRetrieve.' +
                                   name + ' wrapped\nby JPype and an '
                                   'additional automatic typeconversion.\n\n')
                    setattr(self, name[3:], fnw)
                    self.fields.append(name[3:])
                    continue
                except:
                    # function is not supported by this specific reader
                    pass

    def __repr__(self):
        return '<MetadataRetrieve> Available loci.formats.meta.' + \
               'MetadataRetrieve functions: ' + ', '.join(self.fields)


class BioformatsReader(FramesSequenceND):
    """Reads multidimensional images from the frames of a file supported by
    bioformats into an iterable object that returns images as numpy arrays.
    The axes inside the numpy array (czyx, zyx, cyx or yx) depend on the
    value of `bundle_axes`. It defaults to zyx or yx.

    The iteration axis depends on `iter_axes`. It defaults to t.

    Parameters
    ----------
    filename: str
    meta: bool, optional
        When true, the metadata object is generated. Takes time to build.
    java_memory : str, optional
        The max heap size of the java virtual machine, default 512m. As soon as
        the virtual machine is started, python has to be restarted to change
        the max heap size.
    read_mode : {'auto', 'jpype', 'stringbuffer', 'javacasting'}
        JPype can automatically convert java arrays to numpy arrays. On some
        installations, this will not work. In this case, using a Stringbuffer
        is the preferred option. However this doesn't work on Py3 and Unix
        systems. In that case, java can cast the type. Default 'auto'.
    series: int, optional
        Active image series index, defaults to 0. Changeable via the `series`
        property.

    Attributes
    ----------
    axes : list of strings
        List of all available axes
    ndim : int
        Number of image axes
    sizes : dict of int
        Dictionary with all axis sizes
    size_series : int
        Number of series inside file
    frame_shape : tuple of int
        Shape of frames that will be returned by get_frame
    iter_axes : iterable of strings
        This determines which axes will be iterated over by the FramesSequence.
        The last element in will iterate fastest. x and y are not allowed.
    bundle_axes : iterable of strings
        This determines which axes will be bundled into one Frame. The axes in
        the ndarray that is returned by get_frame have the same order as the
        order in this list. The last two elements have to be ['y', 'x'].
        Defaults to ['z', 'y', 'x'], when 'z' is available.
    default_coords: dict of int
        When a dimension is not present in both iter_axes and bundle_axes, the
        coordinate contained in this dictionary will be used.
    metadata : MetadataRetrieve object
        This object contains loci.formats.meta.MetadataRetrieve functions for
        metadata reading. Not available when meta == False.
    frame_metadata : dict
        This dictionary sets which metadata fields are read and passed into the
        Frame.metadata field obtained by get_frame. This will only work if
        meta=True. Only MetadataRetrieve methods with signature (series, plane)
        will be accepted.
    series : int
        active series that is read by get_frame. Writeable.
    pixel_type : numpy.dtype
        numpy datatype of pixels
    java_log : string
        contains everything printed to java system.out and system.err
    isRGB : boolean
        True if the image is an RGB image
    isInterleaved : boolean
        True if the image is interleaved
    read_mode : {'auto', 'jpype', 'stringbuffer', 'javacasting'}
        See parameter
    colors : list of rgb values (floats)
        The rgb values of all active channels set by the channels property. If
        not supported by the underlying reader, this returns None
    calibration : float
        The pixel size in microns per pixel, in x/y direction
    calibrationZ : float
        The pixel size in microns per pixel, in z direction
    reader_class_name : string
        The name of the used Bioformats reader

    Methods
    ----------
    get_metadata_raw(form) : dict or list or string
        returns the raw metadata from the file. Form defaults to 'dict', other
        options are 'list' and 'string'.
    close() :
        closes the reader

    Examples
    ----------
    >>> frames.metadata.PlaneDeltaT(0, 50)
    ...    # evaluates loci.formats.meta.MetadataRetrieve.getPlaneDeltaT(0, 50)

    Notes
    ----------
    It is not necessary to shutdown the JVM at end. It will be automatically
    done when JPype is unloaded at python exit.

    Dependencies:
    https://pypi.python.org/pypi/JPype1

    Tested with files from http://loci.wisc.edu/software/sample-data
    Working for:
        Zeiss Laser Scanning Microscopy, IPLab, Gatan Digital Micrograph,
        Image-Pro sequence, Leica, Image-Pro workspace, Nikon NIS-Elements ND2,
        Image Cytometry Standard, QuickTime movie, Olympus Fluoview TIFF,
        Andor Bio-imaging Division TIFF, PerkinElmer, Leica LIF

    Bio-Rad PIC and Openlab LIFF can only be loaded as single frames

    For files larger than 4GB, 64 bits Python is required

    Metadata automatically provided by get_frame, as dictionary:
        plane: index of image in series
        series: series index
        c, z, t: indexes of C, Z, T
        x_um, y_um, z_um: physical location of the image in microns
        t_s: timestamp of the image in seconds
    """
    @classmethod
    def class_exts(cls):
        return {'lsm', 'ipl', 'dm3', 'seq', 'nd2', 'ics', 'ids',
                'ipw', 'tif', 'tiff', 'jpg', 'bmp', 'lif', 'lei'}

    class_priority = 2
    propagate_attrs = ['frame_shape', 'pixel_type', 'metadata',
                       'get_metadata_raw', 'reader_class_name']

    @property
    def pixel_type(self):
        return self._pixel_type

    def __init__(self, filename, meta=True, java_memory='512m',
                 read_mode='auto', series=0):
        global loci
        super(BioformatsReader, self).__init__()

        if read_mode not in ['auto', 'jpype', 'stringbuffer', 'javacasting']:
            raise ValueError('Invalid read_mode value.')

        # Make sure that file exists before starting java
        if not os.path.isfile(filename):
            raise IOError('The file "{}" does not exist.'.format(filename))

        # Start java VM and initialize logger (globally)
        if not jpype.isJVMStarted():
            loci_path = _find_jar()
            # If we can turn off string auto-conversion, do so,
            # since this is the recommended practice.
            if (packaging.version.parse(jpype.__version__)
                    >= packaging.version.parse('0.7.0')):
                startJVM_kwargs = {'convertStrings': False}
            else:
                startJVM_kwargs = {}  # convertStrings kwarg not supported for earlier jpype versions
            jpype.startJVM(jpype.getDefaultJVMPath(), '-ea',
                           '-Djava.class.path=' + loci_path,
                           '-Xmx' + java_memory, **startJVM_kwargs)
            log4j = jpype.JPackage('org.apache.log4j')
            log4j.BasicConfigurator.configure()
            log4j_logger = log4j.Logger.getRootLogger()
            log4j_logger.setLevel(log4j.Level.ERROR)

        if hasattr(jpype.java.lang, 'Thread'):
            if not jpype.java.lang.Thread.isAttached():
                jpype.java.lang.Thread.attach()
        else:
            if not jpype.isThreadAttachedToJVM():
                jpype.attachThreadToJVM()

        loci = jpype.JPackage('loci')

        # Initialize reader and metadata
        self.filename = str(filename)
        self.rdr = loci.formats.ChannelSeparator(loci.formats.ChannelFiller())

        # patch for issue with ND2 files and the Chunkmap implemented in 5.4.0
        # See https://github.com/openmicroscopy/bioformats/issues/2955
        # circumventing the reserved keyword 'in'
        try:
            mo = getattr(loci.formats, 'in').DynamicMetadataOptions()
        except AttributeError:
            # Attribute name conflict causes mangling of `in` to `in_`
            mo = getattr(loci.formats, 'in_').DynamicMetadataOptions()
        mo.set('nativend2.chunkmap', 'False')  # Format Bool as String
        self.rdr.setMetadataOptions(mo)

        if meta:
            self._metadata = loci.formats.MetadataTools.createOMEXMLMetadata()
            self.rdr.setMetadataStore(self._metadata)
        self.rdr.setId(self.filename)
        if meta:
            self.metadata = MetadataRetrieve(self._metadata)

        # Checkout reader dtype and define read mode
        isLittleEndian = self.rdr.isLittleEndian()
        LE_prefix = ['>', '<'][isLittleEndian]
        FormatTools = loci.formats.FormatTools
        self._dtype_dict = {FormatTools.INT8: 'i1',
                            FormatTools.UINT8: 'u1',
                            FormatTools.INT16: LE_prefix + 'i2',
                            FormatTools.UINT16: LE_prefix + 'u2',
                            FormatTools.INT32: LE_prefix + 'i4',
                            FormatTools.UINT32: LE_prefix + 'u4',
                            FormatTools.FLOAT: LE_prefix + 'f4',
                            FormatTools.DOUBLE: LE_prefix + 'f8'}
        self._dtype_dict_java = {}
        for loci_format in self._dtype_dict.keys():
            self._dtype_dict_java[loci_format] = \
                (FormatTools.getBytesPerPixel(loci_format),
                 FormatTools.isFloatingPoint(loci_format),
                 isLittleEndian)

        # Set the correct series and initialize the sizes
        self.size_series = self.rdr.getSeriesCount()
        if series >= self.size_series or series < 0:
            self.rdr.close()
            raise IndexError('Series index out of bounds.')
        self._series = series
        self._change_series()

        # Set read mode. When auto, tryout fast and check the image size.
        if read_mode == 'auto':
            Jarr = self.rdr.openBytes(0)
            try:
                memoryview(Jarr)
            except TypeError:
                warn('Due to an issue with JPype 0.6.0, reading is slower. '
                     'Please consider upgrading JPype to 0.6.1 or later.')
                try:
                    im = self._jbytearr_stringbuffer(Jarr)
                    im.reshape(self._sizeRGB, self._sizeX, self._sizeY)
                except (AttributeError, ValueError):
                    read_mode = 'javacasting'
                else:
                    read_mode = 'stringbuffer'
            else:
                read_mode = 'jpype'
        self.read_mode = read_mode

        # Define the names of the standard per frame metadata.
        self.frame_metadata = {}
        if meta:
            if hasattr(self.metadata, 'PlaneDeltaT'):
                self.frame_metadata['t_s'] = 'PlaneDeltaT'
            if hasattr(self.metadata, 'PlanePositionX'):
                self.frame_metadata['x_um'] = 'PlanePositionX'
            if hasattr(self.metadata, 'PlanePositionY'):
                self.frame_metadata['y_um'] = 'PlanePositionY'
            if hasattr(self.metadata, 'PlanePositionZ'):
                self.frame_metadata['z_um'] = 'PlanePositionZ'

    def _change_series(self):
        """Changes series and rereads axes, sizes and metadata.
        """
        series = self._series
        self._clear_axes()
        self.rdr.setSeries(series)
        sizeX = self.rdr.getSizeX()
        sizeY = self.rdr.getSizeY()
        sizeT = self.rdr.getSizeT()
        sizeZ = self.rdr.getSizeZ()
        self.isRGB = self.rdr.isRGB()
        self.isInterleaved = self.rdr.isInterleaved()
        if self.isRGB:
            sizeC = self.rdr.getRGBChannelCount()
            if self.isInterleaved:
                self._frame_shape_2D = (sizeY, sizeX, sizeC)
                self._register_get_frame(self.get_frame_2D, 'yxc')
            else:
                self._frame_shape_2D = (sizeC, sizeY, sizeX)
                self._register_get_frame(self.get_frame_2D, 'cyx')
        else:
            sizeC = self.rdr.getSizeC()
            self._frame_shape_2D = (sizeY, sizeX)
            self._register_get_frame(self.get_frame_2D, 'yx')

        self._init_axis('x', sizeX)
        self._init_axis('y', sizeY)
        if sizeC > 1:
            self._init_axis('c', sizeC)
        if sizeT > 1:
            self._init_axis('t', sizeT)
        if sizeZ > 1:
            self._init_axis('z', sizeZ)

        # determine pixel type
        pixel_type = self.rdr.getPixelType()
        dtype = self._dtype_dict[pixel_type]
        java_dtype = self._dtype_dict_java[pixel_type]

        self._jbytearr_stringbuffer = \
            lambda arr: _jbytearr_stringbuffer(arr, dtype)
        self._jbytearr_javacasting = \
            lambda arr: _jbytearr_javacasting(arr, dtype, *java_dtype)
        self._pixel_type = dtype

        if 'z' in self.axes:
            self.bundle_axes = 'zyx'
        if 't' in self.axes:
            self.iter_axes = 't'

        # get some metadata fields
        try:
            self.colors = [_jrgba_to_rgb(self.metadata.ChannelColor(series, c))
                           for c in range(sizeC)]
        except AttributeError:
            self.colors = None
        try:
            self.calibration = self.metadata.PixelsPhysicalSizeX(series)
        except AttributeError:
            try:
                self.calibration = self.metadata.PixelsPhysicalSizeY(series)
            except:
                self.calibration = None
        try:
            self.calibrationZ = self.metadata.PixelsPhysicalSizeZ(series)
        except AttributeError:
            self.calibrationZ = None

    def close(self):
        self.rdr.close()

    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, value):
        if value >= self.size_series or value < 0:
            raise IndexError('Series index out of bounds.')
        else:
            if value != self._series:
                self._series = value
                self._change_series()

    def get_frame_2D(self, **coords):
        """Actual reader, returns image as 2D numpy array and metadata as
        dict.
        """
        _coords = {'t': 0, 'c': 0, 'z': 0}
        _coords.update(coords)
        if self.isRGB:
            _coords['c'] = 0
        j = self.rdr.getIndex(int(_coords['z']), int(_coords['c']),
                              int(_coords['t']))
        if self.read_mode == 'jpype':
            # The explicit memoryview cast avoids leaving a dangling buffer.
            im = np.frombuffer(memoryview(self.rdr.openBytes(j)),
                               dtype=self._pixel_type)
        elif self.read_mode == 'stringbuffer':
            im = self._jbytearr_stringbuffer(self.rdr.openBytes(j))
        elif self.read_mode == 'javacasting':
            im = self._jbytearr_javacasting(self.rdr.openBytes(j))

        im.shape = self._frame_shape_2D
        im = im.astype(self._pixel_type, copy=False)

        metadata = {'frame': j,
                    'series': self._series}
        if self.colors is not None:
            metadata['colors'] = self.colors
        if self.calibration is not None:
            metadata['mpp'] = self.calibration
        if self.calibrationZ is not None:
            metadata['mppZ'] = self.calibrationZ
        metadata.update(coords)
        for key, method in self.frame_metadata.items():
            metadata[key] = getattr(self.metadata, method)(self._series, j)

        return Frame(im, metadata=metadata)

    def get_metadata_raw(self, form='dict'):
        hashtable = self.rdr.getGlobalMetadata()
        keys = hashtable.keys()
        if form == 'dict':
            result = {}
            while keys.hasMoreElements():
                key = keys.nextElement()
                result[key] = str(hashtable.get(key))
        elif form == 'list':
            result = []
            while keys.hasMoreElements():
                key = keys.nextElement()
                result.append(key + ': ' + str(hashtable.get(key)))
        elif form == 'string':
            result = u''
            while keys.hasMoreElements():
                key = keys.nextElement()
                result += key + ': ' + str(hashtable.get(key)) + '\n'
        return result

    @property
    def reader_class_name(self):
        return self.rdr.getFormat()

    @property
    def version(self):
        return loci.formats.FormatTools.VERSION
