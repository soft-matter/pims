import os
import warnings
import numpy as np

from .frame import Frame
from .base_frames import FramesSequence


class Spec(object):
    """SPE file specification data

    Tuples of (offset, datatype, count), where offset is the offset in the SPE
    file and datatype is the datatype as used in `numpy.fromfile`()

    `data_start` is the offset of actual image data.

    `dtypes` translates SPE datatypes (0...4) to numpy ones, e. g. dtypes[0]
    is dtype("<f") (which is np.float32).

    `controllers` maps the `type` metadata to a human readable name

    `readout_modes` maps the `readoutMode` metadata to something human readable
    although this may not be accurate since there is next to no documentation
    to be found.
    """
    metadata = {
        #essential information
        "datatype": (108, "<h"), #dtypes
        "xdim": (42, "<H"),
        "ydim": (656, "<H"),
        "NumFrames": (1446, "<i"),

        #ROI information
        "NumROIsInExperiment": (1488, "<h"),
        "NumROI": (1510, "<h"),
        "ROIs": (1512, np.dtype([("startx", "<H"),
                                 ("endx", "<H"),
                                 ("groupx", "<H"),
                                 ("starty", "<H"),
                                 ("endy", "<H"),
                                 ("groupy", "<H")]), 10),

        #chip-related sizes
        "xDimDet": (6, "<H"),
        "yDimDet": (18, "<H"),
        "VChipXdim": (14, "<h"),
        "VChipYdim": (16, "<h"),

        #other stuff
        "ControllerVersion": (0, "<h"),
        "LogicOutput": (2, "<h"),
        "AmpHiCapLowNoise": (4, "<H"), #enum?
        "mode": (8, "<h"), #enum?
        "exp_sec": (10, "<f"),
        "date": (20, "<10S"),
        "DetTemperature": (36, "<f"),
        "DetType": (40, "<h"),
        "stdiode": (44, "<h"),
        "DelayTime": (46, "<f"),
        "ShutterControl": (50, "<H"), #normal, disabled open, disabled closed
                                      #but which one is which?
        "AbsorbLive": (52, "<h"), #bool?
        "AbsorbMode": (54, "<H"),
        "CanDoVirtualChipFlag": (56, "<h"), #bool?
        "ThresholdMinLive": (58, "<h"), #bool?
        "ThresholdMinVal": (60, "<f"),
        "ThresholdMinLive": (64, "<h"), #bool?
        "ThresholdMinVal": (66, "<f"),
        "ExperimentTimeLocal": (172, "<7S"),
        "ExperimentTimeUTC": (179, "<7S"),
        "ADCoffset": (188, "<H"),
        "ADCrate": (190, "<H"),
        "ADCtype": (192, "<H"),
        "ADCresolution": (194, "<H"),
        "ADCbitAdjust": (196, "<H"),
        "gain": (198, "<H"),
        "comments": (200, "<80S", 5),
        "geometric": (600, "<H"), #flags
        "swversion": (688, "<16S"),
        "spare4": (742, "<436S"),
        "XPrePixels": (98, "<h"),
        "XPostPixels": (100, "<h"),
        "YPrePixels": (102, "<h"),
        "YPostPixels": (104, "<h"),
        "ReadoutTime": (672, "<f"),
        "type": (704, "<h"), #controllers
        "clkspd_us": (1428, "<f"),
        "readoutMode": (1480, "<H"), #readout_modes
        "WindowSize": (1482, "<H"),
        "file_header_ver": (1992, "<f")
    }

    data_start = 4100

    dtypes = [np.dtype("<f"), np.dtype("<i"), np.dtype("<h"),
              np.dtype("<H"), np.dtype("<I")]

    controllers = [
        "new120 (Type II)", "old120 (Type I)", "ST130", "ST121", "ST138",
        "DC131 (PentaMax)", "ST133 (MicroMax/Roper)", "ST135 (GPIB)", "VTCCD",
        "ST116 (GPIB)", "OMA3 (GPIB)", "OMA4"
    ]

    #This was gathered from random places on the internet and own experiments
    #with the camera. May not be accurate.
    readout_modes = ["full frame", "frame transfer", "kinetics"]

    #do not decode the following metadata keys into strings, but leave them
    #as byte arrays
    no_decode = ["spare4"]


class SpeStack(FramesSequence):
    """Read image data from SPE files

    Attributes
    ----------
    default_char_encoding : string
        Default character encoding used to decode metadata strings. This is a
        class attribute. By setting `SpeStack.default_char_encoding =
        "my_encoding"`, "my_encoding" will be used as a default in all SpeStack
        instances thereafter, unless a different one is explicitly passed to
        the constructor. Defaults to "latin1".
    metadata : dict
        Contains additional metadata.
    """
    default_char_encoding = "latin1"

    @classmethod
    def class_exts(cls):
        return {"spe"} | super(SpeStack, cls).class_exts()

    def __init__(self, filename, char_encoding=None, check_filesize=True):
        """Create an iterable object that returns image data as numpy arrays

        Arguments
        ---------
        filename : string
            Name of the SPE file
        char_encoding : str or None, optional
            Specifies what character encoding is used to decode metatdata
            strings. If None, use the `default_char_encoding` class attribute.
            Defaults to None.
        check_filesize : bool, optional
            The number of frames in an SPE file should be recorded in the
            file's header. Some software fails to do so correctly. If
            `check_filesize` is `True`, calculate the number of frames from
            the file size. A warning is emitted if this doesn't match the
            number of frames from the file header. Defaults to True.
        """
        self._filename = filename
        self._file = open(filename, "rb")
        self._char_encoding = (char_encoding if char_encoding is not None
                               else self.default_char_encoding)

        ### Read metadata ###
        self.metadata = {}
        #Decode each string from the numpy array read by np.fromfile
        decode = np.vectorize(lambda x: x.decode(self._char_encoding))

        for name, sp in Spec.metadata.items():
            self._file.seek(sp[0])
            cnt = (1 if len(sp) < 3 else sp[2])
            v = np.fromfile(self._file, dtype=sp[1], count=cnt)
            if v.dtype.kind == "S" and name not in Spec.no_decode:
                #silently ignore string decoding failures
                try:
                    v = decode(v)
                except:
                    pass
            if cnt == 1:
                #for convenience, if the array contains only one single entry,
                #return this entry itself.
                v = v.item()
            self.metadata[name] = v

        ### Some metadata is "special", deal with it
        #Determine data type
        self._dtype = Spec.dtypes[self.metadata.pop("datatype")]

        #movie dimensions
        self._width = self.metadata.pop("xdim")
        self._height = self.metadata.pop("ydim")
        self._len = self.metadata.pop("NumFrames")

        if check_filesize:
            # Some software writes incorrecet `NumFrames` metadata
            # Use the file size to determine the number of frames
            fsz = os.path.getsize(filename)
            l = fsz - Spec.data_start
            l //= self._width * self._height * self._dtype.itemsize
            if l != self._len:
                warnings.warn("Number of frames according to file header "
                              "does not match the size of file " +
                              filename + ".")
                self._len = min(l, self._len)

        #The number of ROIs is given in the SPE file. Only return as many
        #ROIs as given
        num_rois = self.metadata.pop("NumROI", None)
        num_rois = (1 if num_rois < 1 else num_rois)
        self.metadata["ROIs"] = self.metadata["ROIs"][:num_rois]

        #chip sizes
        self.metadata["ChipSize"] = (self.metadata.pop("xDimDet", None),
                                     self.metadata.pop("yDimDet", None))
        self.metadata["VirtChipSize"] = (self.metadata.pop("VChipXdim", None),
                                         self.metadata.pop("VChipYdim", None))

        #geometric operations
        g = []
        f = self.metadata.pop("geometric", 0)
        if f & 1:
            g.append("rotate")
        if f & 2:
            g.append("reverse")
        if f & 4:
            g.append("flip")
        self.metadata["geometric"] = g

        #Make some additional information more human-readable
        t = self.metadata["type"]
        if 1 <= t <= len(Spec.controllers):
            self.metadata["type"] = Spec.controllers[t - 1]
        else:
            self.metadata.pop("type", None)
        m = self.metadata["readoutMode"]
        if 1 <= m <= len(Spec.readout_modes):
            self.metadata["readoutMode"] = Spec.readout_modes[m - 1]
        else:
            self.metadata.pop("readoutMode", None)

    @property
    def frame_shape(self):
        return self._height, self._width

    def __len__(self):
        return self._len

    def get_frame(self, j):
        if j >= self._len:
            raise ValueError("Frame number {} out of range.".format(j))
        self._file.seek(Spec.data_start
                        + j*self._width*self._height*self.pixel_type.itemsize)
        data = np.fromfile(self._file, dtype=self.pixel_type,
                           count=self._width*self._height)
        return Frame(data.reshape(self._height, self._width),
                     frame_no=j, metadata=self.metadata)

    def close(self):
        """Clean up and close file"""
        super(SpeStack, self).close()
        self._file.close()

    @property
    def pixel_type(self):
        return self._dtype

    def __repr__(self):
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w} x {h}
Pixel Datatype: {dtype}""".format(w=self._width,
                                  h=self._height,
                                  count=self._len,
                                  filename=self._filename,
                                  dtype=self._dtype)
