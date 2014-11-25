from __future__ import (absolute_import, division, print_function)

import numpy as np

from pims.base_frames import FramesSequence
from pims.frame import Frame


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

def jwtofloat(jwrapper):
    if jwrapper != None:
        return float(str(jwrapper))
    else:
        return None
        
def jwtoint(jwrapper):
    if jwrapper != None:
        return int(str(jwrapper))
    else:
        return None
        
def jwtostr(jwrapper):
    if jwrapper != None:
        return str(jwrapper)
    else:
        return None
        
def jwtoauto(jwrapper):
    if jwrapper == None:
        return None
    jw = str(jwrapper)
    try:
        return int(jw)
    except ValueError:
        try:
            return float(jw)
        except ValueError:
            return jw    
            
class BioformatsReader2D(FramesSequence):
    """Reads 2D images from the frames of a file supported by bioformats into an
    iterable object that returns images as numpy arrays.
    
    Required:
    https://github.com/CellProfiler/python-bioformats
    https://github.com/CellProfiler/python-javabridge
     or (windows compiled) http://www.lfd.uci.edu/~gohlke/pythonlibs/#javabridge   
    
    Only tested with Nikon ND2 files
    For files larger than 4GB, 64 bits Python, Javabridge and JDK are required

    Parameters
    ----------
    filename: string
    process_func: function, optional
        callable with signature `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    series: active image series index (change with property series)
    
    Metadata
    ----------
    Standard metadata: 
     - number of series
     - per series:   - size C, T, Z, Y, X
                     - physical pixelsizes X, Y, Z
     - per frame:         - indices plane, series, Z, C, T
        (as dictionary)   - physical locations X, Y, Z, T
                    
     - more can be read out using metadataretrieve, passing the method as a 
     string. All further arguments will be passed to that method. Example: 
     
         frames.metadataretrieve('getPlaneDeltaT', 0, 50)
         evaluates MetadataRetrieve.getPlaneDeltaT(0, 50)
     
     documentation on MetadataRetrieve methods can be found here:
     http://downloads.openmicroscopy.org/bio-formats/5.0.4/api/loci/formats/meta/MetadataRetrieve.html
     - a bioformats.OMEXML object can also be obtained using .omexml, see 
     https://github.com/CellProfiler/python-bioformats"""
     
    @classmethod
    def class_exts(cls):
        return {'nd2'} | super(BioformatsReader2D, cls).class_exts()

    def __init__(self, filename, series=0, process_func=None, as_grey=False):
        self.filename = str(filename)
        self._series = series        
        self._validate_process_func(process_func)
        self._initializereader()
        self._change_series() 
    
    def _initializereader(self):
        javabridge.start_vm(class_path=bioformats.JARS,max_heap_size='512m')
        self._reader = bioformats.get_image_reader(self.filename, self.filename) 
        if self._reader.rdr.isRGB():
            raise NotImplementedError('RGB images are not supported')
        self._jmd = javabridge.JWrapper(self._reader.rdr.getMetadataStore())        
        self._size_series = self._reader.rdr.getSeriesCount()
        self._metadatacolumns = ['plane', 'series', 'indexC', 'indexZ', 
                                 'indexT','X', 'Y', 'Z', 'T']
           
    def _change_series(self):   
        series = self._series   
        self._reader.rdr.setSeries(series)
        
        # make use of built-in methods of bioformats to determine numpy dtype
        im, md = self._get_frame_2D(series, 0)     
        self._pixel_type = im.dtype
        
        self._sizeC = self._reader.rdr.getSizeC()
        self._sizeT = self._reader.rdr.getSizeT()
        self._sizeZ = self._reader.rdr.getSizeZ()
        self._sizeY = self._reader.rdr.getSizeY()
        self._sizeX = self._reader.rdr.getSizeX()
        self._planes = self._reader.rdr.getImageCount()       
        self._pixelX = jwtofloat(self._jmd.getPixelsPhysicalSizeX(series))
        self._pixelY = jwtofloat(self._jmd.getPixelsPhysicalSizeY(series))
        self._pixelZ = jwtofloat(self._jmd.getPixelsPhysicalSizeZ(series))
        if self._pixelY == None:
            self._pixelY = self._pixelX                       
            
    def __len__(self):
        return self._planes

    def close(self):  
        bioformats.release_image_reader(self.filename)
        

    @property
    def pixelsizes(self):
        return {'X': self._pixelX, 'Y': self._pixelY, 'Z': self._pixelZ}        
        
    @property
    def sizes(self):
        return {'series': self._size_series, 'X': self._sizeX, 'Y': self._sizeY, 
                'Z': self._sizeZ, 'C': self._sizeC, 'T': self._sizeT}
                        
    @property
    def series(self):
        return self._series
    @series.setter
    def series(self, value):
        if value >= self._size_series:
            raise IndexError('Series index out of bounds.')
        else:
            if value != self._series:
                self._series = value
                self._change_series()
        
    @property
    def frame_shape(self):
        return self._sizeX, self._sizeY
        
    def get_frame(self, j):
        im, metadata = self._get_frame(self.series, j)
        imageproc = self.process_func(im)
        return Frame(imageproc, frame_no=j, metadata=metadata) 
        
    def _get_frame(self, series, j):
        im, metadata = self._get_frame_2D(series, j)
        metadataproc = dict(zip(self._metadatacolumns, metadata))        
        return im, metadataproc
        
    def _get_frame_2D(self, series, j):       
        im = self._reader.read(series=series, index=j)    

        metadata = (j,
                series,
                jwtoint(self._jmd.getPlaneTheC(series, j)),
                jwtoint(self._jmd.getPlaneTheZ(series, j)),
                jwtoint(self._jmd.getPlaneTheT(series, j)),
                jwtofloat(self._jmd.getPlanePositionX(series, j)),
                jwtofloat(self._jmd.getPlanePositionY(series, j)),
                jwtofloat(self._jmd.getPlanePositionZ(series, j)),
                jwtofloat(self._jmd.getPlaneDeltaT(series, j)))

        return im, metadata
        
    def omexml(self):
        xml = bioformats.get_omexml_metadata(self.filename)
        return bioformats.OMEXML(xml)        
        
    def metadataretrieve(self, mdr, *args):
        try:
            jw = getattr(self._jmd, mdr)(*args)
        except AttributeError or TypeError:
            return None
        return jwtoauto(jw)        
     
    @property
    def pixel_type(self):
        return self._pixel_type

    def __repr__(self):
        # May be overwritten by subclasses
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
        
        
class BioformatsReader3D(BioformatsReader2D):
    """Extends BioformatsReader2D
    Reads 3D images from the frames of a file supported by bioformats into an
    iterable object that returns images as numpy arrays, indexed by T index.

    Parameters
    ----------
    filename : string
    process_func : function, optional
        callable with signature `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    C : list of color channels to read (change with property channel)
    series : active Series index (change with property series)
    
    Metadata
    ----------
    see BioformatsReader2D class docstring
    
    """
    @classmethod
    def class_exts(cls):
        return {'nd2'} | super(BioformatsReader3D, cls).class_exts()

    def __init__(self, filename, mode='auto', C=[0], series=0, 
                 process_func=None, as_grey=False):
        if not hasattr(C, '__iter__'):
            C = [C]    
        self._channel = C
        
        super(BioformatsReader3D, self).__init__(filename, series, process_func)                
        
    def __len__(self):
        return self._sizeT    
                    
    @property
    def channel(self):
        return self._channel
    @channel.setter
    def channel(self, value):
        if not hasattr(value, '__iter__'):
            value = [value]
        if np.any(np.greater_equal(value, self._sizeC)) or np.any(np.less(value, 0)):
            raise IndexError('Channel index out of bounds.')
        if value != self._channel:
            self._channel = value
                
    def _get_frame(self, series, t):
        imlist = np.empty((len(self.channel), self._sizeZ, 
                           self._sizeY, self._sizeX))
        metadata = np.empty((self._sizeZ, len(self._metadatacolumns)))  
                           
        for (Nc, c) in enumerate(self.channel):            
            for z in range(self._sizeZ):
                imlist[Nc, z], metadata[z] = \
                    self._get_frame_2D(series, self._reader.rdr.getIndex(z,c,t))    

        if DataFrame != None:
            metadata = DataFrame(metadata, columns=self._metadatacolumns)
            if len(self.channel) > 1:
                metadata = metadata.drop('indexC', 1)
        
        return imlist.squeeze(), metadata
