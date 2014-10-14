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

        
class BioformatsReader(FramesSequence):
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
    filename : string
    process_func : function, optional
        callable with signature `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    MP : active multipoint index (change with property multipoint)
    
    Metadata
    ----------
    Standard metadata: 
     - number of subimages (MP)
     - per subimage: - size C, T, Z, Y, X
                     - physical pixelsizes X, Y, Z
     - per frame:         - indices plane, MP, Z, C, T
        (as dictionary)   - physical locations X, Y, Z, T
                    
     - more can be read out using metadataretrieve, passing function & parameters 
      as string to mdstr, mdfloat, mdint for string, float and integer values.
     http://downloads.openmicroscopy.org/bio-formats/5.0.4/api/loci/formats/meta/MetadataRetrieve.html
     - a bioformats.OMEXML object can also be obtained using .omexml, see 
     https://github.com/CellProfiler/python-bioformats"""
     
    @classmethod
    def class_exts(cls):
        return {'nd2'} | super(BioformatsReader, cls).class_exts()

    def __init__(self, filename, MP=0, process_func=None):
        self.filename = str(filename)
        self._multipoint = MP        
        self._validate_process_func(process_func)
        self._initializereader()
        self._changemultipoint() 
    
    
    def _initializereader(self):
        javabridge.start_vm(class_path=bioformats.JARS,max_heap_size='512m')
        self._reader = bioformats.get_image_reader(self.filename, self.filename) 
        self._jmd = javabridge.JWrapper(self._reader.rdr.getMetadataStore())        
        self._sizeMP = jwtoint(self._jmd.getImageCount())
        self._metadatacolumns = ['plane', 'indexMP', 'indexC', 'indexZ', 
                                 'indexT','X', 'Y', 'Z', 'T']
        self._lastframe = (-1,-1)
        self._current = None
           

    def _changemultipoint(self):   
        MP = self._multipoint                
        self._reader.rdr.setSeries(MP)
        self._sizeC = jwtoint(self._jmd.getPixelsSizeC(MP))
        self._sizeT = jwtoint(self._jmd.getPixelsSizeT(MP))
        self._sizeZ = jwtoint(self._jmd.getPixelsSizeZ(MP))
        self._sizeY = jwtoint(self._jmd.getPixelsSizeY(MP))
        self._sizeX = jwtoint(self._jmd.getPixelsSizeX(MP))
        self._planes = jwtoint(self._jmd.getPlaneCount(MP))               
        self._pixelX = jwtofloat(self._jmd.getPixelsPhysicalSizeX(MP))
        self._pixelY = jwtofloat(self._jmd.getPixelsPhysicalSizeY(MP))
        self._pixelZ = jwtofloat(self._jmd.getPixelsPhysicalSizeZ(MP))
        if self._pixelY == None:
            self._pixelY = self._pixelX                       
            
            
    def __len__(self):
        return self._planes

            
    def close(self):  
        bioformats.release_image_reader(self.filename)
        

    @property
    def pixelsizes(self):
        {'X': self._pixelX, 'Y': self._pixelY, 'Z': self._pixelZ}        
        
        
    @property
    def sizes(self):
        return {'MP': self._sizeMP, 'X': self._sizeX, 'Y': self._sizeY, 
                'Z': self._sizeZ, 'C': self._sizeC, 'T': self._sizeT}
                        
    
    @property
    def multipoint(self):
        return self._multipoint
    @multipoint.setter
    def multipoint(self, value):
        if value >= self._sizeMP:
            raise IndexError('Multipoint index out of bounds.')
        else:
            if value != self._multipoint:
                self._multipoint = value
                self._changemultipoint()
        
    def frame_shape(self):
        return self._sizeX, self._sizeY
        
        
    def get_frame(self, j):
        im, metadata = self._get_frame_2D(self._multipoint, j)
        
        imageproc = self.process_func(im)
        metadataproc = dict(zip(self._metadatacolumns, metadata))
        self._current = Frame(imageproc, frame_no=j, metadata=metadataproc) 
        self._lastframe = (self._multipoint, j) 
                   
        return self._current
        
    def _get_frame_2D(self, MP, j):
        if (MP, j) == self._lastframe:
            return self._current
        
        im = self._reader.read(series=MP, index=j)    

        metadata = (j,
                MP,
                jwtoint(self._jmd.getPlaneTheC(MP, j)),
                jwtoint(self._jmd.getPlaneTheZ(MP, j)),
                jwtoint(self._jmd.getPlaneTheT(MP, j)),
                jwtofloat(self._jmd.getPlanePositionX(MP, j)),
                jwtofloat(self._jmd.getPlanePositionY(MP, j)),
                jwtofloat(self._jmd.getPlanePositionZ(MP, j)),
                jwtofloat(self._jmd.getPlaneDeltaT(MP, j)))

        return im, metadata
        
     
    def omexml(self):
        xml = bioformats.get_omexml_metadata(self.filename)
        return bioformats.OMEXML(xml)        
    
    def mdfloat(self, MetadataRetrieve):
        try:
            exec('result = self._jmd.' + MetadataRetrieve)
            return jwtofloat(result)
        except:
            return None
    
    def mdint(self, MetadataRetrieve):
        try:
            exec('result = self._jmd.' + MetadataRetrieve)
            return jwtoint(result)
        except:
            return None
            
    def mdstr(self, MetadataRetrieve):
        try:
            exec('result = self._jmd.' + MetadataRetrieve)
            return jwtostr(result)
        except:
            return None
     
    @property
    def pixel_type(self):
        raise NotImplemented()

    def __repr__(self):
        # May be overwritten by subclasses
        result = """<Frames>
            Source: {filename}
            Multipoint: {mp}, active: {mpa}
            Framecount: {count} frames
            Colordepth: {c}
            Zstack depth: {z}
            Time frames: {t}
            Frame Shape: {w} x {h}""".format(w=self._sizeX,
                                              h=self._sizeY,
                                              mp=self._sizeMP,
                                              mpa=self._multipoint,
                                              count=self._planes,
                                              z=self._sizeZ,
                                              t=self._sizeT,
                                              c=self._sizeC,
                                              filename=self.filename)
        return result
        
        
class BioformatsReader3D(BioformatsReader):
    """Extends BioformatsReader3D
    Reads 3D images from the frames of a file supported by bioformats into an
    iterable object that returns images as numpy arrays.
    
    Required:
    https://github.com/CellProfiler/python-bioformats
    https://github.com/CellProfiler/python-javabridge
     or (windows compiled) http://www.lfd.uci.edu/~gohlke/pythonlibs/#javabridge   
    
    Only tested with Nikon ND2 files
    For files larger than 4GB, 64 bits Python, Javabridge and JDK are required

    Parameters
    ----------
    filename : string
    process_func : function, optional
        callable with signature `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    C : list of color channels to read (change with property channel)
    MP : active multipoint index (change with property multipoint)
    mode : 'auto', '2D', '3D'
    
    Metadata
    ----------
    Standard metadata: 
     - number of subimages (MP)
     - per subimage: - size C, T, Z, Y, X
                     - physical pixelsizes X, Y, Z
     - per 2D / 3D frame: - indices plane, MP, Z, T
        (as DataFrame)    - physical locations X, Y, Z, T
                    
     - more can be read out using metadataretrieve, passing function & parameters 
      as string to mdstr, mdfloat, mdint for string, float and integer values.
     http://downloads.openmicroscopy.org/bio-formats/5.0.4/api/loci/formats/meta/MetadataRetrieve.html
     - a bioformats.OMEXML object can also be obtained using .omexml, see 
     https://github.com/CellProfiler/python-bioformats
    
    """
    @classmethod
    def class_exts(cls):
        return {'nd2'} | super(BioformatsReader3D, cls).class_exts()

    def __init__(self, filename, mode='auto', C=[0], MP=0, process_func=None):
        if not hasattr(C, '__iter__'):
            C = [C]    
        if not mode in ('2D', '3D', 'auto'):   
            raise NotImplementedError('Unsupported mode')
        self._mode = mode
        self._channel = C
        
        super(BioformatsReader3D, self).__init__(filename, MP, process_func)        
        
        if self._mode == 'auto':
            if self._sizeZ > 1:
                self._mode = '3D'
            else:
                self._mode = '2D' 
        
        
    def __len__(self):
        if self._mode == '2D':
            return self._planes
        elif self._mode == '3D':
            return self._sizeT    
                    
        
    @property
    def channel(self):
        return self._channel
    @channel.setter
    def channel(self, value):
        if not hasattr(value, '__iter__'):
            value = [value]
        if np.any(np.greater_equal(value, self._sizeC)) or np.any(np.smaller(value, 0)):
            raise IndexError('Channel index out of bounds.')
        if value != self._channel:
            self._channel = value
            self._lastframe = (-1,-1)
            self._current = None
            
            
    @property
    def mode(self):
        return self._mode
    @mode.setter
    def mode(self, value):
        if value in ('2D', '3D') and value != self._mode:
            self._mode = value
            self._changemultipoint()
    
        
    def get_frame(self, t):
        if self._mode == '2D':
            return super(BioformatsReader3D, self).get_frame(t)  
        else:
            if (self._multipoint, t) == self._lastframe:
                return self._current
                
            planelist = np.empty((len(self._channel), self._sizeZ, 2), dtype=np.int32)
            
            for (Nc, c) in enumerate(self._channel):            
                for z in range(self._sizeZ):
                    planelist[Nc, z] = [self._multipoint,
                                 self._reader.rdr.getIndex(z,c,t)]
            
            imlist = np.empty((planelist.shape[0], planelist.shape[1], 
                               self._sizeY, self._sizeX))
                               
                           
            metadata = np.empty((planelist.shape[1], len(self._metadatacolumns)))  
            
            for (Nc, zstack) in enumerate(planelist):   
                for (Nz, plane) in enumerate(zstack):
                    imlist[Nc, Nz], metadata[Nz] = self._get_frame_2D(*plane)
            
            if DataFrame != None:
                metadata = DataFrame(metadata, columns=self._metadatacolumns)
                if len(self._channel) > 1:
                    metadata = metadata.drop('indexC', 1)
                
            self._lastframe = (self._multipoint, t) 
            self._current = Frame(self.process_func(imlist.squeeze()), 
                                  frame_no=t, metadata=metadata)                            
            
            return self._current