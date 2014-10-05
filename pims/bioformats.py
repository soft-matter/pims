from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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
    """Reads 3D images from the frames of a file supported by bioformats into an
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
        callable with signalture `proc_img = process_func(img)`,
        which will be applied to the data from each frame
    C : list of preferred colors channel to read (change with property channel)
    MP : active multipoint index (change with property channel)
    X,Y,Z,W,H,D : cropping positions and sizes in pixels (not implemented)
    dim : 'auto', '3D' or '2D'. In 3D, get_frame returns 3D with j=timeindex
    
    with nikon nd2, bioformats sometimes gives Z stacks as multipoint images. 
    checkifrealMP corrects for that, until openmicroscopy solves the problem
    """
    @classmethod
    def class_exts(cls):
        return {'nd2'} | super(BioformatsReader, cls).class_exts()

    def __init__(self, filename, dim='auto', C=[0], MP=0, X=0, Y=0, Z=0, W=0, H=0, D=0, process_func=None):
        self.filename = str(filename)
        
        if type(C) == int: C = [C]        
        self._channel = C
        self._multipoint = MP        
        self._cropX = X
        self._cropY = Y
        self._cropZ = Z
        self._cropW = W
        self._cropH = H
        self._cropD = D
        dim = str(dim)
        if dim == 'auto':
            self._dimensionauto = True
            self._dimension = 0
        elif dim == '3' or dim == '3D':
            self._dimensionauto = False
            self._dimension = 3
        else:
            self._dimensionauto = False
            self._dimension = 2
        self._initialize()
        self._validate_process_func(process_func)
    
    def _initialize(self):
        javabridge.start_vm(class_path=bioformats.JARS,max_heap_size='512m')
        self._reader = bioformats.get_image_reader(0,self.filename) 
        self._lastframe = (-1,-1)
        self._jmd = javabridge.JWrapper(self._reader.rdr.getMetadataStore())
        self._sizeMP = jwtoint(self._jmd.getImageCount())   
        if self._dimension == 3 and not self._checkifrealMP():
            self._sizeZ = self._sizeMP
            self._sizeMP = 1
            self._multipoint = 0
            self._useMPasZ = True
        else:
            if self._multipoint > self._sizeMP:
                self._multipoint = 0
            self._useMPasZ = False
        self._updatemetadata()      
           

    def _updatemetadata(self): 
        MP = self._multipoint  
        if not self._useMPasZ:
            self._sizeZ = jwtoint(self._jmd.getPixelsSizeZ(MP))
            self._planes = jwtoint(self._jmd.getPlaneCount(MP))
        self._sizeC = jwtoint(self._jmd.getPixelsSizeC(MP))
        self._sizeT = jwtoint(self._jmd.getPixelsSizeT(MP))
        self._sizeY = jwtoint(self._jmd.getPixelsSizeY(MP))
        self._sizeX = jwtoint(self._jmd.getPixelsSizeX(MP))
        self._planes = jwtoint(self._jmd.getPlaneCount(MP))               
        self._pixelX = jwtofloat(self._jmd.getPixelsPhysicalSizeX(MP))
        self._pixelY = jwtofloat(self._jmd.getPixelsPhysicalSizeY(MP))
        self._pixelZ = jwtofloat(self._jmd.getPixelsPhysicalSizeZ(MP))
        if self._pixelY == None:
            self._pixelY = self._pixelX
        if self._dimensionauto:
            if self._sizeZ > 1:
                self._dimension = 3
            else:
                self._dimension = 2
                
  
    def _checkifrealMP(self):
        if self._sizeMP == 1:
            return True
        for MP in xrange(self._sizeMP): 
            if jwtoint(self._jmd.getPixelsSizeZ(MP)) > 1:
                return True
        # check if X or Y differ between frame 1 and 2 (not 0 and 1, 0 has sometimes X=Y=0!!)
        if round(jwtofloat(self._jmd.getPlanePositionX(2,0))) != round(jwtofloat(self._jmd.getPlanePositionX(1,0))):
            if round(jwtofloat(self._jmd.getPlanePositionY(2,0))) != round(jwtofloat(self._jmd.getPlanePositionY(1,0))):
                return True            
        return False
            
    def __len__(self):
        if self._dimension == 3:
            return self._sizeT
        else:
            return self._planes
    
    def close(self):  
        bioformats.release_image_reader(0)
        
    @property
    def pixelsize(self):
        return self._pixelsize
    
    @property
    def channel(self):
        return self._channel
    @channel.setter
    def channel(self, value):
        if type(value) == int:
            self._channel = [value]
        else:
            self._channel = value
        
    @property
    def multipoint(self):
        return self._multipoint
    @multipoint.setter
    def multipoint(self, value):
        if value > self._sizeMP:
            raise 'MP index out of bounds'
        else:
            if value != self._multipoint:
                self._multipoint = value
                self._updatemetadata()
        
    @property
    def cropparams(self):
        return self._cropX, self._cropY, self._cropZ, self._cropW, self._cropH, self._cropD
    @cropparams.setter
    def cropparams(self, value):
        self._cropX, self._cropY, self._cropZ, self._cropW, self._cropH, self._cropD = value
        
    def frame_shape(self):
        return self._sizeX, self._sizeY
 
    def get_frame(self, j): 
        if self._cropH > 0 or self._cropH > 0 or self._cropH > 0:
            raise NotImplemented()

        if (self._multipoint, j) == self._lastframe:
            return self._current
        
        if self._dimension == 3:
            planelist = np.empty((self._sizeZ,4), dtype=np.int32)
            
            if not self._useMPasZ:
                self._reader.rdr.setSeries(self._multipoint) 
                for Nc in range(len(self._channel)):            
                    for z in xrange(self._sizeZ):
                        planelist[Nc * self._sizeZ + z] = [Nc,z,self._multipoint, self._reader.rdr.getIndex(z,self._channel[Nc],j)]
            else: 
                for Nc in range(len(self._channel)):            
                    for z in xrange(self._sizeZ):
                        self._reader.rdr.setSeries(z)                        
                        planelist[Nc * self._sizeZ + z] = [Nc,z,z, self._reader.rdr.getIndex(0,self._channel[Nc],j)]
            
            im = np.empty((len(self._channel),self._sizeZ,self._sizeY,self._sizeX))
            
            for plane in planelist:             
                im[plane[0],plane[1]] = self._reader.read(series=plane[2],index=plane[3])

            im = im.squeeze()
            
            self._Z = np.empty(self._sizeZ, dtype=np.float64)
            self._T = np.empty(self._sizeZ, dtype=np.float64)
            self._X = np.empty(self._sizeZ, dtype=np.float64)
            self._Y = np.empty(self._sizeZ, dtype=np.float64)
            
            for plane in planelist:
                self._T[plane[1]] = jwtofloat(self._jmd.getPlaneDeltaT(plane[2],plane[3]))
                self._X[plane[1]] = jwtofloat(self._jmd.getPlanePositionX(plane[2],plane[3]))
                self._Y[plane[1]] = jwtofloat(self._jmd.getPlanePositionY(plane[2],plane[3]))
                self._Z[plane[1]] = jwtofloat(self._jmd.getPlanePositionZ(plane[2],plane[3]))
                    
            metadata = {'pixelX': self._pixelX,
                        'pixelY': self._pixelY,
                        'pixelZ': self._pixelZ,
                        'plane': planelist[:,3],
                        'X': self._X,
                        'Y': self._Y,
                        'Z': self._Z,
                        'T': self._T,
                        'indexC': self._channel,
                        'indexT': j,
                        'indexMP': self._multipoint}                  
        else:  
            im = self._reader.read(index=j,series=self._multipoint)
            
            self._Z = np.empty(self._sizeZ, dtype=np.float64)
            self._T = np.empty(self._sizeZ, dtype=np.float64)
            self._X = np.empty(self._sizeZ, dtype=np.float64)
            self._Y = np.empty(self._sizeZ, dtype=np.float64)
                
            metadata = {'pixelX': self._pixelX,
                    'pixelY': self._pixelY,
                    'pixelZ': self._pixelZ,
                    'plane': j,
                    'X': jwtofloat(self._jmd.getPlanePositionX(self._multipoint,j)),
                    'Y': jwtofloat(self._jmd.getPlanePositionY(self._multipoint,j)),
                    'Z': jwtofloat(self._jmd.getPlanePositionZ(self._multipoint,j)),
                    'T': jwtofloat(self._jmd.getPlaneDeltaT(self._multipoint,j)),
                    'indexC': self._channel,
                    'indexT': jwtoint(self._jmd.getPlaneTheT(self._multipoint,j)),
                    'indexZ': jwtoint(self._jmd.getPlaneTheZ(self._multipoint,j)),
                    'indexMP': self._multipoint}
        
        self._lastframe = (self._multipoint, j) 
        self._current = Frame(self.process_func(im), frame_no=j, metadata=metadata)                            
        return self._current     
     
    def omexml(self):
        xml = bioformats.get_omexml_metadata(self.filename)
        return bioformats.OMEXML(xml)
    
    def mdfloat(self,MetadataRetrieve):
        try:
            exec('result = self._jmd.' + MetadataRetrieve)
            return jwtofloat(result)
        except:
            return None
    
    def mdint(self,MetadataRetrieve):
        try:
            exec('result = self._jmd.' + MetadataRetrieve)
            return jwtoint(result)
        except:
            return None
            
    def mdstr(self,MetadataRetrieve):
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
        if self._useMPasZ: result = result + "\nInterpreting MP as Z"
        return result
