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
        if not self._useMPasZ:
            MP = self._multipoint   
            self._sizeC = jwtoint(self._jmd.getPixelsSizeC(MP))
            self._sizeT = jwtoint(self._jmd.getPixelsSizeT(MP))
            self._sizeZ = jwtoint(self._jmd.getPixelsSizeZ(MP))
            self._sizeY = jwtoint(self._jmd.getPixelsSizeY(MP))
            self._sizeX = jwtoint(self._jmd.getPixelsSizeX(MP))
            self._len = jwtoint(self._jmd.getPlaneCount(MP))
                         
            self._pixelX = jwtofloat(self._jmd.getPixelsPhysicalSizeX(MP))
            self._pixelY = jwtofloat(self._jmd.getPixelsPhysicalSizeY(MP))
            self._pixelZ = jwtofloat(self._jmd.getPixelsPhysicalSizeZ(MP))
            if self._pixelY == None:
                self._pixelY = self._pixelX
             
            self._indexZ = np.empty(self._len, dtype=np.int32)
            self._indexT = np.empty(self._len, dtype=np.int32)
            self._indexC = np.empty(self._len, dtype=np.int32)
            self._Z = np.empty(self._len, dtype=np.float64)
            self._T = np.empty(self._len, dtype=np.float64)
            self._X = np.empty(self._len, dtype=np.float64)
            self._Y = np.empty(self._len, dtype=np.float64)
            
            for n in range(self._len):
                self._indexZ[n] = jwtoint(self._jmd.getPlaneTheZ(MP,n))
                self._indexT[n] = jwtoint(self._jmd.getPlaneTheT(MP,n))
                self._indexC[n] = jwtoint(self._jmd.getPlaneTheC(MP,n))
                self._T[n] = jwtofloat(self._jmd.getPlaneDeltaT(MP,n))
                self._X[n] = jwtofloat(self._jmd.getPlanePositionX(MP,n))
                self._Y[n] = jwtofloat(self._jmd.getPlanePositionY(MP,n))
                self._Z[n] = jwtofloat(self._jmd.getPlanePositionZ(MP,n))
        else:
            self._sizeC = jwtoint(self._jmd.getPixelsSizeC(0))
            self._sizeT = jwtoint(self._jmd.getPixelsSizeT(0))
            self._sizeY = jwtoint(self._jmd.getPixelsSizeY(0))
            self._sizeX = jwtoint(self._jmd.getPixelsSizeX(0))
            self._len = self._sizeT * self._sizeZ * self._sizeC
            self._pixelX = jwtofloat(self._jmd.getPixelsPhysicalSizeX(0))
            self._pixelY = jwtofloat(self._jmd.getPixelsPhysicalSizeY(0))
            self._pixelZ = jwtofloat(self._jmd.getPixelsPhysicalSizeZ(0))
            
            "Scan through and tabulate contents to enable random access."    
            self._indexZ = np.empty(self._len, dtype=np.int32)
            self._indexT = np.empty(self._len, dtype=np.int32)
            self._indexC = np.empty(self._len, dtype=np.int32)
            self._Z = np.empty(self._len, dtype=np.float64)
            self._T = np.empty(self._len, dtype=np.float64)
            self._X = np.empty(self._len, dtype=np.float64)
            self._Y = np.empty(self._len, dtype=np.float64)
            
            for ii in range(self._sizeZ):            
                for n in range(self._sizeT * self._sizeC):
                    self._indexZ[ii*self._sizeT * self._sizeC+n] = jwtoint(self._jmd.getPlaneTheZ(ii,n))
                    self._indexT[ii*self._sizeT * self._sizeC+n] = jwtoint(self._jmd.getPlaneTheT(ii,n))
                    self._indexC[ii*self._sizeT * self._sizeC+n] = jwtoint(self._jmd.getPlaneTheC(ii,n))
                    self._T[ii*self._sizeT * self._sizeC+n] = jwtofloat(self._jmd.getPlaneDeltaT(ii,n))
                    self._X[ii*self._sizeT * self._sizeC+n] = jwtofloat(self._jmd.getPlanePositionX(ii,n))
                    self._Y[ii*self._sizeT * self._sizeC+n] = jwtofloat(self._jmd.getPlanePositionY(ii,n))
                    self._Z[ii*self._sizeT * self._sizeC+n] = jwtofloat(self._jmd.getPlanePositionZ(ii,n))
      
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
        return self._len
    
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
        
        if self._dimension == 3:
            im = np.empty((len(self._channel),self._sizeZ,self._sizeY,self._sizeX))
                    
            for Nc in range(len(self._channel)):            
                for z in xrange(self._sizeZ):
                    if not self._useMPasZ:    
                        im[Nc,z] = self._reader.read(c=self._channel[Nc], z=z, t=j,series=self._multipoint)
                    else:
                        im[Nc,z] = self._reader.read(c=self._channel[Nc], z=0, t=j,series=z)
            im = im.squeeze()
            
            tlist = []    
            for i in xrange(self._len):
                if self._indexT[i] == j:
                    tlist.append(i)
                    
            metadata = {'pixelX': self._pixelX,
                        'pixelY': self._pixelY,
                        'pixelZ': self._pixelZ,
                        'plane': tlist,
                        'X': self._X[tlist],
                        'Y': self._Y[tlist],
                        'Z': self._Z[tlist],
                        'T': self._T[tlist],
                        'indexC': self._channel,
                        'indexT': j,
                        'indexMP': self._multipoint}                  
        else:  
            im = self._reader.read(index=j,series=self._multipoint)
            metadata = {'pixelX': self._pixelX,
                    'pixelY': self._pixelY,
                    'pixelZ': self._pixelZ,
                    'plane': j,
                    'X': self._X[j],
                    'Y': self._Y[j],
                    'Z': self._Z[j],
                    'T': self._T[j],
                    'indexC': self._channel,
                    'indexT': self._indexT[j],
                    'indexZ': self._indexZ[j],
                    'indexMP': self._multipoint}
                            
        return Frame(self.process_func(im), frame_no=j, metadata=metadata)       
     
    @property
    def pixel_type(self):
        raise NotImplemented()

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
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
                                  count=self._len,
                                  z=self._sizeZ,
                                  t=self._sizeT,
                                  c=self._sizeC,
                                  filename=self.filename)
