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
    mode : 'auto', '2D', '3D', '3D_MPasZ', '3D_MPasT_TasZ'
    
    the latter two are for files that bioformats scrambles:
     - use 3D_MPasZ when the Z stacks are interpreted as a multipoint 
       collection of TCYX (sizeZ = 1)
     - 3D_MPasT_TasZ when time frames are interpreted as a multipoint 
       collection of ZCYZ, with Z stacks in T (sizeZ = 1)
    the auto format does not cover 3D_MPasT_TasZ and is not 100% foolproof for 3D_MPasZ
    """
    @classmethod
    def class_exts(cls):
        return {'nd2'} | super(BioformatsReader, cls).class_exts()

    def __init__(self, filename, mode='auto', C=[0], MP=0, X=0, Y=0, Z=0, W=0, H=0, D=0, process_func=None):
        self.filename = str(filename)
        if type(C) == int: C = [C]        
        if not mode in ('2D','3D','3D_MPasZ','3D_MPasT_TasZ','auto'):   
            raise 'Unsupported mode'
        self._mode = mode
        self._channel = C
        self._multipoint = MP        
        self._cropX = X
        self._cropY = Y
        self._cropZ = Z
        self._cropW = W
        self._cropH = H
        self._cropD = D
        self._validate_process_func(process_func)
        self._initialize()
    
    def _initialize(self):
        javabridge.start_vm(class_path=bioformats.JARS,max_heap_size='512m')
        self._reader = bioformats.get_image_reader(0,self.filename) 
        self._lastframe = (-1,-1)
        self._current = None
        self._jmd = javabridge.JWrapper(self._reader.rdr.getMetadataStore())        
        self._sizeMP = jwtoint(self._jmd.getImageCount())           
        if self._mode == 'auto':
            self._mode = self._automode()
        self._updatemetadata()      
           

    def _updatemetadata(self): 
        if self._mode in ('3D_MPasZ','3D_MPasT_TasZ'):
            self._multipoint = 0                     
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
  
    def _automode(self):
        # when sizeZ > 1, it is 3D
        if jwtoint(self._jmd.getPixelsSizeZ(0)) > 1:
            return '3D'
        
        # when sizeZ == 1 and no multipoint, it is 2D
        if self._sizeMP == 1:
            return '2D'
    
        # when multipoint, check if any other sizeZ > 1, if so then mixed 2D 
        # and 3D, not supported so go to 'raw' 2D mode
        for MP in xrange(self._sizeMP): 
            if jwtoint(self._jmd.getPixelsSizeZ(MP)) > 1:
                return '2D'
                
        #require at least MP of 4 for MPasZ or MPasT
        if self._sizeMP < 5:
            return '2D'
                
        #check if X or Y were displaced, do not use index = 0 as it mostly gives X,Y = 0,0
        if round(jwtofloat(self._jmd.getPlanePositionX(2,0))) != round(jwtofloat(self._jmd.getPlanePositionX(1,0))):
            return '2D'
        if round(jwtofloat(self._jmd.getPlanePositionY(2,0))) != round(jwtofloat(self._jmd.getPlanePositionY(1,0))):
            return '2D'
            
        #check which deltaT is larger: T or MP. The smallest will be Z
        sizeC = jwtoint(self._jmd.getPixelsSizeC(0))
        deltaT_MP = jwtofloat(self._jmd.getPlaneDeltaT(1,0)) - jwtofloat(self._jmd.getPlaneDeltaT(0,0))
        deltaT_T = jwtofloat(self._jmd.getPlaneDeltaT(0,sizeC)) - jwtofloat(self._jmd.getPlaneDeltaT(0,0))
        if deltaT_MP < deltaT_T:
            return '3D_MPasZ'
        else:
            return '3D_MPasT_TasZ'
            
    def __len__(self):
        if self._mode == '2D':
            return self._planes
        elif self._mode == '3D':
            return self._sizeT
        elif self._mode == '3D_MPasZ':
            return self._sizeT
        elif self._mode == '3D_MPasT_TasZ':
            return self._sizeMP
    
    def close(self):  
        bioformats.release_image_reader(0)
        
    @property
    def pixelsizes(self):
        if self._mode == '2D':
            return {'X': self._pixelX, 'Y': self._pixelY}
        else: 
            return {'X': self._pixelX, 'Y': self._pixelY, 'Z': self._pixelZ}
        
    @property
    def sizes(self):
        if self._mode == '2D' or self._mode == '3D':
            return {'MP': self._sizeMP, 'X': self._sizeX, 'Y': self._sizeY, 
                    'Z': self._sizeZ, 'C': self._sizeC, 'T': self._sizeT}
        elif self._mode == '3D_MPasZ':
            return {'MP': 1, 'X': self._sizeX, 'Y': self._sizeY, 
                    'Z': self._sizeMP, 'C': self._sizeC, 'T': self._sizeT}
        elif self._mode == '3D_MPasT_TasZ':
            return {'MP': 1, 'X': self._sizeX, 'Y': self._sizeY, 
                    'Z': self._sizeT, 'C': self._sizeC, 'T': self._sizeMP}
    
    @property
    def channel(self):
        return self._channel
    @channel.setter
    def channel(self, value):
        if value != self._channel:
            if type(value) == int:
                self._channel = [value]
            else:
                self._channel = value
            self._lastframe = (-1,-1)
            self._current = None
            
    @property
    def mode(self):
        return self._mode
    @mode.setter
    def mode(self, value):
        if value in ('2D','3D','3D_MPasZ','3D_MPasT_TasZ') and value != self._mode:
            self._mode = value
            self._lastframe = (-1,-1)
            self._current = None
            self._updatemetadata()
    
    @property
    def multipoint(self):
        return self._multipoint
    @multipoint.setter
    def multipoint(self, value):
        if value > self._sizeMP or self._mode in ('3D_MPasZ','3D_MPasT_TasZ'):
            raise 'MP index out of bounds'
        else:
            if value != self._multipoint:
                self._multipoint = value
                self._lastframe = (-1,-1)
                self._current = None
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
            
        if self._mode == '2D':
            im, metadata = self._get_frame2D(self._multipoint,j)
        elif self._mode == '3D':
            im, metadata = self._get_frame3D(j)
        elif self._mode == '3D_MPasZ':
            im, metadata = self._get_frame3D_MPasZ(j)
        elif self._mode == '3D_MPasT_TasZ':
            im, metadata = self._get_frame3D_MPasT_TasZ(j)
            
        self._lastframe = (self._multipoint, j) 
        self._current = Frame(self.process_func(im), frame_no=j, metadata=metadata)                            
        return self._current

    def _get_frame3D(self, t): 
        planelist = np.empty((len(self._channel),self._sizeZ,2), dtype=np.int32)
        for (Nc,c) in enumerate(self._channel):            
            for z in xrange(self._sizeZ):
                planelist[Nc, z] = [self._multipoint,
                             self._reader.rdr.getIndex(z,c,t)]
                             
        im3D, metadata = self._get_framelist(planelist)      
        
        return im3D, metadata
        
    def _get_frame3D_MPasZ(self, t): 
        planelist = np.empty((len(self._channel),self._sizeMP,2), dtype=np.int32)
        for (Nc,c) in enumerate(self._channel):                  
            for z in xrange(self._sizeMP):
                self._reader.rdr.setSeries(z)                        
                planelist[Nc, z] = [z, self._reader.rdr.getIndex(0,c,t)]
        im3D, metadata = self._get_framelist(planelist)
        
        metadata['indexZ'] = metadata['indexMP']
        metadata['indexMP'] = np.zeros(planelist.shape[1])
        
        return im3D, metadata
        
    def _get_frame3D_MPasT_TasZ(self, t): 
        planelist = np.empty((len(self._channel),self._sizeT,2), dtype=np.int32)
        self._reader.rdr.setSeries(t)
        for (Nc,c) in enumerate(self._channel):                  
            for z in xrange(self._sizeT):                        
                #planelist[Nc, z] = [t, self._reader.rdr.getIndex(0,c,z)]
                planelist[Nc, z] = [(t*self._sizeT)%self._sizeMP + z,(t*self._sizeT)//self._sizeMP + c]
        im3D, metadata = self._get_framelist(planelist)
        
        metadata['indexZ'] = metadata['indexT']
        metadata['indexT'] = metadata['indexMP']        
        metadata['indexMP'] = np.zeros(planelist.shape[1])
        
        return im3D, metadata
        
        
    def _get_framelist(self, planelist):         
        imlist = np.empty((planelist.shape[0],planelist.shape[1],self._sizeY,self._sizeX))
        dt = np.dtype([('plane', ">i4"),('indexMP', ">i4"),('indexZ', ">i4"),('indexT', ">i4"),  
                       ('X', ">f8"), ('Y', ">f8"),('Z', ">f8"), ('T', ">f8")])
        metadata = np.empty(planelist.shape[1], dtype=dt)  
        
        for (Nc,zstack) in enumerate(planelist):   
            for (Nz,plane) in enumerate(zstack):
                imlist[Nc,Nz], metadata[Nz] = self._get_frame2D(MP=plane[0],j=plane[1])
        
        return imlist.squeeze(), metadata
    
    def _get_frame2D(self, MP, j):
        im = self._reader.read(series=MP,index=j)    
        dt = np.dtype([('plane', ">i4"),('indexMP', ">i4"),('indexZ', ">i4"),('indexT', ">i4"),  
                       ('X', ">f8"), ('Y', ">f8"),('Z', ">f8"), ('T', ">f8")])
        metadata = np.array((j,
                MP,
                jwtoint(self._jmd.getPlaneTheZ(MP,j)),
                jwtoint(self._jmd.getPlaneTheT(MP,j)),
                jwtofloat(self._jmd.getPlanePositionX(MP,j)),
                jwtofloat(self._jmd.getPlanePositionY(MP,j)),
                jwtofloat(self._jmd.getPlanePositionZ(MP,j)),
                jwtofloat(self._jmd.getPlaneDeltaT(MP,j))),dtype=dt)

        return self.process_func(im), metadata
     
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
            Frame Shape: {w} x {h}
            Reader mode: {mode}""".format(w=self._sizeX,
                                              h=self._sizeY,
                                              mp=self._sizeMP,
                                              mpa=self._multipoint,
                                              count=self._planes,
                                              z=self._sizeZ,
                                              t=self._sizeT,
                                              c=self._sizeC,
                                              filename=self.filename,
                                              mode = self._mode)
        return result
