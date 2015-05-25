Custom Readers
==============

You can define your own PIMS reader with a minimal amount of customized code.

Components
----------

By subclassing ``FramesSequence``, you get PIMS's lazy-loading and slicing
behavior for free. You have to provide:

* a method for reading a single frame into a numpy array
* a method for knowing the length of the sequence
* properties giving the shape and the numpy data type of each frame

Basic Example
-------------

.. code-block:: python

   from pims import FramesSequence, Frame

   class MyReader(FramesSequence):

       def __init__(self, filename):
           self.filename = filename
           self._len =  # however many frames there will be
           self._dtype =  # the numpy datatype of the frames
           self._frame_shape =  # the shape, like (512, 512), of an
                                # individual frame -- maybe get this by
                                # opening the first frame
           # Do whatever setup you need to do to be able to quickly access
           # individual frames later.

       def get_frame(self, i):
           # Access the data you need and get it into a numpy array.
           # Then return a Frame like so:
           return Frame(my_numpy_array, frame_no=i)

        def __len__(self):
            return self._len

        @property
        def frame_shape(self):
            return self._frame_shape

        @property
        def pixel_type(self):
            return self._dtype


The ``__init__`` method is completely customizable. It can take whatever
arguments make sense for your file format.

Optionally, you might also wish to customize the ``__repr__`` and define a
``close`` method, which the base class calls when exiting a context manager.

Plugging into PIMS's open function
----------------------------------

The function ``pims.open`` dispatches to a PIMS reader based on the file
extension. To associate your reader with particle file
extensions, add a ``class_exts`` class method. For example, the following
code will invoke ``MyReader`` to open files ending in :file:`.xyz`.

.. code-block:: python

   class MyReader(FramesSequence):

       ...

       @classmethod
       def class_exts(cls):
           return {'xyz'} | super(MyReader, cls).class_exts()

New readers can be defined or imported at any time. They will be detected the
next time ``pims.open`` is called, as it searches all subclasses of
``FramesSequence`` for an eligible reader.

To prioritize readers, a field ``class_priority`` can optionally be given to the
reader. A higher priority will be chosen over a lower priority. Default for
all readers is 10.

Example Demonstrating Generality of PIMS Design
-----------------------------------------------

Here is a reader that extracts tiles from a tiled image or "sprite sheet."

.. code-block:: python

    class SpriteSheet(FramesSequence):
    """
    This is a class for providing an easy interface into
    a sprite sheet of uniformly sized images.

    Parameters
    ----------
    sheet : ndarray
        The sprite sheet.  It should consist of N paneled images.  In
        this version all possible positions have an image, this may be changed
        to limit the number of acessable images in the sheet to be less than
        the possible number.
    rows : int
    cols : int
        The number of rows and columns of sprites.
        The sprite size is computed from these + the shape of the sheet.
    """

    def __init__(self, sheet, rows, cols):
        self._sheet = sheet
        sheet_height, sheet_width = sheet.shape
        if sheet_width % cols != 0:
            raise ValueError("Sheet width not evenly divisible by cols")
        if sheet_height % rows != 0:
            raise ValueError("Sheet height not evenly divisible by rows")

        self._sheet_shape = (rows, cols)

        self._im_sz = sheet_height // rows, sheet_width // cols
        self._sprite_height, self._sprite_width = self._im_sz

        self._dtype = sheet.dtype

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def frame_shape(self):
        return self._im_sz

    def __len__(self):
        return np.prod(self._sheet_shape)

    def get_frame(self, n):
        r, c = np.unravel_index(n, self._sheet_shape)
        slc_r = slice(r*self._sprite_height, (r+1)*self._sprite_height)
        slc_c = slice(c*self._sprite_width, (c+1)*self._sprite_width)
        tmp = self._sheet[slc_r, slc_c]
        return Frame(self.process_func(tmp), frame_no=n)
