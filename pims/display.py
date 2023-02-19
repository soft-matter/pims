import uuid
import itertools
import numpy as np
import tempfile
from io import BytesIO
from base64 import b64encode
from contextlib import contextmanager, ExitStack
from fractions import Fraction
import warnings

try:
    from matplotlib.colors import ColorConverter
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    ColorConverter = None
    mpl = None
    plt = None

try:
    import av
except ImportError:
    av = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from moviepy.editor import VideoClip
except ImportError:
    VideoClip = None
except RuntimeError:
    # there is an incompatibility between moviepy 2.3.5 and imageio >= 2.5.0
    VideoClip = None


def export_pyav(sequence, filename, rate=30, bitrate=None,
                width=None, height=None, format=None, codec='mpeg4',
                pixel_format='yuv420p', autoscale=None, quality=None,
                options=None, rate_range=(16, 32)):
    """Export a sequence of images as a standard video file using PyAv.

    N.B. If the quality and detail are insufficient, increase the
    bitrate.

    Parameters
    ----------
    sequence : any iterator or array of array-like images
        The images should have two dimensions plus an
        optional third dimensions representing color.
    filename : string
        name of output file
    rate : integer
        frame rate of output file, 30 by default
    bitrate : integer
        Video bitrate is crudely guessed if None is given.
    width : integer
        By default, set the width of the images.
    height : integer
        By default, set the  height of the images. If width is specified
        and height is not, the height is autoscaled to maintain the aspect
        ratio.
    codec : string
        a valid video encoding, 'mpeg4' by default. Must be supported by the
        container format. Examples are {'mpeg4', 'wmv2', 'libx264', 'rawvideo'}
        Check https://www.ffmpeg.org/ffmpeg-codecs.html#Video-Encoders.
    format : string
        The container format. Guesses from the filename by default.
    pixel_format: string
        Video stream format, 'yuv420p' by default.
        Another possibility is 'bgr24' in combination with the 'rawvideo' codec.
    autoscale : boolean
        Linearly rescale the brightness to use the full gamut of black to
        white values. False by default for uint8 readers, True otherwise.
    quality: number or string, optional
        For 'mpeg4' codec: sets qmin and qmax
        For 'libx264' codec: sets crf. 0 = lossless, 23 = default.
        For 'wmv2' codec: sets fraction of lossless bitrate, 0.01 = default
    options : dictionary, optional
        Dictionary that will be passed to ffmpeg. Avoid using
        {'qscale:v', 'crf', 'pixel_format'}.
    rate_range : tuple of two numbers
        As extreme frame rates have playback issues on many players, by default
        the frame rate is limited between 16 and 32. When the desired frame rate
        is too low, frames will be multiplied an integer number of times. When
        the desired frame rate is too high, frames will be skipped at constant
        intervals.

    """
    if av is None:
        raise("This feature requires PyAV with FFmpeg or libav installed.")

    export_rate = _normalize_framerate(rate, *rate_range)
    sequence = CachedFrameGenerator(sequence, rate, autoscale)

    # pyav is picky with unicode strings
    codec = str(codec)
    if format is not None:
        format = str(format)
    if options is not None:
        for key in options:
            options[str(key)] = str(options[key])
    else:
        options = dict()

    if codec == str('wmv2') and bitrate is None and quality is None:
        quality = 0.01

    if quality is not None:
        if codec == str('libx264'):
            options[str('crf')] = str(quality)
        elif codec == str('wmv2'):
            if bitrate is not None:
                warnings.warn("(wmv) quality is ignored when bitrate is set.")
        elif codec == str('mpeg4'):
            options[str('qmin')] = str(quality)
            options[str('qmax')] = str(quality)
        else:
            raise NotImplemented

    # Maximum allowed timebase is 66535 (at least for mpeg4)
    # see https://github.com/mikeboers/PyAV/issues/242
    export_rate_frac = Fraction(export_rate).limit_denominator(65535)

    output = av.open(str(filename), str('w'), format=format)
    try:
        # from PyAv 6.0, options can be supplied here
        stream = output.add_stream(
            codec, rate=export_rate_frac, options=options
        )
    except TypeError:  # before, we should supply it at .open
        output = av.open(
            str(filename), str('w'), format=format, options=options
        )
        stream = output.add_stream(codec, rate=export_rate_frac)

    stream.pix_fmt = str(pixel_format)

    for frame_no in itertools.count():
        try:
            img = sequence(frame_no / export_rate)
        except IndexError:
            break
        if frame_no == 0:
            # Inspect first frame to set up stream.
            if width is None:
                stream.height = img.shape[0]
                stream.width = img.shape[1]
            else:
                stream.width = width
                stream.height = (height or
                                 width * img.shape[0] // img.shape[1])

            if bitrate is not None:
                stream.bit_rate = int(bitrate)
            elif quality is not None and codec == str('wmv2'):
                bitrate = quality * _estimate_bitrate([stream.height,
                                                       stream.width],
                                                      export_rate)
                stream.bit_rate = int(bitrate)

        # Ensure correct memory layout
        img = img.astype(img.dtype, order='C', copy=False)
        frame = av.VideoFrame.from_ndarray(img, format=str('rgb24'))
        packet = stream.encode(frame)
        if packet is not None:
            output.mux(packet)

    # Finish encoding the stream
    while True:
        try:
            packet = stream.encode()
        except av.AVError:  # End of file raises AVError since after av 0.4
            break
        if packet is None:
            break
        output.mux(packet)

    output.close()


def play(sequence, rate=30, bitrate=None,
         width=None, height=None, autoscale=True):
    """In an IPython notebook, display a sequence of images as
    an embedded video.

    N.B. If the quality and detail are insufficient, increase the
    bit rate.

    Parameters
    ----------
    sequence : any iterator or array of array-like images
        The images should have two dimensions plus an
        optional third dimensions representing color.
    rate : integer
        frame rate of output file, 30 by default
    bitrate : integer
        Video bitrate is crudely guessed if None is given.
    width : integer
        By default, set the width of the images.
    height : integer
        By default, set the  height of the images. If width is specified
        and height is not, the height is autoscaled to maintain the aspect
        ratio.
    autoscale : boolean
        Linearly rescale the brightness to use the full gamut of black to
        white values. If the datatype of the images is not 'uint8', this must
        be set to True, as it is by default.

    """
    try:
        from IPython.display import display
    except ImportError:
        raise ImportError("This feature requires IPython.")
    with tempfile.NamedTemporaryFile(suffix='.webm') as temp:
        export_pyav(sequence, bytes(temp.name), codec='libvpx', rate=rate,
                    width=width, height=height, bitrate=bitrate,
                    format='yuv420p', autoscale=True)
        temp.flush()
        display(repr_video(temp.name, 'x-webm'))


class CachedFrameGenerator(object):
    def __init__(self, sequence, rate, autoscale=None, to_bgr=False):
        self.sequence = sequence
        self._cached_frame_no = None
        self._cache = None
        self.autoscale = autoscale
        self.rate = rate
        self.to_bgr = to_bgr

    def __call__(self, t):
        frame_no = int(t * self.rate)
        if self._cached_frame_no != frame_no:
            self._cached_frame_no = frame_no
            self._cache = _to_rgb_uint8(self.sequence[frame_no], self.autoscale)
        if self.to_bgr:
            return self._cache[:, :, ::-1]
        else:
            return self._cache


def export_moviepy(sequence, filename, rate=30, bitrate=None, width=None,
                   height=None, codec='mpeg4', pixel_format='yuv420p',
                   autoscale=None, quality=None, verbose=True,
                   options=None, rate_range=(16, 32)):
    """Export a sequence of images as a standard video file using MoviePy.

    Parameters
    ----------
    sequence : any iterator or array of array-like images
        The images should have two dimensions plus an
        optional third dimensions representing color.
    filename : string
        name of output file
    rate : integer, optional
        frame rate of output file, 30 by default
        NB: The output frame rate will be limited between `rate_range`
    bitrate : integer or string, optional
        Preferably use the parameter `quality` for controlling the bitrate.
    width : integer, optional
        By default, set the width of the images.
    height : integer, optional
        By default, set the  height of the images. If width is specified
        and height is not, the height is autoscaled to maintain the aspect
        ratio.
    codec : string
        a valid video encoding, 'mpeg4' by default. Must be supported by the
        container format. Examples are {'mpeg4', 'wmv2', 'libx264', 'rawvideo'}
        Check https://www.ffmpeg.org/ffmpeg-codecs.html#Video-Encoders.
    pixel_format: string, optional
        Pixel format, 'yuv420p' by default.
        Another possibility is 'bgr24' in combination with the 'rawvideo' codec.
    quality: number or string, optional
        For 'mpeg4' codec: sets qscale:v. 1 = high quality, 5 = default.
        For 'libx264' codec: sets crf. 0 = lossless, 23 = default.
        For 'wmv2' codec: sets fraction of lossless bitrate, 0.01 = default
    autoscale : boolean, optional
        Linearly rescale the brightness to use the full gamut of black to
        white values. False by default for uint8 readers, True otherwise.
    verbose : boolean, optional
        Determines whether MoviePy will print progress. True by default.
    options : dictionary, optional
        Dictionary of parameters that will be passed to ffmpeg. Avoid using
        {'qscale:v', 'crf', 'pixel_format'}.
    rate_range : tuple of two numbers
        As extreme frame rates have playback issues on many players, by default
        the frame rate is limited between 16 and 32. When the desired frame rate
        is too low, frames will be multiplied an integer number of times. When
        the desired frame rate is too high, frames will be skipped at constant
        intervals.

    See Also
    --------
    http://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html#moviepy.video.VideoClip.VideoClip.write_videofile
    """
    if VideoClip is None:
        raise ImportError('The MoviePy exporter requires moviepy to work.')

    if options is None:
        options = dict()
    ffmpeg_params = []
    for key in options:
        ffmpeg_params.extend(['-{}'.format(key), str(options[key])])

    if rate <= 0:
        raise ValueError
    export_rate = _normalize_framerate(rate, *rate_range)

    clip = VideoClip(CachedFrameGenerator(sequence, rate, autoscale,
                                          to_bgr=(pixel_format == 'bgr24')))
    clip.duration = len(sequence) / rate
    if not (height is None and width is None):
        clip = clip.resize(height=height, width=width)

    if codec == 'wmv2' and bitrate is None and quality is None:
        quality = 0.01

    if quality is not None:
        if codec == 'libx264':
            ffmpeg_params.extend(['-crf', str(quality)])
        elif codec == 'mpeg4':
            ffmpeg_params.extend(['-qscale:v', str(quality)])
        elif codec == 'wmv2':
            if bitrate is not None:
                warnings.warn("(wmv) quality is ignored when bitrate is set.")
            else:
                bitrate = quality * _estimate_bitrate(clip.size, export_rate)
        else:
            raise NotImplemented

    if format is not None:
        ffmpeg_params.extend(['-pixel_format', str(pixel_format)])
    if bitrate is not None:
        bitrate = str(bitrate)

    clip.write_videofile(filename, export_rate, codec, bitrate, audio=False,
                         verbose=verbose, ffmpeg_params=ffmpeg_params)


if av is not None:
    export = export_pyav
elif VideoClip is not None:
    export = export_moviepy
else:
    export = None


def repr_video(fname, mimetype):
    """Load the video in the file `fname`, with given mimetype,
    and display as HTML5 video.
    """
    try:
        from IPython.display import HTML
    except ImportError:
        raise ImportError("This feature requires IPython.")
    video_encoded = open(fname, "rb").read().encode("base64")

    video_tag = """<video controls>
<source alt="test" src="data:video/{0};base64,{1}" type="video/webm">
Use Google Chrome browser.</video>""".format(mimetype, video_encoded)
    return HTML(data=video_tag)


def _scrollable_stack(sequence, width, normed=True):
    # See the public function, scrollable_stack, below.
    # This does all the work, and it returns a string of HTML and JS code,
    # as expected by Frame._repr_html_(). The public function wraps this
    # in IPython.display.HTML for the user.
    from IPython.display import Javascript, HTML, display_png
    from jinja2 import Template

    SCROLL_STACK_JS = Template("""
require(['jquery'], function() {
  if (!(window.PIMS)) {
    var stack_cursors = {};
    window.PIMS = {stack_cursors: {}};
  }
  $('#stack-{{stack_id}}-slice-0').css('display', 'block');
  window.PIMS.stack_cursors['{{stack_id}}'] = 0;
});

require(['jquery'],
$('#image-stack-{{stack_id}}').bind('mousewheel DOMMouseScroll', function(e) {
  var direction;
  var cursor = window.PIMS.stack_cursors['{{stack_id}}'];
  e.preventDefault();
  if (e.type == 'mousewheel') {
    direction = e.originalEvent.wheelDelta < 0;
  }
  else if (e.type == 'DOMMouseScroll') {
    direction = e.originalEvent.detail < 0;
  }
  var delta = direction * 2 - 1;
  if (cursor + delta < 0) {
    return;
  }
  else if (cursor + delta > {{length}} - 1) {
    return;
  }
  $('#stack-{{stack_id}}-slice-' + cursor).css('display', 'none');
  $('#stack-{{stack_id}}-slice-' + (cursor + delta)).css('display', 'block');
  window.PIMS.stack_cursors['{{stack_id}}'] = cursor + delta;
}));""")
    TAG = Template('<img src="data:image/png;base64,{{data}}" '
                   'style="display: none;" '
                   'id="stack-{{stack_id}}-slice-{{i}}" />')
    WRAPPER = Template('<div id="image-stack-{{stack_id}}", style='
                       '"width: {{width}}; float: left; display: inline;">')
    stack_id = uuid.uuid4()  # random unique identifier
    js = SCROLL_STACK_JS.render(length=len(sequence), stack_id=stack_id)
    output = '<script>{0}</script>'.format(js)
    output += WRAPPER.render(width=width, stack_id=stack_id)
    if normed:
        sequence = normalize(np.asarray(sequence))
    for i, s in enumerate(sequence):
        output += TAG.render(
            data=b64encode(_as_png(s, width, normed=False)).decode('utf-8'),
            stack_id=stack_id, i=i)
    output += "</div>"
    return output


def scrollable_stack(sequence, width=512, normed=True):
    """Display a sequence or 3D stack of frames as an interactive image
    that responds to scrolling.

    Parameters
    ----------
    sequence: a 3D Frame (or any array) or an iterable of 2D Frames (or arrays)
    width: integer
        Optional, defaults to 512. The height is auto-scaled.
    normed : Rescale the brightness to fill the gamut. All pixels in the
        stack rescaled uniformly.

    Returns
    -------
    an interactive image, contained in a IPython.display.HTML object
    """
    from IPython.display import HTML
    return HTML(_scrollable_stack(sequence, width=width, normed=normed))


def _as_png(arr, width, normed=True):
    """Create a PNG image buffer from an array."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("This feature requires PIL/Pillow.")
    w = width  # for brevity
    h = arr.shape[0] * w // arr.shape[1]
    if normed:
        arr = normalize(arr)
    img = Image.fromarray((arr * 255).astype('uint8')).resize((w, h))
    img_buffer = BytesIO()
    img.save(img_buffer, format='png')
    return img_buffer.getvalue()


def normalize(arr):
    """This normalizes an array to values between 0 and 1.

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    ndarray of float
        normalized array
    """
    ptp = arr.max() - arr.min()
    # Handle edge case of a flat image.
    if ptp == 0:
        ptp = 1
    scaled_arr = (arr - arr.min()) / ptp
    return scaled_arr


def _to_rgb_uint8(image, autoscale):
    if autoscale is None:
        autoscale = image.dtype != np.uint8

    if autoscale:
        image = (normalize(image) * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
            # sometimes 12-bit images are stored as unsigned 16-bit
            if max_value == 2**16 - 1 and image.max() < 2**12:
                max_value = 2**12 - 1
            image = (image / max_value * 255).astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)

    ndim = image.ndim
    shape = image.shape
    if ndim == 3 and shape.count(3) == 1:
        # This is a color image. Ensure that the color axis is axis 2.
        color_axis = shape.index(3)
        image = np.rollaxis(image, color_axis, 3)
    elif image.ndim == 3 and shape.count(4) == 1:
        # This is an RGBA image. Ensure that the color axis is axis 2, and 
        # drop the A values.
        color_axis = shape.index(4)
        image = np.rollaxis(image, color_axis, 3)[:, :, :3]
    elif ndim == 2:
        # Expand into color to satisfy moviepy's expectation
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    else:
        raise ValueError("Images have the wrong shape.")

    return np.asarray(image)


def _estimate_bitrate(shape, frame_rate):
    """Return a bitrate that will guarantee lossless video."""
    # Total Pixels x 8 bits x 3 channels x FPS
    return shape[0] * shape[1] * 8 * 3 * frame_rate


def _normalize_framerate(rate, min_rate=16, max_rate=32):
    """Limits the frame rate between min_rate and max_rate by integer multiples.
    """
    if rate < min_rate:
        factor = min_rate // rate
        if min_rate % rate > 0:
            factor += 1
        return rate * factor

    if rate > max_rate:
        factor = rate // max_rate
        if rate % max_rate > 0:
            factor += 1
        return rate / factor

    return rate


def _monochannel_to_rgb(image, rgb):
    """This converts a greyscale image to an RGB image, using given rgb value.

    Parameters
    ----------
    image : ndarray
        image; there should be no channel axis
    rgb : tuple of uint8
        output color in (r, g, b) format

    Returns
    -------
    ndarray of float
        rgb image, with extra inner dimension of length 3

    """
    image_rgb = normalize(image).reshape(*(image.shape + (1,)))
    image_rgb = image_rgb * np.asarray(rgb).reshape(*((1,)*image.ndim + (3,)))
    return image_rgb


def to_rgb(image, colors=None, normed=True):
    """This converts a greyscale or multichannel image to an RGB image, with
    given channel colors.

    Parameters
    ----------
    image : ndarray
        Multichannel image (channel dimension is first dimension). When first
        dimension is longer than 4, the file is interpreted as a greyscale.
    colors : list of matplotlib.colors
        List of either single letters, or rgb(a) as lists of floats. The sum
        of these lists should equal (1.0, 1.0, 1.0), when clipping needs to
        be avoided.
    normed : bool, optional
        Multichannel images will be downsampled to 8-bit RGB, if normed is
        True. Greyscale images will always give 8-bit RGB.

    Returns
    -------
    ndarray
        RGB image, with inner dimension of length 3. The RGB image is clipped
        so that values lay between 0 and 255. When normed = True (default),
        datatype is np.uint8, else it is float.
    """
    # identify whether the image has a (leading) channel axis
    if colors is None:
        has_channel_axis = image.ndim > 2 and image.shape[0] < 5
    else:
        has_channel_axis = len(colors) == image.shape[0]
    # identify number of channels and resulting shape
    if has_channel_axis:
        channels = image.shape[0]
        shape_rgb = image.shape[1:] + (3,)
    else:
        channels = 1
        shape_rgb = image.shape + (3,)
    if colors is None:
        # pick colors with high RGB luminance
        if channels == 1:    # white
            rgbs = [[255, 255, 255]]
        elif channels == 2:  # green, magenta
            rgbs = [[0, 255, 0], [255, 0, 255]]
        elif channels == 3:  # cyan, green, magenta
            rgbs = [[0, 255, 255], [0, 255, 0], [255, 0, 255]]
        elif channels == 4:  # cyan, green, magenta, red
            rgbs = [[0, 255, 255], [0, 255, 0], [255, 0, 255], [255, 0, 0]]
        else:
            raise IndexError('Not enough color values to build rgb image')
    else:
        # identify rgb values of channels using matplotlib ColorConverter
        if ColorConverter is None:
            raise ImportError('Matplotlib required for conversion to rgb')
        if channels > len(colors):
            raise IndexError('Not enough color values to build rgb image')
        rgbs = (ColorConverter().to_rgba_array(colors)*255).astype('uint8')
        rgbs = rgbs[:channels, :3]

    if has_channel_axis:
        result = np.zeros(shape_rgb)
        for i in range(channels):
            result += _monochannel_to_rgb(image[i], rgbs[i])
    else:
        result = _monochannel_to_rgb(image, rgbs[0])

    result = result.clip(0, 255)

    if normed:
        result = (normalize(result) * 255).astype('uint8')

    return result


@contextmanager
def _fig_size_cntx(fig, fig_props):
    """Resize a figure in a context

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to resize
    fig_props : dict
        Figure properties that are temporarily set.  The 'layout_engine' key is
        special-cased to provide back-compatibility with versions of Matplotlib
        that did not provide it, and further also controls the 'savefig.bbox'
        rcParam.
    """
    with ExitStack() as stack:
        if hasattr(fig, 'set_layout_engine'):
            # No matter what, explicitly restore the original layout engine at
            # exit, to work around a bug in matplotlib 3.6 where
            # savefig(..., bbox_inches="tight") would modify the engine.
            stack.callback(fig.set_layout_engine, fig.get_layout_engine())
        for k, v in fig_props.items():
            if k == 'layout_engine' and not hasattr(fig, 'set_layout_engine'):
                k, v = ('tight_layout', v != 'none')
            stack.callback(fig.set, **{k: getattr(fig, f'get_{k}')()})
            fig.set(**{k: v})
        if hasattr(fig, 'get_layout_engine'):
            sb = ('standard'
                  if type(fig.get_layout_engine()).__name__ in [
                      'NoneType', 'PlaceHolderLayoutEngine']
                  else 'tight')
        else:
            sb = 'tight' if fig.get_tight_layout() else 'standard'
        stack.enter_context(plt.rc_context({'savefig.bbox': sb}))
        yield fig


def plot_to_frame(fig, width=512, close_fig=False, fig_size_inches=None,
                  bbox_inches=None):
    """ Renders a matplotlib figure or axes object into a numpy array
    containing RGBA data of the rendered image.

    Parameters
    ----------
    fig : matplotlib Figure or Axes object
    width : integer
        The width of the resulting frame, in pixels
    close_fig : boolean
        When True, the figure will be closed after plotting
    fig_size_inches : tuple
        The figure (height, width) in inches. If None, the size is not changed.
    bbox_inches : {None, 'standard', 'tight'}
        When 'tight', tight layout is used.

    Returns
    -------
    pims.Frame object containing RGBA values (dtype uint8)
    """
    if mpl is None:
        raise ImportError("This feature requires matplotlib.")
    from pims import Frame
    if isinstance(fig, mpl.axes.Axes):
        fig = fig.figure
    fig_props = {}
    if fig_size_inches is not None:
        if fig_size_inches[0] == 0 or fig_size_inches[1] == 0:
            raise ValueError('Figure size cannot be zero.')
        fig_props['size_inches'] = fig_size_inches
    if bbox_inches is None:
        pass
    elif str(bbox_inches) == 'standard':
        fig_props['layout_engine'] = 'none'
    elif str(bbox_inches) == 'tight':
        fig_props['layout_engine'] = 'tight'
    else:
        raise ValueError("bbox_inches must be in {None, 'standard', 'tight'}")

    buf = BytesIO()
    with _fig_size_cntx(fig, fig_props) as fig:
        width_in, height_in = fig.get_size_inches()
        dpi = width / width_in
        if plt.rcParams['savefig.bbox'] == 'tight':  # set by _fig_size_cntx
            # slower, but allows tight layout
            fig.savefig(buf, format='png', dpi=dpi)
            buf.seek(0)
            image = plt.imread(buf)
        else:
            # faster, but only possible without tight layout
            fig.savefig(buf, format='rgba', dpi=dpi)
            buf.seek(0)
            buf_shape = (int(height_in * dpi), int(width_in * dpi), 4)
            image = np.frombuffer(buf.read(),
                                  dtype='uint8').reshape(*buf_shape)

    if close_fig:
        plt.close(fig)
    return Frame(image)


def plots_to_frame(figures, width=512, close_fig=False, fig_size_inches=None,
                   bbox_inches=None):
    """ Renders an iterable of matplotlib figures or axes objects into a
    pims Frame object, that will be displayed as scrollable stack in IPython.

    Parameters
    ----------
    figures : iterable of matplotlib Figure or Axes objects
    width : integer
        The width of the resulting frame, in pixels
    close_fig : boolean
        When True, the figure will be closed after plotting
    fig_size_inches : tuple
        The figure (height, width) in inches. If None, the size is not changed.
    bbox_inches : {'tight', None}
        When 'tight', tight layout is used.

    Returns
    -------
    pims.Frame object containing a stack of RGBA values (dtype uint8)
    """
    if mpl is None:
        raise ImportError("This feature requires matplotlib.")
    from pims import Frame
    if isinstance(figures, mpl.axes.Axes) or \
       isinstance(figures, mpl.figure.Figure):
        raise ValueError('Use plot_to_frame for single figures, or supply '
                         'an iterable of figures to plots_to_frame.')

    width = int(width)
    h = None

    frames = []
    for n, fig in enumerate(figures):
        im = plot_to_frame(fig, width, close_fig, fig_size_inches, bbox_inches)
        if h is None:
            h = im.shape[0]
        else:
            # make the image the same size as the first image
            if im.shape[0] != h:
                im = np.pad(im[:h], ((0, max(0, h - im.shape[0])), (0, 0),
                                     (0, 0)), mode=str('constant'))
        frames.append(im)

    return Frame(np.array(frames))
