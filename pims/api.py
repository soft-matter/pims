from pims.image_sequence import ImageSequence

def not_available(requirement):
    def raiser():
        raise ImportError(
            "This reader requires {0}.".format(requirement))
    return raiser

try:
    import pims.ffmpeg_reader
    if pims.ffmpeg_reader.available():
        Video = pims.ffmpeg_reader.FFmpegVideoReader
    else:
        raise ImportError()
except (ImportError, IOError):
    Video = not_available("ffmpeg")

from pims.tiff_stack import TiffStack_pil, TiffStack_libtiff
