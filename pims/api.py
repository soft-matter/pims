from pims.image_sequence import ImageSequence

def not_available(requirement):
    raise ImportError("This reader requires {0}.".format(requirement))

try:
    import pims.ffmpeg_reader
    if pims.ffmpeg_reader.available():
        Video = pims.ffmpeg_reader.FFmpegVideoReader
    else:
        raise ImportError()
except ImportError:
    Video = not_available("ffmpeg")
