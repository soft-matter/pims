class FileLocker(object):
    """
    A context manager to lock and unlock a file

    See http://docs.python.org/2/library/contextlib.html
    http://docs.python.org/2/library/stdtypes.html#typecontextmanager
    http://docs.python.org/2/reference/datamodel.html#context-managers
    """
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        return False