class StatBase:
    def __init__(self, filename, dirname='', dump_period=-1):
        self._dirname = dirname
        self._filename = filename
        self._dump_period = dump_period

    def set_dirname(self, dirname):
        self._dirname = dirname

    def get_filename(self):
        return self._filename

    def get(self):
        assert False, 'You need to implement this function in a subclass'

    def save(self):
        assert False, 'You need to implement this function in a subclass'

    def __call__(self, simu):
        assert False, 'You need to implement this function in a subclass'
