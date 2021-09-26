from io import BytesIO
import pprint

class Printer:
    def __init__(self, io: BytesIO):
        self._io = io
    
    def pprint(self, object: object):
        pprint.pprint(object, stream=self._io)

    def print(self, *objects: object):
        print(*objects, file=self._io)

