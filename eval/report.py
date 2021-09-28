from io import BytesIO
import pprint
from typing import List

class Printer:
    def __init__(self, *ios: List[BytesIO]):
        self._ios = ios
    
    def pprint(self, object: object):
        for io in self._ios:
            pprint.pprint(object, stream=io)

    def print(self, *objects: object):
        for io in self._ios:
            print(*objects, file=io)

