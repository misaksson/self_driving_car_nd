import os
import pickle
from collections import namedtuple
import hashlib

cache_path = "../cache/"

Record = namedtuple('Record', ['frame_idx', 'objects'])


class ClassifierCache(object):
    def __init__(self, classifier_path):
        # The md5 hash of the classifier file is used as a unique ID for the cache record.
        cache_ID = generate_file_md5(classifier_path)
        file_name = cache_ID + ".p"
        self.file_path = os.path.join(cache_path, file_name)
        if os.path.isfile(self.file_path):
            print("Loading classifier cache from ", self.file_path)
            self._load()
        else:
            print("No previous classifier cache available. Initialing a new cache at ", self.file_path)
            self.cache = dict()

    def _load(self):
        self.cache = dict()
        with open(self.file_path, "rb") as fid:
            while 1:
                try:
                    record = pickle.load(fid)
                except EOFError:
                    break
                self.cache[record.frame_idx] = record.objects

    def add(self, frame_idx, objects):
        if frame_idx not in self.cache:
            with open(self.file_path, "ab") as fid:
                pickle.dump(Record(frame_idx=frame_idx, objects=objects), fid)
        else:
            assert(self.cache[frame_idx] == objects)

    def get(self, frame_idx):
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        else:
            return None


def generate_file_md5(file_path):
    hash_obj = hashlib.md5()
    with open(file_path, "rb") as fid:
        while True:
            buf = fid.read(2**20)
            if not buf:
                break
            hash_obj.update(buf)
    return hash_obj.hexdigest()
