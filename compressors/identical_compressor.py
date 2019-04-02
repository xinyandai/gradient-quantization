class IdenticalCompressor(object):
    def __init__(self, size=None, shape=None, args=None):
        pass

    @staticmethod
    def compress(vec):
        return vec.clone()

    @staticmethod
    def decompress(signature):
        return signature
