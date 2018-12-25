class IdenticalCompressor(object):
    @staticmethod
    def compress(vec):
        return vec.clone()

    @staticmethod
    def decompress(signature):
        return signature
