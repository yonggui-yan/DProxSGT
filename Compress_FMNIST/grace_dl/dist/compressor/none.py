from grace_dl.dist import Compressor


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor, name):
        #return [tensor], None
        return [tensor.detach().clone()], None

    def decompress(self, tensors, ctx):
        tensor, = tensors
        #return tensor
        return tensor.detach().clone()
