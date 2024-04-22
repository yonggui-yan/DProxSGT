import torch

from grace_dl.dist import Compressor


class QSGD_infCompressor(Compressor):

    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm(float('inf'))
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, shape

    def decompress(self, tensors, ctx):
        tensors, norm = tensors
        decode_output = tensors.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        #print('tesnor_decompressed size is ', tensor_decompressed.size(),flush=True)
        #print('ctx  is ', ctx,flush=True)
        tensor_decompressed = tensor_decompressed.view(*ctx)
        #tensor_decompressed = tensor_decompressed.view(ctx)
        return tensor_decompressed
