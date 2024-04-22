from grace_dl.dist import Memory


class FilterResidualMemory(Memory):
    def __init__(self, beta=1.0):
        self.residuals = {}
        self.beta = beta

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor =  self.residuals[name] +  tensor
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        if name in self.residuals:
        	self.residuals[name] = (1-self.beta)*self.residuals[name] +self.beta* residual
        else:
            self.residuals[name] = residual
