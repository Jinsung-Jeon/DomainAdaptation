from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(x):
        return x.view_as(x)

    @staticmethod
    def backward(grad_output):
        output = grad_output.neg()
        return output, None


