from torch.autograd import Function
import torch

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k, t = k.cuda(), t.cuda()
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output.to(input.device)
        
        return grad_input, None, None

class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, k):
            if k >= 31:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input, None, None
        
def T_quantize(input, prec):
    # quantize, normalization and recover normalization
    if prec >= 31:
        return input
    x = input
    Tmax = torch.max(input).detach()  # Tmax[0]  # torch.max(input).detach()#
    Tmin = torch.min(input).detach()  # Tmax[1]  # torch.min(input).detach()#
    T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
    T = torch.clamp(T, 1e-10, 255.)   # this is clamp normalization
    x = torch.clamp(x, 0 - T, T)
    x_s = x / T
    weight_q = qfn().apply(x_s, prec)
    weight_q = weight_q * T
    return weight_q

def T_quantize_activation(input, prec, T_a):
    if prec >= 31:
        return input
    x = input
    x_s = x / T_a
    activation_q = qfn().apply(x_s, prec)
    activation_q = activation_q * T_a
    return activation_q

def quantize(input, prec):
    if prec >= 31:
        return input
    x = input
    Tmax = torch.max(input).detach()
    Tmin = torch.min(input).detach()
    T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
    T = torch.clamp(T, 1e-10, 255.)
    x = torch.clamp(x, 0 - T, T)
    x_s = x / T
    activation_q = qfn().apply(x_s, prec)
    # no recover because the original input is already [-1 - +1]
    return activation_q