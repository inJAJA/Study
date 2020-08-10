import torch

from torch import nn    # 다양한 데이터 구조, 레이어 정의되어 있음(CNN, LSTM, activation, loss, ...)
from torch.nn import init   
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random


def init_linear(linear):
    init.xavier_normal(linear.weight)   # He Initialization
    linear.bias.data.zero_()

def init_conv(conv, glu = True):
    init.kaiming_normal(conv.weight)    # Glorot Initialization
    if conv.bais is not None:
        conv.bias.data.zero_()

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')   # getattr(object, name[, default]) : object에 존재하는 속성의 값 가져옴
        fan_in = weight.data.size(1) * weight.data[0][0].numel()    # fan_in : Input Node 개수 
                                                                    # .numel() : size의 모든 인수 곱하기
        return weight * sqrt(2 / fan_in)

    @staticmethod   # 정적 메소드
                    # class에서 직접 접근할 수 있는 메소드
                    # self를 통한 접근 x
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Paranmeter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)      # setattr(object, name, value)  : object에 존재하는 속성의 값 설정

    def equal_lr(module, name='weight'):
        EqualLR.apply(module, name)

        return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding = 0):      # 모델에서 사용될 module 정의
        super().__init__()  # 이 클래스의 기반함수의 __init__가 호출된다

        weight = torch.randn(in_chanel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):                                                   # 모델에서 행해져야 하는 계산 정의 (대개 train할 때)
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])             # __init__() 에서 정의한 module들을 그대로 갖다 쓴다.
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride = 2, padding = self.pad)

        return out

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, keepdim=True) + 1e-8 )

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding = 1, groups = grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding = 1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding = 1, groups = input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype = torch.float32)
        weight = weight.view(1, 1, 3, 3)                # .view() = .reshape()
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2,3])         # .flip( input, dims ) : input=적용 데이터 / dims=적용하는 축 : reverse
        
        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups = input.shape[1])

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2 = None, 
        padding2= None,
        downsample = False,
        fused = False
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding = pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding = pad2),
                    nn.LeakyReLU(0.2)
                )
            
            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding = pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding = pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.con2(out)

        return out

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3) # .unsqueeze : 인수로 받은 위치에 새로운 차원을 삽입
        gamma, beta = style.chunk(2, 1)                     # .squeeze   : 차원의 원소가 1인 차원을 없애준다.

        out = self.norm(input)
        out = gamma * out + beta

        return out      # 284 line 