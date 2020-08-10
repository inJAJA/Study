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

