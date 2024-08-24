"""
Copyright 2024 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.LazyLinear(120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

###########################################################################################
# Corruptions (see also https://github.com/google-research/mnist-c for reference)

def spatter(x, severity=4, seed=0):
    np.random.seed(seed)
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.


    liquid_layer = np.random.normal(size=(1,28,28), loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    
    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0

    # mud spatter
    color = 63 / 255. * np.ones_like(x) * m
    x *= (1 - m)

    return np.clip(x + color, 0, 1) * 255

def line_from_points(c0, r0, c1, r1):
    if c1 == c0:
        return np.zeros((28, 28))

    # Decay function defined as log(1 - d/2) + 1
    cc, rr = np.meshgrid(np.linspace(0, 27, 28), np.linspace(0, 27, 28), sparse=True)

    m = (r1 - r0) / (c1 - c0)
    f = lambda c: m * (c - c0) + r0
    dist = np.clip(np.abs(rr - f(cc)), 0, 2.3 - 1e-10)
    corruption = np.log(1 - dist / 2.3) + 1
    corruption = np.clip(corruption, 0, 1)
    
    l = np.int32(np.floor(c0))
    r = np.int32(np.ceil(c1))
    
    corruption[:,:l] = 0
    corruption[:,r:] = 0
    
    return np.clip(corruption, 0, 1)

def zigzag(x, severity=2, seed=0):
    
    np.random.seed(seed)
    x = np.array(x) / 255.
    # a, b are length and width of zigzags
    a = 2.
    b = 2.
    
    c0, c1 = 2, 25
    r0 = np.random.randint(low=0, high=27)
    r1 = r0 + np.random.randint(low=-5, high=5)
    
    theta = np.arctan((r1 - r0) / (c1 - c0))

    # Calculate length of straight line
    d = (c1 - c0) / np.cos(theta)
    endpoints = [(0, 0)]

    for i in range(int((d - a) // (2 * a)) + 1):
        c_i = (2 * i + 1) * a
        r_i = (-1) ** i * b
        endpoints.append((c_i, r_i))
    
    max_c = (2 * a) * (d // (2 * a))
    if d != max_c:
        endpoints.append((d, r_i / (2 * (d - max_c))))
    endpoints = np.array(endpoints).T
  
    # Rotate by theta
    M = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    endpoints = M.dot(endpoints)

    cs, rs = endpoints
    cs += c0
    rs += r0

    for i in range(1, endpoints.shape[1]):
        x += line_from_points(cs[i-1], rs[i-1], cs[i], rs[i])
        x = np.clip(x, 0, 1)

    x = x * 255
    return x.astype(np.float32)

def normalize_result(result):
    return np.divide(result,np.abs(result).max() , out=np.zeros_like(result), where=result!=0)