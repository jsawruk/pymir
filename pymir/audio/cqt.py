#!/usr/bin/env python
#-*- coding:utf-8 -*-
from numpy import fft,array,arange,zeros,dot,transpose
from math import sqrt,cos,pi

def cqt(x):
    N = len(x)
    y = array(zeros(N))
    a = sqrt(2/float(N))
    for k in range(N):
        for n in range(N):
            y[k] += x[n]*cos(pi*(2*n+1)*k/float(2*N))
        if k==0:
            y[k] = y[k]*sqrt(1/float(N))
        else:
            y[k] = y[k]*a
    return y

def idct(y):
    N = len(y)
    x = array(zeros(N))
    a = sqrt(2/float(N))
    for n in range(N):
        for k in range(N):
            if k==0:
                x[n] += sqrt(1/float(N))*y[k]*cos(pi*(2*n+1)*k/float(2*N))
            else:
                x[n] += a*y[k]*cos(pi*(2*n+1)*k/float(2*N))
    return x

def cqtmatrix(N, M, fs):
    return P
