#!/usr/bin/env python
#-*- coding:utf-8 -*-
from numpy import fft,array,arange,zeros,dot,transpose
from math import sqrt,cos,pi

qmatrix = [[ 16 , 11 , 10 , 16 , 24  , 40  , 51  , 61 ],
           [ 12 , 12 , 14 , 19 , 26  , 58  , 60  , 55 ],
           [ 14 , 13 , 16 , 24 , 40  , 57  , 69  , 56 ],
           [ 14 , 17 , 22 , 29 , 51  , 87  , 80  , 62 ],
           [ 18 , 22 , 37 , 56 , 68  , 109 , 103 , 77 ],
           [ 24 , 35 , 55 , 64 , 81  , 104 , 113 , 92 ],
           [ 49 , 64 , 78 , 87 , 103 , 121 , 120 , 101],
           [ 72 , 92 , 95 , 98 , 112 , 100 , 103 , 99 ] ]

def dct(x):
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

