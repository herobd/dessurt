import os, math
import utils.img_f as img_f
import struct
import torch
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def primeFactors(n):
    ret = [1]
    # Print the number of two's that divide n 
    while n % 2 == 0:
        if len(ret)==0:
            ret.append(2)
        n = n / 2

    # n must be odd at this point 
    # so a skip of 2 ( i = i + 2) can be used 
    for i in range(3,int(math.sqrt(n))+1,2):

        # while i divides n , print i ad divide n 
        while n % i== 0:
            ret.append(i)
            n = n / i

    # Condition if n is a prime 
    # number greater than 2 
    if n > 2:
        ret.append(n)
    return ret

def getGroupSize(channels,goalSize=None):
    if goalSize is None:
        if channels>=32:
            goalSize=8
        else:
            goalSize=4
    if channels%goalSize==0:
        return goalSize
    factors=primeFactors(channels)
    bestDist=9999
    for f in factors:
        if abs(f-goalSize)<=bestDist: #favor larger
            bestDist=abs(f-goalSize)
            bestGroup=f
    return int(bestGroup)
