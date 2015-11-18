import numpy as np


def getDeltas(seq, derivative=2, winsize=2):
    # First stack the static features
    ret = seq[:]
    for i in xrange(derivative):
        seq = _getSingleDeltas(seq)
        ret.extend(seq)
    return ret


def _getSingleDeltas(feature, winsize=2):
    '''
    Calculates a single pass deltas for the given feature
    returns the calculated feature stacked upon the given feature
    '''
    ret = []
    # Calculates the denominator: 2* \sum_n^N n*n
    denom = 2. * sum(x**2 for x in xrange(1, winsize + 1))
    # iterate over all frames
    for frameindex in xrange(len(feature)):
        # We calculate the difference in between two frames
        # In the border case of having the current frame is < winsize, we use the
        # Current frame as the "replacement" effectively exting the array left and right by
        # the frames at the positions +- winsize
        fwd = bwd = feature[frameindex]
        innersum = 0
        # Winsize will range between 1 and winsize+1, since we want to have the
        # adjacent frames
        for k in xrange(1, winsize + 1):
            # Check if our features are in range, if not we use the default
            # setting
            # Since one of the features will certainly be not out of range (
            # except having
            # a zero or one length frame length), we don't get any zeros in the
            # result
            if frameindex + k < len(feature):
                fwd = feature[frameindex + k]
            if frameindex - k >= 0:
                bwd = feature[frameindex - k]
            innersum += k * (fwd - bwd)
        ret.append(innersum / denom)
    return ret
