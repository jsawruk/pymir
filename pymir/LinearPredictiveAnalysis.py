import numpy as np
import pymir
from scipy.linalg import toeplitz
import itertools

def lpc(seq, order=None):
    '''
    Function: lpc
    Summary: Computes for each given sequence the LPC ( Linear predictive components ) sequence.
    Examples:
    Attributes:
        @param (seq):A sequence of time-domain frames, usually obtained by .frames()
        @param (order) default=None: Size of the returning cepstral components. If None is given,
                                     we use len(seq) as default, otherwise order +1
    Returns: A tuple, which elements are (lpc coefficents,error_term). The error term is the sqare root of the squared prediction error.
    '''
    # In this lpc method we use the slow( if the order is >50) autocorrelation approach.
    acseq = np.array(pymir.Frame.autocorr(seq, order))
    # Using pseudoinverse to obtain a stable estiamte of the toeplitz matrix
    a_coef = np.dot(np.linalg.pinv(toeplitz(acseq[:-1])), -acseq[1:].T)
    # Squared prediction error, defined as e[n] = a[n] + \sum_k=1^order (a_k *
    # s_{n-k})
    err_term = acseq[0] + sum(a * c for a, c in zip(acseq[1:], a_coef))
    return a_coef.tolist(), np.sqrt(err_term)


def lpcc(seq, err_term, order=None):
    '''
    Function: lpcc
    Summary: Computes the linear predictive cepstral compoents. Note: Returned values are in the frequency domain
    Examples: audiofile = AudioFile.open('file.wav',16000)
              frames = audiofile.frames(512,np.hamming)
              for frame in frames:
                frame.lpcc()
              Note that we already preprocess in the Frame class the lpc conversion!
    Attributes:
        @param (seq):A sequence of lpc components. Need to be preprocessed by lpc()
        @param (err_term):Error term for lpc sequence. Returned by lpc()[1]
        @param (order) default=None: Return size of the array. Function returns order+1 length array. Default is len(seq)
    Returns: List with lpcc components with default length len(seq), otherwise length order +1
    '''
    if order is None:
        order = len(seq) - 1
    lpcc_coeffs = [np.log(err_term), -seq[0]]
    for n in xrange(2, order + 1):
        # Use order + 1 as upper bound for the last iteration
        upbound = (order + 1 if n > order else n)
        lpcc_coef = -sum(i * lpcc_coeffs[i] * seq[n - i - 1]
                         for i in xrange(1, upbound)) * 1. / upbound
        lpcc_coef -= seq[n - 1] if n <= len(seq) else 0
        lpcc_coeffs.append(lpcc_coef)
    return lpcc_coeffs

def lsp(lpcseq,rectify=True):
    '''
    Function: lsp
    Summary: Computes Line spectrum pairs ( also called  line spectral frequencies [lsf]). Does not use any fancy algorithm except np.roots to solve
    for the zeros of the given polynom A(z) = 0.5(P(z) + Q(z))
    Examples: audiofile = AudioFile.open('file.wav',16000)
              frames = audiofile.frames(512,np.hamming)
              for frame in frames:
                frame.lpcc()
    Attributes:
        @param (lpcseq):The sequence of lpc coefficients as \sum_k=1^{p} a_k z^{-k}
        @param (rectify) default=True: If true returns only the values >= 0, since the result is symmetric. If all values are wished, specify rectify = False
    Returns: A list with same length as lpcseq (if rectify = True), otherwise 2*len(lpcseq), which represents the line spectrum pairs
    '''
    # We obtain 1 - A(z) +/- z^-(p+1) * (1 - A(z))
    # After multiplying everything through it becomes
    # 1 - \sum_k=1^p a_k z^{-k} +/- z^-(p+1) - \sum_k=1^p a_k z^{k-(p+1)}
    # Thus we have on the LHS the usual lpc polynomial and on the RHS we need to reverse the coefficient order
    # We assume further that the lpc polynomial is a valid one ( first coefficient is 1! )

    # the rhs does not have a constant expression and we reverse the coefficients
    rhs = [0] + lpcseq[::-1] + [1]
    # The P polynomial
    P = []
    # The Q polynomial
    Q = []
    # Assuming constant coefficient is 1, which is required. Moreover z^{-p+1} does not exist on the lhs, thus appending 0
    lpcseq = [1] + lpcseq[:] + [0]
    for l,r in itertools.izip_longest(lpcseq,rhs):
        P.append(l + r)
        Q.append(l - r)
    # Find the roots of the polynomials P,Q ( numpy assumes we have the form of: p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    # mso we need to reverse the order)
    p_roots = np.roots(P[::-1])
    q_roots = np.roots(Q[::-1])
    # Keep the roots in order
    lsf_p = sorted(np.angle(p_roots))
    lsf_q = sorted(np.angle(q_roots))
    # print sorted(lsf_p+lsf_q),len([i for  i in lsf_p+lsf_q if i > 0.])
    if rectify:
        # We only return the positive elements, and also remove the final Pi (3.14) value at the end,
        # since it always occurs
        return sorted(i for i in lsf_q + lsf_p if (i > 0))[:-1]
    else:
        # Remove the -Pi and +pi at the beginning and end in the list
        return sorted(i for i in lsf_q + lsf_p)[1:-1]
