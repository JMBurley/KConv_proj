import numpy as np
import pandas as pd
""" See README.md in this folder for motivation, context, and the problem solved.  The rest of this docstring explains 
 the current state of code's structure and cleanliness.   

This code was written in 2018 as part of a heavy time-crunch project which has three major impediments to being 
clean code.
    - Original solution was written in R and the structure of the solution was ported directly to Python
    - The author was still thinking in MATLAB formalism, so even in those constraints the style is not pythonic
    - Code needs to maintain compatibility with a demonstration notebook (making some sensible refactors 
      time-prohibitive)
    
For now, the code is being left as-is because it illustrates a useful technique despite the implementation being far 
below production code.  For this reason, readers should feel free to adapt the method without being beholden to the 
exact formalisms of this code; I would rewrite this if implementing today.  Since 2018, edits have been made for 
readability (docstrings, type hints, variable renaming that does not require significant refactor).
"""


def _apply_convolution(kernel: np.ndarray, input_data: np.ndarray, c: float, conv_type: str = 'fft') -> np.ndarray:
    """ Convolve kernel & input_data using the chosen `conv_type` AND apply background constant `c` to the result.
    Mathematical operation is y = Conv(a*x^n,input_data)+c.

    Args:
        kernel: convolution kernel, intended to represent a mapping from input to some output.  Can often be interpreted
                as system memory from input to transient response.
        input_data: array-like to convolve with kernel.
        c: background constant in y= Conv(a*x^n,input_data)+c.
        conv_type: one of ['discrete', 'fft']

    Returns:
        y: output array Conv(a*x^n,input_data)+c
    """
    # init vars
    nx = len(input_data)
    nk = len(kernel)
    y  = np.zeros(nx+nk)

    # --- Apply selected convolve method
    if conv_type == 'discrete':
        # -- Discrete convolution ('moving window' -- rolling summation of kernel*ydata[i] at the points in xdata)
        for i in range(nx):
            y[i:(i + nk)] += kernel * input_data[i]
        # Trim
        y = y[0:nx]

    elif conv_type == 'fft':
        # -- FFT convolution. (Faster, but tricker to verify validity of soln. Trim is different)
        y = np.convolve(input_data, kernel, mode='full')
        # Trim
        y = y[0:len(y)-nk+1]

    else:
        raise ValueError(f'conv_type `{conv_type}` is not a valid input')

    # Add background level
    y = y+c
    return y


def JB_Kernel(input_data: pd.Series, a: float, n: float, c: float, trim_ratio: float) -> np.ndarray:
    """ Establishes a kernel for convolution that is compatible with `_apply_convolution` function's use of
    np.convolve(gk, input_data)

    Schematically, we fit Conv(a*x^n,input_predictor)+c, where x is a monotonically increasing timeseries 1:max:1, and
    the kernel is defined by a*x^n and the global background constant c

    Note that many structural choices in this function are constrained by using scipy curvefit to optimise the model
    fit, and the design decision that all model functions should accept inputs f(input_data, *scipy_curvefit_output) in
    order to be consistent with scipy syntax.

    Args:
        input_data: array-like to convolve with Kernel. (length of this input controls the length of the kernel)
        a: factor in kernel a*x^n
        n: exponent in kernel a*x^n
        c: not used in kernel, as it is the background constant in y= Conv(a*x^n,input_data)+c.
            Arg exists so that this function can take the scipy curvefit(JB_Model) output as input
        trim_ratio: Truncate kernel `gk` when value is less than max(gk)/trim_ratio, or len(input_data) if condition is
            never true in input_data range.

    Returns:
        gk: general kernel, a numpy array that can be used in np.convolve(gk, input_data)
    """
    # -- timeseries for kernel
    # (start at 1 to avoid 'divide by zero'-esque errors; len()+1 to maintain timeseries length)
    lag = 0
    xdata = list(range(1, len(input_data) + 1))
    # -- make kernel
    # create: gk = a*(xdata)^n
    gk = np.multiply(a, np.power(xdata, n))
    # add a lag time
    gk = np.pad(gk, (lag, 0), 'constant')

    # Trim
    idx = np.argmax(gk <= (max(gk)/trim_ratio))  # cut at threshold.  If no match, returns 0
    if idx == 0:
        idx = len(gk)  # same length as input
    gk = gk[0:idx]
    return gk
    

def JB_Model(input_data, a, n, c, trim_ratio) -> np.ndarray:
    """ Define and apply kernel convolution of kernel `a*x^n` like Conv(a*x^n,input_data)+c
    in a manner compatible with optimisation via:
        scipy.optimize.curve_fit(JB_Model, input_data, outputData_to_match, bounds=Tuple[Tuple,Tuple])

    nb./ curve_fit is agnostic as to the name of the four variables tuned (vars are determined by the entries in
    bounds=), but for internal consistency we consider these variables to be `a, n, c, trim_ratio`.

    Args:
        input_data: array-like to convolve with Kernel.
        a: factor in kernel a*x^n
        n: exponent in kernel a*x^n
        c: background constant in y= Conv(a*x^n,input_data)+c.
        trim_ratio: Truncate kernel `gk` when value is less than max(gk)/trim_ratio, or len(input_data) if condition is
            never true in input_data range.

    Returns:
        y: output array Conv(a*x^n,input_data)+c
    """
    # Establish a kernel
    gk = JB_Kernel(input_data, a, n, c, trim_ratio)

    # Convolve
    y = _apply_convolution(gk, input_data, c, conv_type='fft')
    return y

