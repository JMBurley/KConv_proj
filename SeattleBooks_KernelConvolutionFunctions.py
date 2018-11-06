import numpy as np 

def ConvDisc(GK,input_data,c):
    # init vars
    nx = len(input_data)
    nk = len(GK)
    Y   = np.zeros(nx+nk)

    #### Apply discrete convolution ('moving window' -- rolling summation of kernel*ydata[i] at the points in xdata)
#     for i in range(nx):
#         Y[i:(i+nk)]+= GK*input_data[i]
# #         Y[i:(i+nk)]= GK*input_data[i] + Y[i:(i+nk)]
#     # Trim
#     Y = Y[0:(nx)]

    #### Apply FFT convolution. (Faster, but tricker to verify validity of soln. Trim is different)
    Y = np.convolve(input_data, GK, mode='full')
    # Trim
    Y = Y[0:len(Y)-nk+1]
        

    # Add background level
    Y = Y+c
    return Y

def JB_Kernel(input_data,a,n,c,TrimRatio):
    '''
    #### Establish a kernel
    # Schematically, we fit Conv(a*x^n,input_predictor)+c, where x is a monotonically increasing timeseries 1:max:1   
    #  x and y would be easier to read if I wasn't working around scipy
    # (correcting for annoying scipy issue where I can only use a single series for prediction)
    '''
    ### timeseries for kernel
    #(start at 1 to avoid 'divide by zero'-esque errors; len()+1 to maintain timeseries length)
    lag = 0
    xdata = list(range(1,len(input_data)+1))
    ### make kernel
    #create: GK    = a*(xdata)^n
    GK = np.multiply(a,np.power(xdata,n))
    # add a lag time
    GK = np.pad(GK, (lag,0), 'constant')

    # Trim
    idx = np.argmax(GK<=(max(GK)/TrimRatio)) # cut at threshold.  if no match, returns 0
#     print(f'idx1: {idx}')
    if idx == 0:
        idx = len(GK) # same length as input
#     print(f'idx2: {idx}')
    GK = GK[0:(idx)]    
    return GK
    

### Define and apply Kernel function for the convolution
def JB_Model(input_data,a,n,c,TrimRatio):
    lag = 0
    ### Establish a kernel
    GK = JB_Kernel(input_data,a,n,c,TrimRatio)
    
    ### Convolve
    Y = ConvDisc(GK,input_data,c)
    return Y

