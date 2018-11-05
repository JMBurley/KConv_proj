# KConv_proj
## Introduction
This repo shows basic validation of a method for pred. timeseries data (the output) from a causally-connected but poorly correlated predictive variable (the input).  

First, an example: below we show google search interest for a particular book (the input) and library rentals for that book in Seattle (the output).
[Figure 1: WWZ](IMAGES/WWZ_Plot_V1.png?raw=true "Fig1")

A human read of this data is that there is a clear relationship between the two -- the sharp spike in search interest is associated with an increase in rentals, with a long tail that persists after the search interest has dissipated. ie. there is a causative relationship, but poor correlation between these timeseries.  However, without direct correlation, a machine read of this data would generally struggle to find a good relationship. This repo contains the early stages of a model to find such fits, and produces the following prediction of the output for two simple cases
[Figure 2: WWZ](IMAGES/UPDATE_WWZ.png?raw=true "Fig2")
[Figure 3: HG](IMAGES/UPDATE_HG.png?raw=true "Fig3")

## So What Does This Method Do?
The work optimises a kernel that is convolved with the input timeseries to approximate the output.  In principle, such a kernel could take any form, although we probably want to be smart and constrain it to a time signature that represents sensible physical behaviour.  Currently this repo uses a truncated power law, but in theory we could optimse a broad suite of problems (particularly if the author can find time to write a wrapper around SciPy curvefit to allow optimisation of integer parameters)

Our main function operates at $O(n \log n)$, so the method is fast (although exact speed will depend on the local gradients that SciPy curvefit can utilise to fit the data ie. how many calls to the $O(n \log n)$ method does curve_fit need to make?).  Again, the author could write a wrapper to SciPy to improve this --- performing a low-order curvefit solve with which to seed a higher-order solve --- but that's outside the scope of my work for now.

## Is There Anything Else in this Repo?
Yes.  There is a Jupyter Notebook showing some iteractive tools on a toy dataset (various versions of the Books example above) where you can call 1) the kernel convolution method and 2) a lagged linear machine learning model with a Ridge Regression.  We see that the kernel convolution is resilient to overfitting, faster, and generally produces better fits.

