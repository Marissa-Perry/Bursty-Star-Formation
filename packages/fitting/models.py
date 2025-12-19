import numpy as np

def gaussian(x, A, mu, sigma):
    '''
    Gaussian model for line fitting
    '''
    return A * np.exp(-(x - mu)**2/ (sigma**2))

def linear(x, mu, m, b):
    '''
    Continuum of the spectral line using y = mx + b
    This linear function is adjust so that the free parameter slope (m) value is being adjusted at the line center
    '''
    return  m * (x - mu) + b

def single_gauss_line_model(x, A, mu, sigma, m, b):
    '''
    args:
        A: amplitude
        mu: center of line
        sigma: sigma 
        b: continuum offset
    '''
    return gaussian(x, A, mu, sigma) + linear(x, mu, m, b)

# originally named "grating_Ha_line_model"
def blended_Ha_line_model(x, A, A1_NII, mu, sigma, m, b):
    '''
    Ha blended with [NII] doublet, fitting with triple Gaussian

    args:
        A: H-alpha amplitude
        A1_NII: Amplitude of bluer NII line (in which the redder line depends upon)
        mu: center of H-alpha line
        sigma: sigmas for H-alpha and NII doublet are set as the same (theoretically coming from the same gas cloud)
        m: slope of continuum offset of H-alpha
        b: continuum offset of H-alpha
    '''
    # rest frame wavelengths
    Ha_rest = 6562.819 # [A]
    NII_1_rest = 6548.050
    NII_2_rest = 6583.460
    # expected emission line centers
    mu1_NII = mu*(NII_1_rest / Ha_rest)
    mu2_NII = mu*(NII_2_rest / Ha_rest)
    
    # initialize amplitudes of NII lines to have a ~2.94:1 ratio
    A2_NII = A1_NII * 2.94
    
    return gaussian(x, A1_NII, mu1_NII, sigma) + gaussian(x, A, mu, sigma) + gaussian(x, A2_NII, mu2_NII, sigma) + linear(x, mu, m, b)

# originally named "prism_Hb_line_model"
def blended_Hb_line_model(x, A, A1_OIII, mu, sigma, m, b):
    '''
    H-beta blended with [OIII] doublet, fitting with triple Gaussian

    args:
        A: H-beta amplitude
        A1_OIII: Amplitude of bluer OIII line (in which the redder line depends upon)
        mu: center of H-beta line
        sigma: sigmas for H-beta and OIII doublet are set as the same (theoretically coming from the same gas cloud)
        m: slope of continuum offset of H-beta
        b: continuum offset of H-beta
    '''
    # rest frame wavelengths
    Hb_rest = 4861.333 # [A]
    OIII_1_rest = 4958.911
    OIII_2_rest = 5006.843
    # expected emission line centers
    mu1_OIII = mu*(OIII_1_rest / Hb_rest)
    mu2_OIII = mu*(OIII_2_rest / Hb_rest)
    
    # initialize amplitudes of OIII lines to have a ~2.98:1 ratio
    A2_OIII = A1_OIII * 2.98
    
    return gaussian(x, A1_OIII, mu1_OIII, sigma) + gaussian(x, A, mu, sigma) + gaussian(x, A2_OIII, mu2_OIII, sigma) + linear(x, mu, m, b)