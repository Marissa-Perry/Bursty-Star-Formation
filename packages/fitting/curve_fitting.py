import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import Akima1DInterpolator

# importing functions from self-written modules ----
from .models import (
    gaussian,      
    linear,  
    single_gauss_line_model,
    blended_Ha_line_model,
    blended_Hb_line_model
)

from plotting.spectral_plots import create_window
# ---------------------------------------------------

def initial_fits_Ha_CEERS(ID, wavs, fluxes, fluxes_err, redshift, Ha_line_obs, Ha_center):
    '''
    This function does an initial fit on the data using curve fit which we then pass in those parameters into emcee
    to do the full MCMC fit later

    fits using a triple Gaussian model, since H-alpha can blend with [NII] within the grating data 
    '''
    # fitting window
    wav_window, flux_window = create_window((wavs,fluxes),Ha_line_obs)   
    __ , flux_err_window = create_window((wavs,fluxes_err),Ha_line_obs)
    
    #initial guesses for the optimization
    guess_A = fluxes[Ha_center]
    guess_A_NII = 0.2 * guess_A   # [NII] doublet is a factor of 0.1 weaker than H-alpha emission
    guess_mu = wavs[Ha_center]

    # interpolating small window to make a guess for sigma
    spec_interp = Akima1DInterpolator(wav_window, flux_window)
    x = np.linspace(wav_window[0], wav_window[-1], 10000)
    spec = spec_interp(x)
    half_max = guess_A / 2
    idx = np.where(spec > half_max)[0]
    wave_left, wave_right = x[idx[0]], x[idx[-1]]
    guess_sigma = (wave_right - wave_left)/2

    min_idx = np.argmin(np.abs(guess_mu - wav_window))
    y_continuum = np.concatenate((np.array(flux_window[min_idx - 5:min_idx - 2]),np.array(flux_window[min_idx + 2:min_idx + 5])))
    x_continuum = np.concatenate((np.array(wav_window[min_idx - 5:min_idx - 2]),np.array(wav_window[min_idx + 2:min_idx + 5])))

    guess_m, guess_b = np.polyfit(x_continuum, y_continuum, 1) # linear fit

    # fixing an error in sigma guess for some sources
    if guess_sigma >= 300:
        guess_sigma = 80

    x0 = [guess_A, guess_A_NII, guess_mu, guess_sigma, guess_m, guess_b]

    low_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    high_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        
    ppot, ppcov = curve_fit(blended_Ha_line_model,wav_window,flux_window,p0=x0,bounds=[low_bounds, high_bounds]) 
    ppot[0] = np.abs(ppot[0]) # ensuring positive amplitude
    ppot[1] = np.abs(ppot[1]) # ensuring positive amplitude
    perr = np.sqrt(np.diag(ppcov))
    new_row = [ID, redshift, ppot, x0, wav_window, flux_window, flux_err_window, ppot[0], perr[0], ppot[2], ppot[3], ppot[1], perr[1]]

    return new_row


def initial_fits_Hb_CEERS(ID, wavs, fluxes, fluxes_err, redshift, Hb_line_obs, Hb_center):
    '''
    This function does an initial fit on the data using curve fit which we then pass in those parameters into emcee
    to do the full MCMC fit later

    fits using a single Gaussian model, since the grating data is able to fully resolve H-beta
    '''

    # fitting window
    wav_window, flux_window = create_window((wavs,fluxes),Hb_line_obs)
    __ , flux_err_window = create_window((wavs, fluxes_err),Hb_line_obs)
    
    #initial guesses for the optimization
    guess_A = fluxes[Hb_center]
    guess_mu = wavs[Hb_center]

    # interpolating small window to make a guess for sigma
    spec_interp = Akima1DInterpolator(wav_window, flux_window)
    x = np.linspace(wav_window[0], wav_window[-1], 10000)
    spec = spec_interp(x)
    half_max = guess_A / 2
    idx = np.where(spec > half_max)[0]
    wave_left, wave_right = x[idx[0]], x[idx[-1]]
    guess_sigma = (wave_right - wave_left)/2

    min_idx = np.argmin(np.abs(guess_mu - wav_window))
    y_continuum = np.concatenate((np.array(flux_window[min_idx - 5:min_idx - 2]),np.array(flux_window[min_idx + 2:min_idx + 5])))
    x_continuum = np.concatenate((np.array(wav_window[min_idx - 5:min_idx - 2]),np.array(wav_window[min_idx + 2:min_idx + 5])))

    guess_m, guess_b = np.polyfit(x_continuum, y_continuum, 1)  # linear fit

    # fixing an error in sigma guess for some sources
    if guess_sigma >= 80:
        guess_sigma = 30

    x0 = [guess_A, guess_mu, guess_sigma, guess_m, guess_b]

    low_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    high_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
        
    ppot, ppcov = curve_fit(single_gauss_line_model,wav_window,flux_window,p0=x0,bounds=[low_bounds, high_bounds]) 
    ppot[0] = np.abs(ppot[0]) # ensuring positive amplitude
    perr = np.sqrt(np.diag(ppcov))
    new_row = [ID, redshift, ppot, x0, wav_window, flux_window, flux_err_window, ppot[0], perr[0], ppot[1], ppot[2]]

    return new_row


def initial_fits_Ha_RUBIES1(source, df, ID, redshift, Ha_line_obs, Ha_center, data_type):
    '''
    This function does an initial fit on the data using curve fit which we then pass in those parameters into emcee
    to do the full MCMC fit later

    fits using a single or triple Gaussian model, depending on the data_type value
    '''

    # if using grating data, [NII] parameter
    if data_type == 'GRATING':
        wavs = df['RED GRATING WAVS'][source]
        fluxes = df['RED GRATING FLUXES'][source]
        fluxes_err = df['RED GRATING FLUXES ERR'][source]

        # use provided index if valid, otherwise compute from the wavelength
        center_idx = Ha_center
        if (center_idx is None) or (center_idx < 0) or (center_idx >= len(wavs)):
            center_idx = int(np.argmin(np.abs(wavs - Ha_line_obs)))

        # fitting window 
        wav_window, flux_window = create_window((wavs, fluxes),line_obs=Ha_line_obs,line_center_idx=center_idx,velocity_window=False)
        __, flux_err_window = create_window((wavs, fluxes_err),line_obs=Ha_line_obs,line_center_idx=center_idx,velocity_window=False)

        #initial guesses for the optimization
        guess_A = fluxes[center_idx]
        guess_A_NII = 0.2 * guess_A   # [NII] doublet is a factor of 0.1 weaker than H-alpha emission
        guess_mu = wavs[center_idx]
        # med_continuum = np.median(flux_window)

        # for linear portion of fit
        # essentially removing the emission line from the spectrum and performing a linear fit to the underlying continuum?
        min_idx = np.argmin(np.abs(guess_mu - wav_window))
        y_continuum = np.concatenate((np.array(flux_window[min_idx - 7:min_idx - 5]),np.array(flux_window[min_idx + 5:min_idx + 7])))
        x_continuum = np.concatenate((np.array(wav_window[min_idx - 7:min_idx - 5]),np.array(wav_window[min_idx + 5:min_idx + 7])))
        guess_m, guess_b = np.polyfit(x_continuum, y_continuum, 1) # linear fit
    
        # interpolating small window to make a guess for sigma
        spec_interp = Akima1DInterpolator(wav_window, flux_window)
        x = np.linspace(wav_window[0], wav_window[-1], 10000)
        spec = spec_interp(x)
        half_max = guess_A / 1.5  # changed from 2 to 1.5 to fix initial guess from overestimating sigma
        idx = np.where(spec > half_max)[0]
        wave_left, wave_right = x[idx[0]], x[idx[-1]]
        guess_sigma = (wave_right - wave_left)/2

        # fixing an error in sigma guess for some sources
        if guess_sigma >= 200:
            guess_sigma = 30
    
        x0 = [guess_A, guess_A_NII, guess_mu, guess_sigma, guess_m, guess_b]

        low_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        high_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            
        ppot, ppcov = curve_fit(blended_Ha_line_model,wav_window,flux_window,p0=x0,bounds=[low_bounds, high_bounds]) 
        ppot[0] = np.abs(ppot[0]) # ensuring positive amplitude
        ppot[1] = np.abs(ppot[1]) # ensuring positive amplitude
        perr = np.sqrt(np.diag(ppcov))
        new_row = [ID, redshift, ppot, x0, wav_window, flux_window, flux_err_window, ppot[0], perr[0], ppot[2], ppot[3], ppot[1], perr[1]]

    # if using PRISM data, no [NII] parameter
    else:
        wavs = df['PRISM WAVS'][source]
        fluxes = df['PRISM FLUXES'][source]
        fluxes_err = df['PRISM FLUXES ERR'][source]

        # use provided index if valid, otherwise compute from the wavelength
        center_idx = Ha_center
        if (center_idx is None) or (center_idx < 0) or (center_idx >= len(wavs)):
            center_idx = int(np.argmin(np.abs(wavs - Ha_line_obs)))

        # fitting window 
        wav_window, flux_window = create_window((wavs, fluxes),line_obs=Ha_line_obs,line_center_idx=center_idx,velocity_window=False)
        __, flux_err_window = create_window((wavs, fluxes_err),line_obs=Ha_line_obs,line_center_idx=center_idx,velocity_window=False) 
        
        #initial guesses for the optimization
        guess_A = fluxes[center_idx]
        guess_mu = wavs[center_idx]
        # med_continuum = np.median(flux_window)
        
        # for linear portion of fit
        min_idx = np.argmin(np.abs(guess_mu - wav_window))
        y_continuum = np.concatenate((np.array(flux_window[min_idx - 7:min_idx - 5]),np.array(flux_window[min_idx + 5:min_idx + 7])))
        x_continuum = np.concatenate((np.array(wav_window[min_idx - 7:min_idx - 5]),np.array(wav_window[min_idx + 5:min_idx + 7])))
        guess_m, guess_b = np.polyfit(x_continuum, y_continuum, 1) # linear fit
    
        # interpolating window to make a guess for sigma
        spec_interp = Akima1DInterpolator(wav_window, flux_window)
        x = np.linspace(wav_window[0], wav_window[-1], 10000)
        spec = spec_interp(x)
        half_max = guess_A / 1.5  # changed from 2 to 1.5 to fix initial guess from overestimating sigma
        idx = np.where(spec > half_max)[0]
        wave_left, wave_right = x[idx[0]], x[idx[-1]]
        guess_sigma = (wave_right - wave_left)/2

        # fixing an error in sigma guess for some sources
        if guess_sigma >= 200:
            guess_sigma = 130
    
        x0 = [guess_A, guess_mu, guess_sigma, guess_m, guess_b]

        low_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        high_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
            
        ppot, ppcov = curve_fit(single_gauss_line_model,wav_window,flux_window,p0=x0,bounds=[low_bounds, high_bounds]) 
        ppot[0] = np.abs(ppot[0]) # ensuring positive amplitude
        perr = np.sqrt(np.diag(ppcov))
        new_row = [ID, redshift, ppot, x0, wav_window, flux_window, flux_err_window, ppot[0], perr[0], ppot[1], ppot[2], None, None]

    return new_row


def initial_fits_Hb_RUBIES(source, df, ID, redshift, Hb_center, data_type):
    '''
    This function does an initial fit on the data using curve fit which we then pass in those parameters into emcee
    to do the full MCMC fit later

    fits using a single or triple Gaussian model, depending on the data_type value
    '''

    # if using PRISM data, [OIII] doublet blends with H-beta emission
    if data_type != 'GRATING':

        wavs = df['PRISM WAVS'][source]
        fluxes = df['PRISM FLUXES'][source]
        fluxes_err = df['PRISM FLUXES ERR'][source]
        
        # fitting window
        wav_window, flux_window = create_window((wavs, fluxes), line_center_idx=Hb_center, velocity_window=False)   
        __ , flux_err_window = create_window((wavs, fluxes_err), line_center_idx=Hb_center, velocity_window=False) 
        
        #initial guesses for the optimization
        guess_A = fluxes[Hb_center]
        guess_A_OIII = 1.5 * guess_A # [OIII] doublet is a factor of 2-5 stronger than H-beta emission
        guess_mu = wavs[Hb_center]
        # med_continuum = np.median(flux_window)
        
        # for linear portion of fit

        # essentially removing the emission line from the spectrum and performing a linear fit to the underlying continuum?
        min_idx = np.argmin(np.abs(guess_mu - wav_window))
        y_continuum = np.concatenate((np.array(flux_window[min_idx - 5:min_idx - 2]),np.array(flux_window[min_idx + 2:min_idx + 5])))
        x_continuum = np.concatenate((np.array(wav_window[min_idx - 5:min_idx - 2]),np.array(wav_window[min_idx + 2:min_idx + 5])))
        guess_m, guess_b = np.polyfit(x_continuum, y_continuum, 1) # linear fit
    
        # interpolating window to make a guess for sigma
        spec_interp = Akima1DInterpolator(wav_window, flux_window)
        x = np.linspace(wav_window[0], wav_window[-1], 10000)
        spec = spec_interp(x)
        half_max = guess_A / 2
        idx = np.where(spec > half_max)[0]
        wave_left, wave_right = x[idx[0]], x[idx[-1]]
        guess_sigma = (wave_right - wave_left)/2

        # fixing an error in sigma guess for some sources
        if guess_sigma >= 200:
            guess_sigma = 130

        # fixing an error on negative amplitude for one source
        if guess_A < 0:
            guess_A = np.abs(guess_A)
            guess_A_OIII = np.abs(guess_A_OIII)
        
        x0 = [guess_A, guess_A_OIII, guess_mu, guess_sigma, guess_m, guess_b]

        low_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        high_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            
        ppot, ppcov = curve_fit(blended_Hb_line_model,wav_window,flux_window,p0=x0,bounds=[low_bounds, high_bounds]) 
        ppot[0] = np.abs(ppot[0]) # ensuring positive amplitude
        ppot[1] = np.abs(ppot[1]) # ensuring positive amplitude
        perr = np.sqrt(np.diag(ppcov))
        new_row = [ID, redshift, ppot, x0, wav_window, flux_window, flux_err_window, ppot[0], perr[0], ppot[2], ppot[3]]

    # if grating data, no [OIII] parameter
    else:

        wavs = df['RED GRATING WAVS'][source]
        fluxes = df['RED GRATING FLUXES'][source]
        fluxes_err = df['RED GRATING FLUXES ERR'][source]

        # fitting window
        wav_window, flux_window = create_window((wavs, fluxes), line_center_idx=Hb_center, velocity_window=False)   
        __ , flux_err_window = create_window((wavs, fluxes_err), line_center_idx=Hb_center, velocity_window=False)
        
        #initial guesses for the optimization
        guess_A = fluxes[Hb_center]
        guess_mu = wavs[Hb_center]
        # med_continuum = np.median(flux_window)
        
        # for linear portion of fit
        min_idx = np.argmin(np.abs(guess_mu - wav_window))
        y_continuum = np.concatenate((np.array(flux_window[min_idx - 5:min_idx - 2]),np.array(flux_window[min_idx + 2:min_idx + 5])))
        x_continuum = np.concatenate((np.array(wav_window[min_idx - 5:min_idx - 2]),np.array(wav_window[min_idx + 2:min_idx + 5])))
        guess_m, guess_b = np.polyfit(x_continuum, y_continuum, 1) # linear fit
    
        # interpolating small window to make a guess for sigma
        spec_interp = Akima1DInterpolator(wav_window, flux_window)
        x = np.linspace(wav_window[0], wav_window[-1], 10000)
        spec = spec_interp(x)
        half_max = guess_A / 2
        idx = np.where(spec > half_max)[0]
        wave_left, wave_right = x[idx[0]], x[idx[-1]]
        guess_sigma = (wave_right - wave_left)/2

        # fixing an error in sigma guess for some sources
        if guess_sigma >= 200:
            guess_sigma = 30

        x0 = [guess_A, guess_mu, guess_sigma, guess_m, guess_b]

        low_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        high_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
            
        ppot, ppcov = curve_fit(single_gauss_line_model,wav_window,flux_window,p0=x0,bounds=[low_bounds, high_bounds]) 
        ppot[0] = np.abs(ppot[0]) # ensuring positive amplitude
        perr = np.sqrt(np.diag(ppcov))
        new_row = [ID, redshift, ppot, x0, wav_window, flux_window, flux_err_window, ppot[0], perr[0], ppot[1], ppot[2]]

    return new_row

def initial_fits_Ha_RUBIES2(source, df, ID, redshift, Ha_line_obs, Ha_center, data_type):
    '''
    This function does an initial fit on the data using curve fit which we then pass in those parameters into emcee
    to do the full MCMC fit later

    fits sources which use PRISM + Grating data. In this step, the grating data on H-alpha is being fit using a triple Gaussian model
    '''

    wavs = df['RED GRATING WAVS'][source]
    fluxes = df['RED GRATING FLUXES'][source]
    fluxes_err = df['RED GRATING FLUXES ERR'][source]

    # use provided index if valid, otherwise compute from the wavelength
    center_idx = Ha_center
    if (center_idx is None) or (center_idx < 0) or (center_idx >= len(wavs)):
        center_idx = int(np.argmin(np.abs(wavs - Ha_line_obs)))

    # fitting window 
    wav_window, flux_window = create_window((wavs, fluxes),line_obs=Ha_line_obs,line_center_idx=center_idx,velocity_window=False)
    __, flux_err_window = create_window((wavs, fluxes_err),line_obs=Ha_line_obs,line_center_idx=center_idx,velocity_window=False)

    #initial guesses for the optimization
    guess_A = fluxes[center_idx]
    guess_A_NII = 0.2 * guess_A   # [NII] doublet is a factor of 0.1 weaker than H-alpha emission
    guess_mu = wavs[center_idx]
    # med_continuum = np.median(flux_window)

    # for linear portion of fit
    # essentially removing the emission line from the spectrum and performing a linear fit to the underlying continuum?
    min_idx = np.argmin(np.abs(guess_mu - wav_window))
    y_continuum = np.concatenate((np.array(flux_window[min_idx - 7:min_idx - 5]),np.array(flux_window[min_idx + 5:min_idx + 7])))
    x_continuum = np.concatenate((np.array(wav_window[min_idx - 7:min_idx - 5]),np.array(wav_window[min_idx + 5:min_idx + 7])))
    guess_m, guess_b = np.polyfit(x_continuum, y_continuum, 1) # linear fit

    # interpolating small window to make a guess for sigma
    spec_interp = Akima1DInterpolator(wav_window, flux_window)
    x = np.linspace(wav_window[0], wav_window[-1], 10000)
    spec = spec_interp(x)
    half_max = guess_A / 1.5  # changed from 2 to 1.5 to fix initial guess from overestimating sigma
    idx = np.where(spec > half_max)[0]
    wave_left, wave_right = x[idx[0]], x[idx[-1]]
    guess_sigma = (wave_right - wave_left)/2

    # fixing an error in sigma guess for some sources
    if guess_sigma >= 200:
        guess_sigma = 30

    x0 = [guess_A, guess_A_NII, guess_mu, guess_sigma, guess_m, guess_b]

    low_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    high_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        
    ppot, ppcov = curve_fit(blended_Ha_line_model,wav_window,flux_window,p0=x0,bounds=[low_bounds, high_bounds]) 
    ppot[0] = np.abs(ppot[0]) # ensuring positive amplitude
    ppot[1] = np.abs(ppot[1]) # ensuring positive amplitude
    perr = np.sqrt(np.diag(ppcov))
    new_row = [ID, redshift, ppot, x0, wav_window, flux_window, flux_err_window, ppot[0], perr[0], ppot[2], ppot[3], ppot[1], perr[1]]

    return new_row