
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee

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

def final_fits_Ha_CEERS(source, initial_fitting_output, ID, redshift, emission_line_label='Ha', diagnose=True):
    '''
    args:
        source (int): index of source within dataframe
        initial_fitting_output (dataframe): parameters obtain from initial fitting
        ID (int): source identification
        redshift (float): redshift of source
        emission_line_label (str): label for emission line
        diagnose (bool): choice to output fitting diagnostic plots

    returns: dataframe with final fitting parameters
    '''

    # inner functions --------------------------------------------------------
    def log_likelihood(theta, x, y, yerr):
        '''
        This is the likelihood function we are using for emcee to run
        
        This likelihood function is the maximum likelihood assuming gaussian errors.
        '''
        ################
        #Making the model of the emission line
        model = blended_Ha_line_model(x, *theta)
        
        #getting the log likelihood, this is similar to chi2
        lnL = -0.5 * np.nansum((y - model) ** 2 / yerr**2)
        
        return lnL

    def log_prior(theta, wave_center, Amp_max):
        '''
        The prior function to be used against the parameters to impose certain criteria for the fitting
        '''
        #Theta values that goes into our Gaussian Model
        A, A1_NII, mu, sigma, m, b = theta
        
        #the left most and right most the central wavelength can vary
        left_mu = wave_center - 20   # [A] # had to change these as well this is how much mu can vary
        right_mu = wave_center + 20  # [A] # had to change these as well this is how much mu can vary
        
        #min and max amplitude of the emission line
        min_A = 0
        max_A = Amp_max * 2
        
        sigma_window_left = 1.5 # [A] # had to change these for the input spectra these are left bounds for sigma
        sigma_window_right = 30 # [A] # had to change these for the input spectra these are right bounds for sigma
            
        if (0 < A < max_A) & (0 < A1_NII < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right) & (b > 0):
            return 0.0
        else:
            return -np.inf

    def log_probability(theta, x, y, yerr, first_wave, Amp_max):
        
        lp = log_prior(theta, first_wave, Amp_max)
        if not np.isfinite(lp):
            #print('Probability is infinite')
            return -np.inf
        
        prob = lp + log_likelihood(theta, x, y, yerr)

        #print(f'Prob:{prob:.3E}')
        return prob
    # -----------------------------------------------------------------------------------

    label = 'Grating spectra'

    # initial fit with curve_fit()
    initial_fit = initial_fitting_output['ppots'][source]
    
    guess_A = initial_fit[0]
    guess_A_NII = initial_fit[1]
    guess_mu = initial_fit[2] 
    guess_sigma = initial_fit[3] 
    guess_m = initial_fit[4] 
    guess_b = initial_fit[5]
    
    #making walkers so that we can use emcee to explore the parameter space
    #centered on the best results from minimization
    amp_jump = np.random.normal(loc = guess_A,            # centered on best A from minimization
                                scale = guess_A/10,       # can wander 1/10 of the value of A
                                size = 32).reshape(-1, 1) 
    
    amp_NII_jump = np.random.normal(loc = guess_A_NII,            # centered on best A from minimization
                                    scale = guess_A_NII/10,       # can wander 1/10 of the value of A
                                    size = 32).reshape(-1, 1) 
    
    wavelength_jump = np.random.normal(loc = guess_mu,    # centered on best mu from minimization
                                    scale = 50,        # can wander +/- 0.005 microns 
                                    size = 32).reshape(-1, 1)
    
    sigma_jump = np.random.normal(loc = guess_sigma, scale = 20, size = 32).reshape(-1, 1)
    
    powerb = np.log10(np.abs(guess_b))
    b_jump = np.random.normal(loc = guess_b, scale = 1*10**powerb, size = 32).reshape(-1, 1)

    powerm = np.log10(np.abs(guess_m))
    m_jump = np.random.normal(loc = guess_m, scale = 1*10**powerm, size = 32).reshape(-1, 1)
    
    # #################
    # # Diagnostic plotting to see if the parameters were jumping to large values
    # # They should be concentrated near their best fit results values
    # #################
    # if diagnose == True:
    #     print('Checking the Walker Jumps')
    #     fig, ax = plt.subplots(nrows = 3, ncols = 2, constrained_layout = True)
        
    #     ax[0, 0].hist(amp_jump)
    #     ax[0, 0].set_xlabel('Amplitude')

    #     ax[0, 0].hist(amp_NII_jump)
    #     ax[0, 0].set_xlabel('NII Amplitude')
        
    #     ax[0, 1].hist(wavelength_jump)
    #     ax[0, 1].set_xlabel(r'$\mu$')
        
    #     ax[1, 0].hist(sigma_jump)
    #     ax[1, 0].set_xlabel(r'$\sigma$')

    #     ax[1, 1].hist(m_jump)
    #     ax[1, 1].set_xlabel('m')

    #     ax[2, 0].hist(b_jump)
    #     ax[2, 0].set_xlabel('b')
        
    #     plt.show()
    
    #stacking along the columns
    starting_walkers = np.hstack((amp_jump,
                                amp_NII_jump,
                                wavelength_jump, 
                                sigma_jump, 
                                m_jump, 
                                b_jump))

    # emcee window
    emcee_flux = initial_fitting_output['flux_window'][source]
    emcee_wave = initial_fitting_output['wav_window'][source]
    emcee_flux_err = initial_fitting_output['flux_err_window'][source]

    #initializing walker positions
    pos = starting_walkers
    nwalkers, ndim = pos.shape

    #initializing sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(emcee_wave, emcee_flux, emcee_flux_err, guess_mu, guess_A)
                                )
    #running it 
    '''
    added skip_initial_state_check=True to fix the following error
    "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance"
    '''
    sampler.run_mcmc(pos, 3000, progress=False, skip_initial_state_check=True)

    #getting values back
    samples = sampler.get_chain() # shape: [nsteps, nwalkers, ndim]
    flat_samples = sampler.get_chain(flat=True)
    LnL_chain = sampler.flatlnprobability
    burn_in = 2000 
    
    emcee_df = pd.DataFrame()
    emcee_df['A'] = flat_samples[burn_in:, 0]
    emcee_df['A_NII'] = flat_samples[burn_in:, 1]
    emcee_df['mu'] = flat_samples[burn_in:, 2]
    emcee_df['sigma'] = flat_samples[burn_in:, 3]
    emcee_df['m'] = flat_samples[burn_in:, 4]
    emcee_df['b'] = flat_samples[burn_in:, 5]
    emcee_df['LnL'] = LnL_chain[burn_in:]
    
    emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
    
    fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi) # M: calculating area of Gaussian
    emcee_df['Fluxes'] = fluxes_emcee
    
#     if diagnose == True:
        
#         print('Checking Parameter Posterior Distributions')
#         fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
        
#         emcee_df.A.hist(ax = ax[0, 0])
#         emcee_df.mu.hist(ax = ax[0, 1])
#         emcee_df.sigma.hist(ax = ax[1, 0])
#         #emcee_df.m.hist(ax = ax[1, 0])
#         emcee_df.b.hist(ax = ax[1, 1])
#         emcee_df.m.hist(ax = ax[1, 1])
#         plt.show()

    # parameter values
    median_params = emcee_df.quantile(q=0.5).values[:-2]
    perc_16th_params = emcee_df.quantile(q=0.16).values[:-2]
    perc_84th_params = emcee_df.quantile(q=0.84).values[:-2]
    A = median_params[0]
    A_NII = median_params[1]
    mu = median_params[2]
    sigma = median_params[3]

    if diagnose == True:
        xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 500)
        gauss_16th_perc = blended_Ha_line_model(xarr, *perc_16th_params)
        gauss_84th_perc = blended_Ha_line_model(xarr, *perc_84th_params)
        
        plt.figure()
        plt.title('ID: '+str(ID)+' line: '+str(emission_line_label)+' z='+str(redshift), fontsize=15)
        plt.step(emcee_wave, emcee_flux, color = 'grey', label = label, where='mid')
        #plt.scatter(emcee_wave, emcee_spec, color = 'black')
        plt.plot(xarr, blended_Ha_line_model(xarr, *median_params), color='#407899', label = 'emcee model')
        plt.fill_between(xarr, gauss_16th_perc, gauss_84th_perc, color='#407899', alpha=0.3)
        plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
        plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)
        plt.legend()
        plt.show()
    
    # gathering errors
    # A
    A_quantile_16 = emcee_df.quantile(q=0.16).values[0]
    A_quantile_84 = emcee_df.quantile(q=0.84).values[0]
    A_16th_error = A-A_quantile_16
    A_84th_error = A_quantile_84-A
    A_mean_quantiles = np.mean([A_16th_error,A_84th_error])

    # A_NII
    A_NII_quantile_16 = emcee_df.quantile(q=0.16).values[1]
    A_NII_quantile_84 = emcee_df.quantile(q=0.84).values[1]
    A_NII_16th_error = A_NII-A_NII_quantile_16
    A_NII_84th_error = A_NII_quantile_84-A_NII
    A_NII_mean_quantiles = np.mean([A_NII_16th_error,A_NII_84th_error])

    # sigma
    sigma_quantile_16 = emcee_df.quantile(q=0.16).values[3]
    sigma_quantile_84 = emcee_df.quantile(q=0.84).values[3]
    sigma_16th_error = sigma-sigma_quantile_16
    sigma_84th_error = sigma_quantile_84-sigma
    sigma_mean_quantiles = np.mean([sigma_16th_error,sigma_84th_error])
    
    # [mean err (mean of 16% & 84%), 16% err, 84% err]
    A_err = [A_mean_quantiles,A_16th_error,A_84th_error]
    A_NII_err = [A_NII_mean_quantiles,A_NII_16th_error,A_NII_84th_error]
    sigma_err = [sigma_mean_quantiles,sigma_16th_error,sigma_84th_error]
    
    new_row = [ID, redshift, samples, emcee_wave, emcee_flux, emcee_flux_err, median_params, 
            A, A_err, mu, sigma, sigma_err, perc_16th_params, perc_84th_params, A_NII, A_NII_err]

    return new_row


def final_fits_Hb_CEERS(source, initial_fitting_output, ID, redshift, emission_line_label='Hb', diagnose=True):
    '''
    args:
        source (int): index of source within dataframe
        initial_fitting_output (dataframe): parameters obtain from initial fitting
        ID (int): source identification
        redshift (float): redshift of source
        emission_line_label (str): label for emission line
        diagnose (bool): choice to output fitting diagnostic plots

    returns: dataframe with final fitting parameters
    '''

    # inner functions --------------------------------------------------------
    def log_likelihood(theta, x, y, yerr):
        '''
        This is the likelihood function we are using for emcee to run
        
        This likelihood function is the maximum likelihood assuming gaussian errors.
        '''
        ################
        # The value we are trying to fit
        #A, mu, sigma, m, b = theta
        
        #Making the model of the emission line
        model = single_gauss_line_model(x, *theta)
        
        #getting the log likelihood, this is similar to chi2
        lnL = -0.5 * np.nansum((y - model) ** 2 / yerr**2)
        
        return lnL

    def log_prior(theta, wave_center, Amp_max):
        '''
        The prior function to be used against the parameters to impose certain criteria for the fitting
        '''
        #Theta values that goes into our Gaussian Model
        A, mu, sigma, m, b = theta
        
        #the left most and right most the central wavelength can vary
        left_mu = wave_center - 20  # [A] # had to change these as well this is how much mu can vary
        right_mu = wave_center + 20 # [A] # had to change these as well this is how much mu can vary
        
        #min and max amplitude of the emission line
        min_A = 0
        max_A = Amp_max * 2
        
        sigma_window_left = 1.5 # [A] # had to change these for the input spectra these are left bounds for sigma
        sigma_window_right = 30 # [A] # had to change these for the input spectra these are right bounds for sigma
            
        if (0 < A < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right) & (b > 0):
            return 0.0
        else:
            return -np.inf
        
    def log_probability(theta, x, y, yerr, first_wave, Amp_max):
        
        lp = log_prior(theta, first_wave, Amp_max)
        if not np.isfinite(lp):
            #print('Probability is infinite')
            return -np.inf
        
        prob = lp + log_likelihood(theta, x, y, yerr)

        #print(f'Prob:{prob:.3E}')
        return prob
    # -----------------------------------------------------------------------------------

    label = 'Grating spectra'
    
    # initial fit with curve_fit()
    initial_fit = initial_fitting_output['ppots'][source]
    
    guess_A = initial_fit[0]
    guess_mu = initial_fit[1] 
    guess_sigma = initial_fit[2] 
    guess_m = initial_fit[3] 
    guess_b = initial_fit[4]
    
    
    #making walkers so that we can use emcee to explore the parameter space
    #centered on the best results from minimization
    amp_jump = np.random.normal(loc = guess_A,            # centered on best A from minimization
                                scale = guess_A/10,       # can wander 1/10 of the value of A
                                size = 32).reshape(-1, 1) 
    
    wavelength_jump = np.random.normal(loc = guess_mu,    # centered on best mu from minimization
                                       scale = 50,        # can wander +/- 0.005 microns 
                                       size = 32).reshape(-1, 1)
    
    
    sigma_jump = np.random.normal(loc = guess_sigma, scale = 20, size = 32).reshape(-1, 1)
    
    powerb = np.log10(np.abs(guess_b))
    b_jump = np.random.normal(loc = guess_b, scale = 1*10**powerb, size = 32).reshape(-1, 1)

    powerm = np.log10(np.abs(guess_m))
    m_jump = np.random.normal(loc = guess_m, scale = 1*10**powerm, size = 32).reshape(-1, 1)
    
    # #################
    # # Diagnostic plotting to see if the parameters were jumping to large values
    # # They should be concentrated near their best fit results values
    # #################
    # if diagnose == True:
    #     print('Checking the Walker Jumps')
    #     fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
        
    #     ax[0, 0].hist(amp_jump)
    #     ax[0, 0].set_xlabel('Amplitude')
        
    #     ax[0, 1].hist(wavelength_jump)
    #     ax[0, 1].set_xlabel(r'$\mu$')
        
    #     ax[1, 0].hist(sigma_jump)
    #     ax[1, 0].set_xlabel(r'$\sigma$')
        
    #     ax[1, 1].hist(b_jump)
    #     ax[1, 1].set_xlabel('b')

    #     ax[1, 1].hist(m_jump)
    #     ax[1, 1].set_xlabel('m')
        
    #     plt.show()
    
    #stacking along the columns
    starting_walkers = np.hstack((amp_jump,
                                  wavelength_jump, 
                                  sigma_jump, 
                                  m_jump, 
                                  b_jump))

    # emcee window
    emcee_flux = initial_fitting_output['flux_window'][source]
    emcee_wave = initial_fitting_output['wav_window'][source]
    emcee_flux_err = initial_fitting_output['flux_err_window'][source]

    #initializing walker positions
    pos = starting_walkers
    nwalkers, ndim = pos.shape

    #initializing sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(emcee_wave, emcee_flux, emcee_flux_err, guess_mu, guess_A)
                                   )
    '''
    added skip_initial_state_check=True to fix the following error
    "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance"
    '''
    sampler.run_mcmc(pos, 3000, progress=False, skip_initial_state_check=True)

    #getting values back
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(flat=True)
    LnL_chain = sampler.flatlnprobability
    burn_in = 2000 
    
    emcee_df = pd.DataFrame()
    emcee_df['A'] = flat_samples[burn_in:, 0]
    emcee_df['mu'] = flat_samples[burn_in:, 1]
    emcee_df['sigma'] = flat_samples[burn_in:, 2]
    emcee_df['m'] = flat_samples[burn_in:, 3]
    emcee_df['b'] = flat_samples[burn_in:, 4]
    emcee_df['LnL'] = LnL_chain[burn_in:]
    
    emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
    
    fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi) # M: calculating area of Gaussian
    emcee_df['Fluxes'] = fluxes_emcee
    
#     if diagnose == True:
        
#         print('Checking Parameter Posterior Distributions')
#         fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
        
#         emcee_df.A.hist(ax = ax[0, 0])
#         emcee_df.mu.hist(ax = ax[0, 1])
#         emcee_df.sigma.hist(ax = ax[1, 0])
#         #emcee_df.m.hist(ax = ax[1, 0])
#         emcee_df.b.hist(ax = ax[1, 1])
#         plt.show()
    
    # parameter values
    median_params = emcee_df.quantile(q=0.5).values[:-2]
    perc_16th_params = emcee_df.quantile(q=0.16).values[:-2]
    perc_84th_params = emcee_df.quantile(q=0.84).values[:-2]
    A = median_params[0]
    mu = median_params[1]
    sigma = median_params[2]

    if diagnose == True:
        xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 500)
        gauss_16th_perc = single_gauss_line_model(xarr, *perc_16th_params)
        gauss_84th_perc = single_gauss_line_model(xarr, *perc_84th_params)
        
        plt.figure()
        plt.title('ID: '+str(ID)+' line: '+str(emission_line_label)+' z='+str(redshift), fontsize=15)
        plt.step(emcee_wave, emcee_flux, color = 'grey', label = label, where='mid')
        #plt.scatter(emcee_wave, emcee_spec, color = 'black')
        plt.plot(xarr, single_gauss_line_model(xarr, *median_params), color='#407899', label = 'emcee model')
        plt.fill_between(xarr, gauss_16th_perc, gauss_84th_perc, color='#407899', alpha=0.3)
        plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
        plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)
        plt.legend()
        plt.show()
    
    # gathering errors
    # A
    A_quantile_16 = emcee_df.quantile(q=0.16).values[0]
    A_quantile_84 = emcee_df.quantile(q=0.84).values[0]
    A_16th_error = A-A_quantile_16
    A_84th_error = A_quantile_84-A
    A_mean_quantiles = np.mean([A_16th_error,A_84th_error])

    # sigma
    sigma_quantile_16 = emcee_df.quantile(q=0.16).values[2]
    sigma_quantile_84 = emcee_df.quantile(q=0.84).values[2]
    sigma_16th_error = sigma-sigma_quantile_16
    sigma_84th_error = sigma_quantile_84-sigma
    sigma_mean_quantiles = np.mean([sigma_16th_error,sigma_84th_error])
    
    # [mean err (mean of 16% & 84%), 16% err, 84% err]
    A_err = [A_mean_quantiles,A_16th_error,A_84th_error]
    sigma_err = [sigma_mean_quantiles,sigma_16th_error,sigma_84th_error]
    
    new_row = [ID, redshift, samples, emcee_wave, emcee_flux, emcee_flux_err, median_params, 
               A, A_err, mu, sigma, sigma_err, perc_16th_params, perc_84th_params]
        
    return new_row


def final_fits_Ha_RUBIES1(source, initial_fitting_output, ID, redshift, data_type, emission_line_label='Ha', diagnose=True):
    '''
    args:
        source (int): index of source within dataframe
        initial_fitting_output (dataframe): parameters obtain from initial fitting
        ID (int): source identification
        redshift (float): redshift of source
        data_type (str): type of spectral data
        emission_line_label (str): label for emission line
        diagnose (bool): choice to output fitting diagnostic plots

    returns: dataframe with final fitting parameters
    '''

    # inner functions --------------------------------------------------------
    def log_likelihood(theta, x, y, yerr, data_type):
        '''
        This is the likelihood function we are using for emcee to run
        
        This likelihood function is the maximum likelihood assuming gaussian errors.
        '''
        ################
        # The value we are trying to fit
        #A, mu, sigma, m, b = theta

        if data_type == 'GRATING':
            #Making the model of the emission line
            model = blended_Ha_line_model(x, *theta)

        else:
            #Making the model of the emission line
            model = single_gauss_line_model(x, *theta)
        
        #getting the log likelihood, this is similar to chi2
        lnL = -0.5 * np.nansum((y - model) ** 2 / yerr**2)
        
        return lnL

    def log_prior(theta, wave_center, Amp_max, data_type):
        '''
        The prior function to be used against the parameters to impose certain criteria for the fitting
        '''
        #the left most and right most the central wavelength can vary
        left_mu = wave_center - 20   # [A] # had to change these as well this is how much mu can vary
        right_mu = wave_center + 20  # [A] # had to change these as well this is how much mu can vary
        
        #min and max amplitude of the emission line
        min_A = 0
        max_A = Amp_max * 2
        
        # if using grating data, [NII] parameter
        if data_type == 'GRATING':
            sigma_window_left = 1.5 # [A] # had to change these for the input spectra these are left bounds for sigma
            sigma_window_right = 30 # [A] # had to change these for the input spectra these are right bounds for sigma
            
            #Theta values that goes into our Gaussian Model
            A, A1_NII, mu, sigma, m, b = theta
            
            if (0 < A < max_A) & (0 <= A1_NII < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right) & (b > 0):
                return 0.0
            else:
                return -np.inf

        # if using PRISM data, no [NII] parameter
        else:
            sigma_window_left = 30 # [A] # had to change these for the input spectra these are left bounds for sigma
            sigma_window_right = 200 # [A] # had to change these for the input spectra these are right bounds for sigma
            
            #Theta values that goes into our Gaussian Model
            A, mu, sigma, m, b = theta
            
            if (0 < A < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right) & (b > 0):
                return 0.0
            else:
                return -np.inf
        
    def log_probability(theta, x, y, yerr, first_wave, Amp_max, data_type):
        
        lp = log_prior(theta, first_wave, Amp_max, data_type)
        if not np.isfinite(lp):
            #print('Probability is infinite')
            return -np.inf
        
        prob = lp + log_likelihood(theta, x, y, yerr, data_type)

        #print(f'Prob:{prob:.3E}')
        return prob
    # ------------------------------------------------------------------------------------------------------


    if data_type == 'GRATING':
        label = 'Grating spectra'

        # initial fit with curve_fit()
        initial_fit = initial_fitting_output['ppots'][source]
        
        guess_A = initial_fit[0]
        guess_A_NII = initial_fit[1]
        guess_mu = initial_fit[2] 
        guess_sigma = initial_fit[3] 
        guess_m = initial_fit[4]
        guess_b = initial_fit[5]
        
        
        #making walkers so that we can use emcee to explore the parameter space
        #centered on the best results from minimization
        amp_jump = np.random.normal(loc = guess_A,            # centered on best A from minimization
                                    scale = guess_A/10,       # can wander 1/10 of the value of A
                                    size = 32).reshape(-1, 1) 
        
        amp_NII_jump = np.random.normal(loc = guess_A_NII,            # centered on best A from minimization
                                        scale = guess_A_NII/10,       #c an wander 1/10 of the value of A
                                        size = 32).reshape(-1, 1) 
        
        wavelength_jump = np.random.normal(loc = guess_mu,    # centered on best mu from minimization
                                           scale = 50,        # can wander +/- 0.005 microns 
                                           size = 32).reshape(-1, 1)
        
        
        sigma_jump = np.random.normal(loc = guess_sigma, scale = 20, size = 32).reshape(-1, 1)
        
        powerb = np.log10(np.abs(guess_b))
        b_jump = np.random.normal(loc = guess_b, scale = 1*10**powerb, size = 32).reshape(-1, 1)

        powerm = np.log10(np.abs(guess_m))
        m_jump = np.random.normal(loc = guess_m, scale = 1*10**powerm, size = 32).reshape(-1, 1)
        
        # #################
        # # Diagnostic plotting to see if the parameters were jumping to large values
        # # They should be concentrated near their best fit results values
        # #################
        # if diagnose == True:
        #     print('Checking the Walker Jumps')
        #     fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
            
        #     ax[0, 0].hist(amp_jump)
        #     ax[0, 0].set_xlabel('Amplitude')
    
        #     ax[0, 0].hist(amp_NII_jump)
        #     ax[0, 0].set_xlabel('NII Amplitude')
            
        #     ax[0, 1].hist(wavelength_jump)
        #     ax[0, 1].set_xlabel(r'$\mu$')
            
        #     ax[1, 0].hist(sigma_jump)
        #     ax[1, 0].set_xlabel(r'$\sigma$')
            
        #     ax[1, 1].hist(b_jump)
        #     ax[1, 1].set_xlabel('b')

        #     ax[1, 1].hist(m_jump)
        #     ax[1, 1].set_xlabel('m')
            
        #     plt.show()
        
        #stacking along the columns
        starting_walkers = np.hstack((amp_jump,
                                      amp_NII_jump,
                                      wavelength_jump, 
                                      sigma_jump, 
                                      m_jump, 
                                      b_jump))
    
        # emcee window
        emcee_flux = initial_fitting_output['flux_window'][source]
        emcee_wave = initial_fitting_output['wav_window'][source]
        emcee_flux_err = initial_fitting_output['flux_err_window'][source]
    
        #initializing walker positions
        pos = starting_walkers
        nwalkers, ndim = pos.shape
    
        #initializing sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(emcee_wave, emcee_flux, emcee_flux_err, guess_mu, guess_A, data_type)
                                       )
        #running it 
        '''
        added skip_initial_state_check=True to fix the following error
        "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance"
        '''
        sampler.run_mcmc(pos, 3000, progress=False, skip_initial_state_check=True)
    
        #getting values back
        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(flat=True)
        LnL_chain = sampler.flatlnprobability
        burn_in = 2000 
        
        emcee_df = pd.DataFrame()
        emcee_df['A'] = flat_samples[burn_in:, 0]
        emcee_df['A_NII'] = flat_samples[burn_in:, 1]
        emcee_df['mu'] = flat_samples[burn_in:, 2]
        emcee_df['sigma'] = flat_samples[burn_in:, 3]
        emcee_df['m'] = flat_samples[burn_in:, 4]
        emcee_df['b'] = flat_samples[burn_in:, 5]
        emcee_df['LnL'] = LnL_chain[burn_in:]
        
        emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
        
        fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi) # M: calculating area of Gaussian
        emcee_df['Fluxes'] = fluxes_emcee
        
        # if diagnose == True:
            
        #     print('checking chain')
        #     fig, axes = plt.subplots(6, figsize=(12, 10), sharex=True)
        #     samples = sampler.get_chain()
        #     labels = ['A','A_NII','mu','sigma','m', 'b']
        #     for i in range(ndim):
        #         ax = axes[i]
        #         ax.plot(samples[:, :, i], "k", alpha=0.3)
        #         ax.set_xlim(0, len(samples))
        #         ax.set_ylabel(labels[i])
        #         ax.yaxis.set_label_coords(-0.1, 0.5)
            
        #     axes[-1].set_xlabel("step number");
        
        if diagnose == True:
            xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 500)
            gauss_16th_perc = blended_Ha_line_model(xarr, *emcee_df.quantile(q = 0.16).values[:-2])
            gauss_84th_perc = blended_Ha_line_model(xarr, *emcee_df.quantile(q = 0.84).values[:-2])
            
            plt.figure()
            plt.title('ID: '+str(ID)+' line: '+str(emission_line_label)+' z='+str(redshift) ,fontsize=15,)
            plt.step(emcee_wave, emcee_flux, color='grey', label = label, where='mid')
            #plt.scatter(emcee_wave, emcee_spec, color = 'black')
            plt.plot(xarr, blended_Ha_line_model(xarr, *emcee_df.quantile(q = 0.5).values[:-2]), color='#407899', label = 'emcee model')
            plt.fill_between(xarr, gauss_16th_perc, gauss_84th_perc, color='#407899', alpha=0.3)
            plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
            plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)
            plt.legend()
            plt.show()
    
        # median parameter values
        median_params = emcee_df.quantile(q=0.5).values[:-2]
        perc_16th_params = emcee_df.quantile(q=0.16).values[:-2]
        perc_84th_params = emcee_df.quantile(q=0.84).values[:-2]
        A = median_params[0]
        A_NII = median_params[1]
        mu = median_params[2]
        sigma = median_params[3]
        
        # gathering errors
        # A
        A_quantile_16 = emcee_df.quantile(q=0.16).values[0]
        A_qunatile_84 = emcee_df.quantile(q=0.84).values[0]
        A_16th_error = A-A_quantile_16
        A_84th_error = A_qunatile_84-A
        A_mean_quantiles = np.mean([A_16th_error,A_84th_error])

        # A_NII
        A_NII_quantile_16 = emcee_df.quantile(q=0.16).values[1]
        A_NII_qunatile_84 = emcee_df.quantile(q=0.84).values[1]
        A_NII_16th_error = A_NII-A_NII_quantile_16
        A_NII_84th_error = A_NII_qunatile_84-A_NII
        A_NII_mean_quantiles = np.mean([A_NII_16th_error,A_NII_84th_error])
    
        # sigma
        sigma_quantile_16 = emcee_df.quantile(q=0.16).values[3]
        sigma_qunatile_84 = emcee_df.quantile(q=0.84).values[3]
        sigma_16th_error = sigma-sigma_quantile_16
        sigma_84th_error = sigma_qunatile_84-sigma
        sigma_mean_quantiles = np.mean([sigma_16th_error,sigma_84th_error])
        
        # [mean err (mean of 16% & 84%), 16% err, 84% err]
        A_err = [A_mean_quantiles,A_16th_error,A_84th_error]
        A_NII_err = [A_NII_mean_quantiles,A_NII_16th_error,A_NII_84th_error]
        sigma_err = [sigma_mean_quantiles,sigma_16th_error,sigma_84th_error]
        
        new_row = [ID, redshift, emcee_df, emcee_wave, emcee_flux, emcee_flux_err, median_params, 
                   A, A_err, mu, sigma, sigma_err, perc_16th_params, perc_84th_params,A_NII, A_NII_err]

    # if using PRISM data, no [NII] parameter
    else:
        label = 'PRISM spectra'
        
        # initial fit with curve_fit()
        initial_fit = initial_fitting_output['ppots'][source]
        
        guess_A = initial_fit[0]
        guess_mu = initial_fit[1] 
        guess_sigma = initial_fit[2] 
        guess_m = initial_fit[3]
        guess_b = initial_fit[4]
        
        
        #making walkers so that we can use emcee to explore the parameter space
        #centered on the best results from minimization
        amp_jump = np.random.normal(loc = guess_A,            # centered on best A from minimization
                                    scale = guess_A/10,       # can wander 1/10 of the value of A
                                    size = 32).reshape(-1, 1) 
        
        wavelength_jump = np.random.normal(loc = guess_mu,    # centered on best mu from minimization
                                           scale = 50,        # can wander +/- 0.005 microns 
                                           size = 32).reshape(-1, 1)
        
        
        sigma_jump = np.random.normal(loc = guess_sigma, scale = 20, size = 32).reshape(-1, 1)
        
        powerb = np.log10(np.abs(guess_b))
        b_jump = np.random.normal(loc = guess_b, scale = 1*10**powerb, size = 32).reshape(-1, 1)

        powerm = np.log10(np.abs(guess_m))
        m_jump = np.random.normal(loc = guess_m, scale = 1*10**powerm, size = 32).reshape(-1, 1)
        
        # #################
        # # Diagnostic plotting to see if the parameters were jumping to large values
        # # They should be concentrated near their best fit results values
        # #################
        # if diagnose == True:
        #     print('Checking the Walker Jumps')
        #     fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
            
        #     ax[0, 0].hist(amp_jump)
        #     ax[0, 0].set_xlabel('Amplitude')
            
        #     ax[0, 1].hist(wavelength_jump)
        #     ax[0, 1].set_xlabel(r'$\mu$')
            
        #     ax[1, 0].hist(sigma_jump)
        #     ax[1, 0].set_xlabel(r'$\sigma$')
            
        #     ax[1, 1].hist(m_jump)
        #     ax[1, 1].set_xlabel('m')

        #     ax[1, 1].hist(b_jump)
        #     ax[1, 1].set_xlabel('b')
            
        #     plt.show()
        
        #stacking along the columns
        starting_walkers = np.hstack((amp_jump,
                                      wavelength_jump, 
                                      sigma_jump, 
                                      m_jump, 
                                      b_jump))
    
        # emcee window
        emcee_flux = initial_fitting_output['flux_window'][source]
        emcee_wave = initial_fitting_output['wav_window'][source]
        emcee_flux_err = initial_fitting_output['flux_err_window'][source]
    
        #initializing walker positions
        pos = starting_walkers
        nwalkers, ndim = pos.shape
    
        #initializing sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(emcee_wave, emcee_flux, emcee_flux_err, guess_mu, guess_A, data_type)
                                       )
        '''
        added skip_initial_state_check=True to fix the following error
        "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance"
        '''
        sampler.run_mcmc(pos, 3000, progress=False, skip_initial_state_check=True)
    
        #getting values back
        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(flat=True)
        LnL_chain = sampler.flatlnprobability
        burn_in = 2000 
        
        emcee_df = pd.DataFrame()
        emcee_df['A'] = flat_samples[burn_in:, 0]
        emcee_df['mu'] = flat_samples[burn_in:, 1]
        emcee_df['sigma'] = flat_samples[burn_in:, 2]
        emcee_df['m'] = flat_samples[burn_in:, 3]
        emcee_df['b'] = flat_samples[burn_in:, 4]
        emcee_df['LnL'] = LnL_chain[burn_in:]
        
        emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
        
        fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi) # M: calculating area of Gaussian
        emcee_df['Fluxes'] = fluxes_emcee
        
        # if diagnose == True:
            
        #     print('checking chain')
        #     fig, axes = plt.subplots(5, figsize=(12, 10), sharex=True)
        #     samples = sampler.get_chain()
        #     labels = ['A','mu','sigma','m', 'b']
        #     for i in range(ndim):
        #         ax = axes[i]
        #         ax.plot(samples[:, :, i], "k", alpha=0.3)
        #         ax.set_xlim(0, len(samples))
        #         ax.set_ylabel(labels[i])
        #         ax.yaxis.set_label_coords(-0.1, 0.5)
            
        #     axes[-1].set_xlabel("step number");

        # median parameter values
        median_params = emcee_df.quantile(q=0.5).values[:-2]
        perc_16th_params = emcee_df.quantile(q=0.16).values[:-2]
        perc_84th_params = emcee_df.quantile(q=0.84).values[:-2]
        A = median_params[0]
        mu = median_params[1]
        sigma = median_params[2]
        
        if diagnose == True:
            xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 500)
            gauss_16th_perc = single_gauss_line_model(xarr, *perc_16th_params)
            gauss_84th_perc = single_gauss_line_model(xarr, *perc_84th_params)
            
            plt.figure()
            plt.title('ID: '+str(ID)+' line: '+str(emission_line_label)+' z='+str(redshift), fontsize=15)
            plt.step(emcee_wave, emcee_flux, color='grey', label = label, where='mid')
            #plt.scatter(emcee_wave, emcee_spec, color = 'black')
            plt.plot(xarr, single_gauss_line_model(xarr, *median_params), color='#407899', label = 'emcee model')
            plt.fill_between(xarr, gauss_16th_perc, gauss_84th_perc, color='#407899', alpha=0.3)
            plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
            plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)
            plt.legend()
            plt.show()
        
        # gathering errors
        # A
        A_quantile_16 = emcee_df.quantile(q=0.16).values[0]
        A_qunatile_84 = emcee_df.quantile(q=0.84).values[0]
        A_16th_error = A-A_quantile_16
        A_84th_error = A_qunatile_84-A
        A_mean_quantiles = np.mean([A_16th_error,A_84th_error])
    
        # sigma
        sigma_quantile_16 = emcee_df.quantile(q=0.16).values[2]
        sigma_qunatile_84 = emcee_df.quantile(q=0.84).values[2]
        sigma_16th_error = sigma-sigma_quantile_16
        sigma_84th_error = sigma_qunatile_84-sigma
        sigma_mean_quantiles = np.mean([sigma_16th_error,sigma_84th_error])
        
        # [mean err (mean of 16% & 84%), 16% err, 84% err]
        A_err = [A_mean_quantiles,A_16th_error,A_84th_error]
        sigma_err = [sigma_mean_quantiles,sigma_16th_error,sigma_84th_error]
        
        new_row = [ID, redshift, samples, emcee_wave, emcee_flux, emcee_flux_err, median_params, 
                   A, A_err, mu, sigma, sigma_err, perc_16th_params, perc_84th_params, None,[None] *3]
        
    return new_row


def final_fits_Hb_RUBIES(source, initial_fitting_output, ID, redshift, data_type, emission_line_label='Hb', diagnose=True):
    '''
    args:
        source (int): index of source within dataframe
        initial_fitting_output (dataframe): parameters obtain from initial fitting
        ID (int): source identification
        redshift (float): redshift of source
        data_type (str): type of spectral data
        emission_line_label (str): label for emission line
        diagnose (bool): choice to output fitting diagnostic plots

    returns: dataframe with final fitting parameters
    '''

    # inner functions --------------------------------------------------------
    def log_likelihood(theta, x, y, yerr, redshift, data_type):
        '''
        This is the likelihood function we are using for emcee to run
        
        This likelihood function is the maximum likelihood assuming gaussian errors.
        '''
        ################
        # The value we are trying to fit
        #A, mu, sigma, m, b = theta

        # if using PRISM data, [OIII] doublet blends with H-beta emission
        if (data_type != 'GRATING') and (redshift <= 7.1):
            #Making the model of the emission line
            model = blended_Hb_line_model(x, *theta)

        else:
            #Making the model of the emission line
            model = single_gauss_line_model(x, *theta)
        
        #getting the log likelihood, this is similar to chi2
        lnL = -0.5 * np.nansum((y - model) ** 2 / yerr**2)
        
        return lnL

    def log_prior(theta, wave_center, Amp_max, redshift, data_type):
        '''
        The prior function to be used against the parameters to impose certain criteria for the fitting
        '''
        #the left most and right most the central wavelength can vary
        left_mu = wave_center - 20   # [A] # had to change these as well this is how much mu can vary
        right_mu = wave_center + 20  # [A] # had to change these as well this is how much mu can vary
        
        #min and max amplitude of the emission line
        min_A = 0
        max_A = Amp_max * 2
        
        # if not using grating data, [OIII] parameter
        if (data_type != 'GRATING') and (redshift <= 7.1):

            sigma_window_left = 30 # [A] # had to change these for the input spectra these are left bounds for sigma
            sigma_window_right = 200 # [A] # had to change these for the input spectra these are right bounds for sigma
            
            #Theta values that goes into our Gaussian Model
            A, A1_OIII, mu, sigma, m, b = theta
            
            if (0 < A < max_A) & (0 <= A1_OIII < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right) & (b > 0):
                return 0.0
            else:
                return -np.inf

        # if using grating data or high-z PRISM, no [OIII] parameter
        else:

            sigma_window_left = 1.5 # [A] # had to change these for the input spectra these are left bounds for sigma
            sigma_window_right = 30 # [A] # had to change these for the input spectra these are right bounds for sigma
            
            #Theta values that goes into our Gaussian Model
            A, mu, sigma, m, b = theta
            
            if (0 < A < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right) & (b > 0):
                return 0.0
            else:
                return -np.inf
        
    def log_probability(theta, x, y, yerr, first_wave, Amp_max, redshift, data_type):
        
        lp = log_prior(theta, first_wave, Amp_max, redshift, data_type)
        if not np.isfinite(lp):
            #print('Probability is infinite')
            return -np.inf
        
        prob = lp + log_likelihood(theta, x, y, yerr, redshift, data_type)

        #print(f'Prob:{prob:.3E}')
        return prob
    # -----------------------------------------------------------------------------------------

    
    # if using PRISM data, [OIII] parameter
    if data_type != 'GRATING':
        label = 'PRISM spectra'

        # initial fit with curve_fit()
        initial_fit = initial_fitting_output['ppots'][source]
        
        guess_A = initial_fit[0]
        guess_A_OIII = initial_fit[1]
        guess_mu = initial_fit[2] 
        guess_sigma = initial_fit[3] 
        guess_m = initial_fit[4]
        guess_b = initial_fit[5]
        
        
        #making walkers so that we can use emcee to explore the parameter space
        #centered on the best results from minimization
        amp_jump = np.random.normal(loc = guess_A,            # centered on best A from minimization
                                    scale = guess_A/10,       # can wander 1/10 of the value of A
                                    size = 32).reshape(-1, 1) 
        
        amp_OIII_jump = np.random.normal(loc = guess_A_OIII,            # centered on best A from minimization
                                        scale = guess_A_OIII/10,       #c an wander 1/10 of the value of A
                                        size = 32).reshape(-1, 1) 
        
        wavelength_jump = np.random.normal(loc = guess_mu,    # centered on best mu from minimization
                                           scale = 50,        # can wander +/- 0.005 microns 
                                           size = 32).reshape(-1, 1)
        
        
        sigma_jump = np.random.normal(loc = guess_sigma, scale = 5, size = 32).reshape(-1, 1)
        
        powerb = np.log10(np.abs(guess_b))
        b_jump = np.random.normal(loc = guess_b, scale = 1*10**powerb, size = 32).reshape(-1, 1)

        powerm = np.log10(np.abs(guess_m))
        m_jump = np.random.normal(loc = guess_m, scale = 1*10**powerm, size = 32).reshape(-1, 1)
        
        # #################
        # # Diagnostic plotting to see if the parameters were jumping to large values
        # # They should be concentrated near their best fit results values
        # #################
        # if diagnose == True:
        #     print('Checking the Walker Jumps')
        #     fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
            
        #     ax[0, 0].hist(amp_jump)
        #     ax[0, 0].set_xlabel('Amplitude')
    
        #     ax[0, 0].hist(amp_OIII_jump)
        #     ax[0, 0].set_xlabel('OIII Amplitude')
            
        #     ax[0, 1].hist(wavelength_jump)
        #     ax[0, 1].set_xlabel(r'$\mu$')
            
        #     ax[1, 0].hist(sigma_jump)
        #     ax[1, 0].set_xlabel(r'$\sigma$')
            
        #     ax[1, 1].hist(b_jump)
        #     ax[1, 1].set_xlabel('b')
            
        #     plt.show()
        
        #stacking along the columns
        starting_walkers = np.hstack((amp_jump,
                                      amp_OIII_jump,
                                      wavelength_jump, 
                                      sigma_jump, 
                                      m_jump, 
                                      b_jump))
    
        # emcee window
        emcee_flux = initial_fitting_output['flux_window'][source]
        emcee_wave = initial_fitting_output['wav_window'][source]
        emcee_flux_err = initial_fitting_output['flux_err_window'][source]
    
        #initializing walker positions
        pos = starting_walkers
        nwalkers, ndim = pos.shape
    
        #initializing sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(emcee_wave, emcee_flux, emcee_flux_err, guess_mu, guess_A, redshift, data_type)
                                       )
        '''
        added skip_initial_state_check=True to fix the following error
        "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance"
        '''
        sampler.run_mcmc(pos, 3000, progress=False, skip_initial_state_check=True)
    
        #getting values back
        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(flat=True)
        LnL_chain = sampler.flatlnprobability
        burn_in = 2000 
        
        emcee_df = pd.DataFrame()
        emcee_df['A'] = flat_samples[burn_in:, 0]
        emcee_df['A_OIII'] = flat_samples[burn_in:, 1]
        emcee_df['mu'] = flat_samples[burn_in:, 2]
        emcee_df['sigma'] = flat_samples[burn_in:, 3]
        emcee_df['m'] = flat_samples[burn_in:, 4]
        emcee_df['b'] = flat_samples[burn_in:, 5]
        emcee_df['LnL'] = LnL_chain[burn_in:]
        
        emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
        
        fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi) # M: calculating area of Gaussian
        emcee_df['Fluxes'] = fluxes_emcee
        
        # if diagnose == True:

        #     print('checking chain')
        #     fig, axes = plt.subplots(6, figsize=(12, 10), sharex=True)
        #     samples = sampler.get_chain()
        #     labels = ['A','A_OIII','mu','sigma','m', 'b']
        #     for i in range(ndim):
        #         ax = axes[i]
        #         ax.plot(samples[:, :, i], "k", alpha=0.3)
        #         ax.set_xlim(0, len(samples))
        #         ax.set_ylabel(labels[i])
        #         ax.yaxis.set_label_coords(-0.1, 0.5)
            
        #     axes[-1].set_xlabel("step number");

        # median parameter values
        median_params = emcee_df.quantile(q=0.5).values[:-2]
        perc_16th_params = emcee_df.quantile(q=0.16).values[:-2]
        perc_84th_params = emcee_df.quantile(q=0.84).values[:-2]
        A = median_params[0]
        mu = median_params[2]
        sigma = median_params[3]
        
        if diagnose == True:
            xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 500)
            gauss_16th_perc = blended_Hb_line_model(xarr, *perc_16th_params)
            gauss_84th_perc = blended_Hb_line_model(xarr, *perc_84th_params)
            
            plt.figure()
            plt.title('ID: '+str(ID)+' line: '+str(emission_line_label)+' z='+str(redshift), fontsize=15)
            plt.step(emcee_wave, emcee_flux, color='grey', label = label, where='mid')
            #plt.scatter(emcee_wave, emcee_spec, color = 'black')
            plt.plot(xarr, blended_Hb_line_model(xarr, *median_params), color='#407899', label = 'emcee model')
            plt.fill_between(xarr, gauss_16th_perc, gauss_84th_perc, color='#407899', alpha=0.3)
            plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
            plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)
            plt.legend()
            plt.show()
        
        # gathering errors
        # A
        A_quantile_16 = emcee_df.quantile(q=0.16).values[0]
        A_qunatile_84 = emcee_df.quantile(q=0.84).values[0]
        A_16th_error = A-A_quantile_16
        A_84th_error = A_qunatile_84-A
        A_mean_quantiles = np.mean([A_16th_error,A_84th_error])
    
        # sigma
        sigma_quantile_16 = emcee_df.quantile(q=0.16).values[3]
        sigma_qunatile_84 = emcee_df.quantile(q=0.84).values[3]
        sigma_16th_error = sigma-sigma_quantile_16
        sigma_84th_error = sigma_qunatile_84-sigma
        sigma_mean_quantiles = np.mean([sigma_16th_error,sigma_84th_error])
        
        # [mean err (mean of 16% & 84%), 16% err, 84% err]
        A_err = [A_mean_quantiles,A_16th_error,A_84th_error]
        sigma_err = [sigma_mean_quantiles,sigma_16th_error,sigma_84th_error]
        
        new_row = [ID, redshift, emcee_df, emcee_wave, emcee_flux, emcee_flux_err, median_params, 
                   A, A_err, mu, sigma, sigma_err, perc_16th_params, perc_84th_params]

    # if using grating data, no [OIII] parameter
    else:

        label = 'Grating spectra'  
        
        # initial fit with curve_fit()
        initial_fit = initial_fitting_output['ppots'][source]
        
        guess_A = initial_fit[0]
        guess_mu = initial_fit[1] 
        guess_sigma = initial_fit[2] 
        guess_m = initial_fit[3] 
        guess_b = initial_fit[4]
        
        
        #making walkers so that we can use emcee to explore the parameter space
        #centered on the best results from minimization
        amp_jump = np.random.normal(loc = guess_A,            # centered on best A from minimization
                                    scale = guess_A/10,       # can wander 1/10 of the value of A
                                    size = 32).reshape(-1, 1) 
        
        wavelength_jump = np.random.normal(loc = guess_mu,    # centered on best mu from minimization
                                           scale = 50,        # can wander +/- 0.005 microns 
                                           size = 32).reshape(-1, 1)
        
        
        sigma_jump = np.random.normal(loc = guess_sigma, scale = 20, size = 32).reshape(-1, 1)
        
        powerb = np.log10(np.abs(guess_b))
        b_jump = np.random.normal(loc = guess_b, scale = 1*10**powerb, size = 32).reshape(-1, 1)

        powerm = np.log10(np.abs(guess_m))
        m_jump = np.random.normal(loc = guess_m, scale = 1*10**powerm, size = 32).reshape(-1, 1)
        
        # #################
        # # Diagnostic plotting to see if the parameters were jumping to large values
        # # They should be concentrated near their best fit results values
        # #################
        # if diagnose == True:
        #     print('Checking the Walker Jumps')
        #     fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
            
        #     ax[0, 0].hist(amp_jump)
        #     ax[0, 0].set_xlabel('Amplitude')
            
        #     ax[0, 1].hist(wavelength_jump)
        #     ax[0, 1].set_xlabel(r'$\mu$')
            
        #     ax[1, 0].hist(sigma_jump)
        #     ax[1, 0].set_xlabel(r'$\sigma$')
            
        #     ax[1, 1].hist(b_jump)
        #     ax[1, 1].set_xlabel('b')
            
        #     plt.show()
        
        #stacking along the columns
        starting_walkers = np.hstack((amp_jump,
                                      wavelength_jump, 
                                      sigma_jump, 
                                      m_jump, 
                                      b_jump))
    
        # emcee window
        emcee_flux = initial_fitting_output['flux_window'][source]
        emcee_wave = initial_fitting_output['wav_window'][source]
        emcee_flux_err = initial_fitting_output['flux_err_window'][source]
    
        #initializing walker positions
        pos = starting_walkers
        nwalkers, ndim = pos.shape
    
        #initializing sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(emcee_wave, emcee_flux, emcee_flux_err, guess_mu, guess_A, redshift, data_type)
                                       )
        '''
        added skip_initial_state_check=True to fix the following error
        "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance"
        '''
        sampler.run_mcmc(pos, 3000, progress=False, skip_initial_state_check=True)
    
        #getting values back
        samples = sampler.get_chain()
        flat_samples = sampler.get_chain(flat=True)
        LnL_chain = sampler.flatlnprobability
        burn_in = 2000 
        
        emcee_df = pd.DataFrame()
        emcee_df['A'] = flat_samples[burn_in:, 0]
        emcee_df['mu'] = flat_samples[burn_in:, 1]
        emcee_df['sigma'] = flat_samples[burn_in:, 2]
        emcee_df['m'] = flat_samples[burn_in:, 3]
        emcee_df['b'] = flat_samples[burn_in:, 4]
        emcee_df['LnL'] = LnL_chain[burn_in:]
        
        emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
        
        fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi) # M: calculating area of Gaussian
        emcee_df['Fluxes'] = fluxes_emcee
        
        # if diagnose == True:
            
        #     print('checking chain')
        #     fig, axes = plt.subplots(5, figsize=(12, 10), sharex=True)
        #     samples = sampler.get_chain()
        #     labels = ['A','mu','sigma','m', 'b']
        #     for i in range(ndim):
        #         ax = axes[i]
        #         ax.plot(samples[:, :, i], "k", alpha=0.3)
        #         ax.set_xlim(0, len(samples))
        #         ax.set_ylabel(labels[i])
        #         ax.yaxis.set_label_coords(-0.1, 0.5)
            
        #     axes[-1].set_xlabel("step number");

        # median parameter values
        median_params = emcee_df.quantile(q=0.5).values[:-2]
        perc_16th_params = emcee_df.quantile(q=0.16).values[:-2]
        perc_84th_params = emcee_df.quantile(q=0.84).values[:-2]
        A = median_params[0]
        mu = median_params[1]
        sigma = median_params[2]

        if diagnose == True:
            xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 500)
            gauss_16th_perc = single_gauss_line_model(xarr, *perc_16th_params)
            gauss_84th_perc = single_gauss_line_model(xarr, *perc_84th_params)
            
            plt.figure()
            plt.title('ID: '+str(ID)+' line: '+str(emission_line_label)+' z='+str(redshift), fontsize=15)
            plt.step(emcee_wave, emcee_flux, color='grey', label = label, where='mid')
            #plt.scatter(emcee_wave, emcee_spec, color = 'black')
            plt.plot(xarr, single_gauss_line_model(xarr, *median_params), color='#407899', label = 'emcee model')
            plt.fill_between(xarr, gauss_16th_perc, gauss_84th_perc, color='#407899', alpha=0.3)
            plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
            plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)
            plt.legend()
            plt.show()
        
        # gathering errors
        # A
        A_quantile_16 = emcee_df.quantile(q=0.16).values[0]
        A_qunatile_84 = emcee_df.quantile(q=0.84).values[0]
        A_16th_error = A-A_quantile_16
        A_84th_error = A_qunatile_84-A
        A_mean_quantiles = np.mean([A_16th_error,A_84th_error])
    
        # sigma
        sigma_quantile_16 = emcee_df.quantile(q=0.16).values[2]
        sigma_qunatile_84 = emcee_df.quantile(q=0.84).values[2]
        sigma_16th_error = sigma-sigma_quantile_16
        sigma_84th_error = sigma_qunatile_84-sigma
        sigma_mean_quantiles = np.mean([sigma_16th_error,sigma_84th_error])
        
        # [mean err (mean of 16% & 84%), 16% err, 84% err]
        A_err = [A_mean_quantiles,A_16th_error,A_84th_error]
        sigma_err = [sigma_mean_quantiles,sigma_16th_error,sigma_84th_error]
        
        new_row = [ID, redshift, samples, emcee_wave, emcee_flux, emcee_flux_err, median_params, 
                   A, A_err, mu, sigma, sigma_err, perc_16th_params, perc_84th_params]
        
    return new_row


def final_fits_Ha_RUBIES2(source, initial_fitting_output, ID, redshift, data_type, emission_line_label='Ha', diagnose=True):
    '''
    args:
        source (int): index of source within dataframe
        initial_fitting_output (dataframe): parameters obtain from initial fitting
        ID (int): source identification
        redshift (float): redshift of source
        data_type (str): type of spectral data
        emission_line_label (str): label for emission line
        diagnose (bool): choice to output fitting diagnostic plots

    returns: dataframe with final fitting parameters
    '''

    # inner functions --------------------------------------------------------
    def log_likelihood(theta, x, y, yerr):
        '''
        This is the likelihood function we are using for emcee to run
        
        This likelihood function is the maximum likelihood assuming gaussian errors.
        '''
        ################
        # The value we are trying to fit
        #A, mu, sigma, m, b = theta

        #Making the model of the emission line
        model = blended_Ha_line_model(x, *theta)

        #getting the log likelihood, this is similar to chi2
        lnL = -0.5 * np.nansum((y - model) ** 2 / yerr**2)
        
        return lnL

    def log_prior(theta, wave_center, Amp_max):
        '''
        The prior function to be used against the parameters to impose certain criteria for the fitting
        '''
        #the left most and right most the central wavelength can vary
        left_mu = wave_center - 20   # [A] # had to change these as well this is how much mu can vary
        right_mu = wave_center + 20  # [A] # had to change these as well this is how much mu can vary
        
        #min and max amplitude of the emission line
        min_A = 0
        max_A = Amp_max * 2
        
        sigma_window_left = 1.5 # [A] # had to change these for the input spectra these are left bounds for sigma
        sigma_window_right = 30 # [A] # had to change these for the input spectra these are right bounds for sigma
        
        #Theta values that goes into our Gaussian Model
        A, A1_NII, mu, sigma, m, b = theta
            
        if (0 < A < max_A) & (0 <= A1_NII < max_A) & (left_mu <= mu <= right_mu) & (sigma_window_left <= sigma < sigma_window_right) & (b > 0):
            return 0.0
        else:
            return -np.inf
        
    def log_probability(theta, x, y, yerr, first_wave, Amp_max):
        
        lp = log_prior(theta, first_wave, Amp_max)
        if not np.isfinite(lp):
            #print('Probability is infinite')
            return -np.inf
        
        prob = lp + log_likelihood(theta, x, y, yerr)

        #print(f'Prob:{prob:.3E}')
        return prob
    # --------------------------------------------------------------------------------


    color = '#62B7BC'
    label = 'Grating spectra'

    # initial fit with curve_fit()
    initial_fit = initial_fitting_output['ppots'][source]
    
    guess_A = initial_fit[0]
    guess_A_NII = initial_fit[1]
    guess_mu = initial_fit[2] 
    guess_sigma = initial_fit[3] 
    guess_m = initial_fit[4]
    guess_b = initial_fit[5]
    
    #making walkers so that we can use emcee to explore the parameter space
    #centered on the best results from minimization
    amp_jump = np.random.normal(loc = guess_A,            # centered on best A from minimization
                                scale = guess_A/10,       # can wander 1/10 of the value of A
                                size = 32).reshape(-1, 1) 
    
    amp_NII_jump = np.random.normal(loc = guess_A_NII,            # centered on best A from minimization
                                    scale = guess_A_NII/10,       #c an wander 1/10 of the value of A
                                    size = 32).reshape(-1, 1) 
    
    wavelength_jump = np.random.normal(loc = guess_mu,    # centered on best mu from minimization
                                       scale = 50,        # can wander +/- 0.005 microns 
                                       size = 32).reshape(-1, 1)
    
    
    sigma_jump = np.random.normal(loc = guess_sigma, scale = 20, size = 32).reshape(-1, 1)
    
    powerb = np.log10(np.abs(guess_b))
    b_jump = np.random.normal(loc = guess_b, scale = 1*10**powerb, size = 32).reshape(-1, 1)

    powerm = np.log10(np.abs(guess_m))
    m_jump = np.random.normal(loc = guess_m, scale = 1*10**powerm, size = 32).reshape(-1, 1)
    
    # #################
    # # Diagnostic plotting to see if the parameters were jumping to large values
    # # They should be concentrated near their best fit results values
    # #################
    # if diagnose == True:
    #     print('Checking the Walker Jumps')
    #     fig, ax = plt.subplots(nrows = 2, ncols = 2, constrained_layout = True)
        
    #     ax[0, 0].hist(amp_jump)
    #     ax[0, 0].set_xlabel('Amplitude')

    #     ax[0, 0].hist(amp_NII_jump)
    #     ax[0, 0].set_xlabel('NII Amplitude')
        
    #     ax[0, 1].hist(wavelength_jump)
    #     ax[0, 1].set_xlabel(r'$\mu$')
        
    #     ax[1, 0].hist(sigma_jump)
    #     ax[1, 0].set_xlabel(r'$\sigma$')
        
    #     ax[1, 1].hist(b_jump)
    #     ax[1, 1].set_xlabel('b')

    #     ax[1, 1].hist(m_jump)
    #     ax[1, 1].set_xlabel('m')
        
    #     plt.show()
    
    #stacking along the columns
    starting_walkers = np.hstack((amp_jump,
                                  amp_NII_jump,
                                  wavelength_jump, 
                                  sigma_jump, 
                                  m_jump, 
                                  b_jump))

    # emcee window
    emcee_flux = initial_fitting_output['flux_window'][source]
    emcee_wave = initial_fitting_output['wav_window'][source]
    emcee_flux_err = initial_fitting_output['flux_err_window'][source]

    #initializing walker positions
    pos = starting_walkers
    nwalkers, ndim = pos.shape

    #initializing sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(emcee_wave, emcee_flux, emcee_flux_err, guess_mu, guess_A)
                                   )
    '''
    added skip_initial_state_check=True to fix the following error
    "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance"
    '''
    sampler.run_mcmc(pos, 3000, progress=False, skip_initial_state_check=True)

    #getting values back
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(flat=True)
    LnL_chain = sampler.flatlnprobability
    burn_in = 2000 
    
    emcee_df = pd.DataFrame()
    emcee_df['A'] = flat_samples[burn_in:, 0]
    emcee_df['A_NII'] = flat_samples[burn_in:, 1]
    emcee_df['mu'] = flat_samples[burn_in:, 2]
    emcee_df['sigma'] = flat_samples[burn_in:, 3]
    emcee_df['m'] = flat_samples[burn_in:, 4]
    emcee_df['b'] = flat_samples[burn_in:, 5]
    emcee_df['LnL'] = LnL_chain[burn_in:]
    
    emcee_df = emcee_df[np.isfinite(emcee_df.LnL.values)]
    
    fluxes_emcee = emcee_df['A'] * emcee_df['sigma'] * np.sqrt(2 * np.pi) # M: calculating area of Gaussian
    emcee_df['Fluxes'] = fluxes_emcee
    
    # median parameter values
    median_params = emcee_df.quantile(q=0.5).values[:-2]
    perc_16th_params = emcee_df.quantile(q=0.16).values[:-2]
    perc_84th_params = emcee_df.quantile(q=0.84).values[:-2]
    A = median_params[0]
    A_NII = median_params[1]
    mu = median_params[2]
    sigma = median_params[3]

    if diagnose == True:
        xarr = np.linspace(emcee_wave[0], emcee_wave[-1], 500)
        gauss_16th_perc = blended_Ha_line_model(xarr, *perc_16th_params)
        gauss_84th_perc = blended_Ha_line_model(xarr, *perc_84th_params)
        
        plt.figure()
        plt.title('ID: '+str(ID)+' line: '+str(emission_line_label)+' z='+str(redshift), fontsize=15)
        plt.step(emcee_wave, emcee_flux, color='grey', label = label, where='mid')
        #plt.scatter(emcee_wave, emcee_spec, color = 'black')
        plt.plot(xarr, blended_Ha_line_model(xarr, *median_params), color='#407899', label = 'emcee model')
        plt.fill_between(xarr, gauss_16th_perc, gauss_84th_perc, color='#407899', alpha=0.3)
        plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
        plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)
        plt.legend()
        plt.show()
    
    # gathering errors
    # A
    A_quantile_16 = emcee_df.quantile(q=0.16).values[0]
    A_qunatile_84 = emcee_df.quantile(q=0.84).values[0]
    A_16th_error = A-A_quantile_16
    A_84th_error = A_qunatile_84-A
    A_mean_quantiles = np.mean([A_16th_error,A_84th_error])

    # A_NII
    A_NII_quantile_16 = emcee_df.quantile(q=0.16).values[1]
    A_NII_qunatile_84 = emcee_df.quantile(q=0.84).values[1]
    A_NII_16th_error = A_NII-A_NII_quantile_16
    A_NII_84th_error = A_NII_qunatile_84-A_NII
    A_NII_mean_quantiles = np.mean([A_NII_16th_error,A_NII_84th_error])

    # sigma
    sigma_quantile_16 = emcee_df.quantile(q=0.16).values[3]
    sigma_qunatile_84 = emcee_df.quantile(q=0.84).values[3]
    sigma_16th_error = sigma-sigma_quantile_16
    sigma_84th_error = sigma_qunatile_84-sigma
    sigma_mean_quantiles = np.mean([sigma_16th_error,sigma_84th_error])
    
    # [mean err (mean of 16% & 84%), 16% err, 84% err]
    A_err = [A_mean_quantiles,A_16th_error,A_84th_error]
    A_NII_err = [A_NII_mean_quantiles,A_NII_16th_error,A_NII_84th_error]
    sigma_err = [sigma_mean_quantiles,sigma_16th_error,sigma_84th_error]
    
    new_row = [ID, redshift, emcee_df, emcee_wave, emcee_flux, emcee_flux_err, median_params, A, A_err, mu, sigma, sigma_err, perc_16th_params, perc_84th_params,A_NII, A_NII_err]

    return new_row
