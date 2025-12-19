import numpy as np
import matplotlib.pyplot as plt

# importing functions from self-written modules ----
from fitting.models import (
    gaussian,      
    linear,  
    single_gauss_line_model,
    blended_Ha_line_model,
    blended_Hb_line_model
)
# ---------------------------------------------------

def create_window(waves_fluxes_tuple, line_obs=None, line_center_idx=None, velocity_window=True, velocity_window_size=2000):
    '''
    creates a window around a certain value 
    args:
        waves_fluxes_tuple (tuple): tuple containing wavelength and flux values for a given galaxy spectrum
        line_obs (float): emission line wavelength in the observed frame
        line_center_idx (float): index of data point at the center of emission line
        velocity_window (bool): whether to use velocity-space (conventional) or wavelength space (if resolution varies)
        velocity_window_size (int): value to set the size of the window (typical values range from 2000-3000 km/s)

    returns (tuple): wavelength and flux values within the window for a given galaxy spectrum
    '''
    waves, fluxes = waves_fluxes_tuple

    # creating window in velocity-space to search for emission lines (velocity space does not change with redshift)
    if velocity_window:
        if line_obs is None:
            raise ValueError("line_obs is required when velocity_window=True")
        
        c = 299792.458   # [km/s]
        delta_lambda = (line_obs * velocity_window_size) / c  # converting velocity-space window into wavelength units
        mask = np.abs(waves - float(line_obs)) < delta_lambda
        return waves[mask], fluxes[mask]

    # wavelength-index window
    if ((line_center_idx is None) or (line_center_idx < 0) or (line_center_idx >= len(waves))):
        # fallback: compute nearest index from the observed wavelength
        if line_obs is None:
            raise ValueError("line_center_idx invalid and line_obs is None")
        line_center_idx = int(np.argmin(np.abs(waves - float(line_obs))))

    window = 13
    start = max(0, line_center_idx - window)
    end   = min(len(waves), line_center_idx + window)
    return waves[start:end], fluxes[start:end]

def plot_full_spectrum(ID, wavs, fluxes, redshift, data_label, data_color, 
                       loc_obs_emission_lines, emission_line_labels, emission_line_colors, 
                       fit=None, save=False):
    '''
    args:
        ID (int): source identification
        wavs (arr): wavelength array
        fluxes (arr): flux array
        redshift (float): redshift of source
        data_label (str): label for spectral data
        data_color (str): color for spectral data
        loc_obs_emission_lines (list): observed wavelengths for emission lines
        emission_line_labels (list): respective labels for emission lines
        emission_line_colors (list): respective colors for emission lines
        fit (str): indicates whether to display fitting results from "curve_fit" or "emcee"
        save (bool): choice to save fig

    returns: figure of the full spectrum
    '''

    plt.figure(figsize=(10,5))
    plt.title(f"ID: {ID} , z={round(redshift, 2)}", fontsize=15)
    plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
    plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)

    # spectral data
    plt.step(wavs, fluxes, color=data_color, where='mid', label=data_label)

    # emission line markers
    for i in range(len(loc_obs_emission_lines)):
        plt.axvline(loc_obs_emission_lines[i], color=emission_line_colors[i], linestyle='dashed', linewidth=0.75, label=emission_line_labels[i])
    
    # top axis label (restframe wavelength)
    secax = plt.gca().secondary_xaxis('top',functions=(lambda x: x/(1+redshift),lambda x: x*(1+redshift)))
    secax.set_xlabel(r'Rest Wavelength [\AA{}]', labelpad=15, fontsize=15)
    secax.tick_params(labelsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.legend(fontsize=14)
    if save == True:
        plt.savefig('../../outputs/plots/example_source.png',dpi=500)
    plt.show()

def plot_emission_line_spectrum(ID, wavs, fluxes, redshift, data_label, data_color, 
                                loc_obs_emission_lines, line_center_idx, emission_line_labels, emission_line_colors, 
                                line_models=None, line_model_outputs=None, line_model_output_errs=None, line_model_color='#407899', 
                                vel_window=True, vel_window_size=2000, fit_type=None, fit=None, save=False):
    '''
    args:
        ID (int): source identification
        wavs (arr): wavelength array
        fluxes (arr): flux array
        redshift (float): redshift of source
        data_label (str): label for spectral data
        data_color (str): color for spectral data
        loc_obs_emission_lines (list): observed wavelengths for emission lines
        line_center_idx (list): indices for centers of emission lines
        emission_line_labels (list): respective labels for emission lines
        emission_line_colors (list): respective colors for emission lines
        line_models (list): models for emission lines
        line_model_outputs (list): emission line model output median value
        line_model_output_errs (list): emission line model output [0] 16th and [1] 84th percentile errors
        line_model_color (str): color for line model
        fit_type (str): indicator of first or second fit for PRISM + Grating sources
        fit (str): indicates whether to display fitting results from "curve_fit" or "emcee"
        save (bool): choice to save fig

    returns: figure for each emission line
    '''

    # if source is PRISM + Grating, only plot H-alpha fits
    if fit_type == "second":
        wav_window, flux_window = create_window((wavs,fluxes),loc_obs_emission_lines[0],line_center_idx[0],velocity_window=vel_window, velocity_window_size=vel_window_size)  
        x_arr = np.linspace(wav_window[0], wav_window[-1],1000)

        plt.figure()
        plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
        plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)

        if fit == 'curve_fit':
            plt.step(wav_window, flux_window, c=data_color, where='mid', label=data_label)

            y_model_func = line_models[0]
            y_model = y_model_func(x_arr, *line_model_outputs[0])
            plt.plot(x_arr, y_model, color=line_model_color,label='curve fit')
            #plt.axvline(loc_obs_emission_lines[i], color='darkslategrey', linestyle='dashed', linewidth=1, label=emission_line_labels[i])
            
        elif fit == 'emcee':
            plt.step(wav_window, flux_window, c=data_color, where='mid', label=data_label)
            y_model_func = line_models[0]

            y_model = y_model_func(x_arr, *line_model_outputs[0])
            plt.plot(x_arr, y_model, color=line_model_color, label='emcee fit')

            y_model_16_err = y_model_func(x_arr, *line_model_output_errs[0][0])
            y_model_84_err = y_model_func(x_arr, *line_model_output_errs[1][0])
            plt.fill_between(x_arr, y_model_16_err, y_model_84_err, color='#407899', alpha=0.3)
            #plt.axvline(loc_obs_emission_lines[1], color='darkslategrey', linestyle='dashed', linewidth=1, label=emission_line_labels[i])

        else:
            plt.step(wav_window, flux_window, c=data_color, where='mid', label=data_label)
            plt.axvline(wavs[line_center_idx[0]], color='darkslategrey', linestyle='dashed', linewidth=1, label=emission_line_labels[0])
            plt.axhline(fluxes[line_center_idx[0]], color='darkslategrey', linestyle='dashed', linewidth=1)
            
        # top axis label (restframe wavelength)
        secax = plt.gca().secondary_xaxis('top',functions=(lambda x: x/(1+redshift),lambda x: x*(1+redshift)))
        secax.set_xlabel(r'Rest Wavelength [\AA{}]', labelpad=15, fontsize=15)
        secax.tick_params(labelsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        plt.text(0.70, 0.70,  
                    f"{data_label} \nID: {ID} \nline: {emission_line_labels[0]} \n$z$ = {round(redshift, 3)}",  
                    color='black', 
                    fontsize=15, 
                    fontweight='heavy',
                    transform=plt.gca().transAxes)

        if save == True:
            plt.savefig('../../outputs/plots/example_source_{emission_line_labels[0]}.png',dpi=500)
        plt.show()

    # plotting both H-alpha and H-beta fits for all other sources
    else:
        for i in range(len(loc_obs_emission_lines)):
            wav_window, flux_window = create_window((wavs,fluxes),loc_obs_emission_lines[i],line_center_idx[i],velocity_window=vel_window, velocity_window_size=vel_window_size)  
            x_arr = np.linspace(wav_window[0], wav_window[-1],1000)

            plt.figure()
            plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
            plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)

            if fit == 'curve_fit':
                plt.step(wav_window, flux_window, c=data_color, where='mid', label=data_label)

                y_model_func = line_models[i]
                y_model = y_model_func(x_arr, *line_model_outputs[i])
                plt.plot(x_arr, y_model, color=line_model_color,label='curve fit')
                #plt.axvline(loc_obs_emission_lines[i], color='darkslategrey', linestyle='dashed', linewidth=1, label=emission_line_labels[i])
                
            elif fit == 'emcee':
                plt.step(wav_window, flux_window, c=data_color, where='mid', label=data_label)
                y_model_func = line_models[i]

                y_model = y_model_func(x_arr, *line_model_outputs[i])
                plt.plot(x_arr, y_model, color=line_model_color, label='emcee fit')

                y_model_16_err = y_model_func(x_arr, *line_model_output_errs[0][i])
                y_model_84_err = y_model_func(x_arr, *line_model_output_errs[1][i])
                plt.fill_between(x_arr, y_model_16_err, y_model_84_err, color='#407899', alpha=0.3)
                #plt.axvline(loc_obs_emission_lines[1], color='darkslategrey', linestyle='dashed', linewidth=1, label=emission_line_labels[i])

            else:
                plt.step(wav_window, flux_window, c=data_color, where='mid', label=data_label)
                plt.axvline(wavs[line_center_idx[i]], color='darkslategrey', linestyle='dashed', linewidth=1, label=emission_line_labels[i])
                plt.axhline(fluxes[line_center_idx[i]], color='darkslategrey', linestyle='dashed', linewidth=1)
                
            # top axis label (restframe wavelength)
            secax = plt.gca().secondary_xaxis('top',functions=(lambda x: x/(1+redshift),lambda x: x*(1+redshift)))
            secax.set_xlabel(r'Rest Wavelength [\AA{}]', labelpad=15, fontsize=15)
            secax.tick_params(labelsize=14)
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)
            
            plt.text(0.70, 0.70,  
                        f"{data_label} \nID: {ID} \nline: {emission_line_labels[i]} \n$z$ = {round(redshift, 3)}",  
                        color='black', 
                        fontsize=15, 
                        fontweight='heavy',
                        transform=plt.gca().transAxes)

            if save == True:
                plt.savefig('../../outputs/plots/example_source_{emission_line_labels[i]}.png',dpi=500)
            plt.show()

def CEERS_specific_spectra_plotting_func(ID, wavs, fluxes, redshift, data_label, data_color, 
                                         loc_obs_emission_lines, line_center_idx, emission_line_labels, emission_line_colors, 
                                         line_models=[blended_Ha_line_model,single_gauss_line_model], line_model_outputs=None, line_model_output_errs=None, line_model_color='#407899', 
                                         vel_window=True, vel_window_size=2000, fit=None, save=False):
    '''
    args:
        ID (int): source identification
        wavs (arr): wavelength array
        fluxes (arr): flux array
        redshift (float): redshift of source
        data_label (str): label for spectral data
        data_color (str): color for spectral data
        loc_obs_emission_lines (list): observed wavelengths for emission lines
        line_center_idx (list): indices for centers of emission lines
        emission_line_labels (list): respective labels for emission lines
        emission_line_colors (list): respective colors for emission lines
        line_models (list): models for emission lines
        line_model_outputs (list): emission line model output median value
        line_model_output_errs (list): emission line model output [0] 16th and [1] 84th percentile errors
        line_model_color (str): color for line model
        fit (str): indicates whether to display fitting results from "curve_fit" or "emcee"
        save (bool): choice to save fig

    returns: a figure of the full spectrum and for each emission line
    '''

    plot_full_spectrum(ID, wavs, fluxes, redshift, data_label, data_color, 
                       loc_obs_emission_lines, emission_line_labels, emission_line_colors, 
                       fit=fit, save=save)
            
    plot_emission_line_spectrum(ID, wavs, fluxes, redshift, data_label, data_color, 
                                loc_obs_emission_lines, line_center_idx, emission_line_labels, emission_line_colors, 
                                line_models=line_models, line_model_outputs=line_model_outputs, line_model_output_errs=line_model_output_errs, line_model_color=line_model_color, 
                                vel_window=vel_window, vel_window_size=vel_window_size, fit=fit, save=save)


def RUBIES_specific_spectra_plotting_func(source, df, ID, redshift, data_color, 
                                          loc_obs_emission_lines, line_center_idx, emission_line_labels, emission_line_colors, 
                                          line_models=None, line_model_outputs=None, line_model_output_errs=None, line_model_color='#407899', fit_type=None, fit=None, save=False):
    '''
    args:
        source (int): index of source in DataFrame
        df (DataFrame): DataFrame
        ID (int): source identification
        redshift (float): redshift of source
        data_color (str): color for spectral data
        loc_obs_emission_lines (list): observed wavelengths for emission lines
        line_center_idx (list): indices for centers of emission lines
        emission_line_labels (list): respective labels for emission lines
        emission_line_colors (list): respective colors for emission lines
        line_models (list): models for emission lines
        line_model_outputs (list): emission line model output median value
        line_model_output_errs (list): emission line model output [0] 16th and [1] 84th percentile errors
        line_model_color (str): color for line model
        fit_type (str): indicator of first or second fit for PRISM + Grating sources
        fit (str): indicates whether to display fitting results from "curve_fit" or "emcee"
        save (bool): choice to save fig

    returns: a figure of the full spectrum and for each emission line
    '''

    if df['SPECTRA TYPE'][source] == 'GRATING':

            wavs = df['RED GRATING WAVS'][source]
            fluxes = df['RED GRATING FLUXES'][source]
            data_label = 'Grating spectra'
            line_models = [blended_Ha_line_model,single_gauss_line_model]

            plot_full_spectrum(ID, wavs, fluxes, redshift, data_label, data_color, 
                               loc_obs_emission_lines, emission_line_labels, emission_line_colors, 
                               fit=fit, save=save)
            
            plot_emission_line_spectrum(ID, wavs, fluxes, redshift, data_label, data_color, 
                                        loc_obs_emission_lines, line_center_idx, emission_line_labels, emission_line_colors, 
                                        line_models=line_models, line_model_outputs=line_model_outputs, line_model_output_errs=line_model_output_errs, line_model_color=line_model_color, vel_window=True, vel_window_size=2000, 
                                        fit=fit, save=save)

    # prism visualization
    elif df['SPECTRA TYPE'][source] == 'PRISM':

            wavs = df['PRISM WAVS'][source]
            fluxes = df['PRISM FLUXES'][source]
            data_label = 'PRISM spectra'
            line_models = [single_gauss_line_model,blended_Hb_line_model]

            plot_full_spectrum(ID, wavs, fluxes, redshift, data_label, data_color, 
                               loc_obs_emission_lines, emission_line_labels, emission_line_colors, 
                               fit=fit, save=save)
            
            plot_emission_line_spectrum(ID, wavs, fluxes, redshift, data_label, data_color, 
                                        loc_obs_emission_lines, line_center_idx, emission_line_labels, emission_line_colors, 
                                        line_models=line_models, line_model_outputs=line_model_outputs, line_model_output_errs=line_model_output_errs, line_model_color=line_model_color, vel_window=False, vel_window_size=2000, 
                                        fit=fit, save=save)

    # prism + grating visualization
    else:
            wavs1 = df['PRISM WAVS'][source]
            fluxes1 = df['PRISM FLUXES'][source]
            wavs2 = df['RED GRATING WAVS'][source]
            fluxes2 = df['RED GRATING FLUXES'][source]
            data_label1 = 'PRISM spectra'
            data_label2 = 'Grating spectra'

            # second round of fitting with Grating data
            if fit_type == "second":
                data_label = data_label2
                line_models = [blended_Ha_line_model,blended_Hb_line_model]

            # first round of fitting with PRISM data
            else:
                data_label = data_label1
                line_models = [single_gauss_line_model,blended_Hb_line_model]

            plt.figure(figsize=(10,5))
            plt.xlabel(r'$\lambda$ ($\rm{\AA}$)', fontsize=15, labelpad=15)
            plt.ylabel(r'$f_{\lambda}$ [$ergs$ $\cdot$ $s^{-1}$ $\cdot$ $cm^{-2}$ $\cdot$ $\rm{\AA}^{-1}$]', fontsize=15, labelpad=15)

            # spectral data
            plt.step(wavs1, fluxes1, color=data_color, where='mid', label=data_label1)
            plt.step(wavs2, fluxes2, color=data_color, where='mid', label=data_label2)

            # emission line markers
            for i in range(len(loc_obs_emission_lines)):
                plt.axvline(loc_obs_emission_lines[i], color=emission_line_colors[i], linestyle='dashed', linewidth=0.75, label=emission_line_labels[i])
            
            # top axis label (restframe wavelength)
            secax = plt.gca().secondary_xaxis('top',functions=(lambda x: x/(1+redshift),lambda x: x*(1+redshift)))
            secax.set_xlabel(r'Rest Wavelength [\AA{}]', labelpad=15, fontsize=15)
            secax.tick_params(labelsize=14)
            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)

            # plot text
            plt.text(0.70, 0.80,
                     f"RUBIES PRISM+Grating \nID: {ID} \n$z$ = {round(redshift, 3)}",  
                     color='black', 
                     fontsize=15, 
                     fontweight='heavy',
                     transform=plt.gca().transAxes)

            if save == True:
                plt.savefig('../../outputs/plots/example_source.png',dpi=500)
            plt.show()

            plot_emission_line_spectrum(ID, wavs1, fluxes1, redshift, data_label, data_color, 
                                        loc_obs_emission_lines, line_center_idx, emission_line_labels, emission_line_colors, 
                                        line_models=line_models, line_model_outputs=line_model_outputs, line_model_output_errs=line_model_output_errs, line_model_color=line_model_color, vel_window=False, vel_window_size=2000, 
                                        fit_type=fit_type, fit=fit, save=save)