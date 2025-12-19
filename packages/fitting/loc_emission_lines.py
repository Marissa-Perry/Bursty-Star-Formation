import numpy as np

def line_center_loc(wave, flux, line_obs, source, delta_velocity = 2000):
    '''
    Saves the closest data point to emission line of interest.

    args:
        wave (list): list of wavelength values
        flux (list): list of flux values
        line_obs (float): emission line wavelengths in the observed frame
        source (int): index of current source
        delta_velocity (int): [km/s] window in velocity-space to search for emission lines (velocity space does not change with redshift)

    returns (int): data point closest to each emission line for all sources 
    '''
    try:
        c = 299792.458   # [km/s]
        delta_lambda = (line_obs * delta_velocity) / c  # converting velocity-space window into wavelength units

        # mask to cast onto wavelength and flux values
        bool_arr = np.abs(np.array(wave) - line_obs) < delta_lambda
        wave_window = wave[bool_arr]  # window to search for peak
        flux_window = flux[bool_arr]

        beginning_wave_window_idx = list(wave).index(wave_window[0]) # index value of the beginning of this window
        peak_idx = np.argmax(flux_window) # index value of the peak within this window
        
        return int(beginning_wave_window_idx + peak_idx)

    except Exception as e:
        print(f"Error processing source {source}: {e}")
        return None