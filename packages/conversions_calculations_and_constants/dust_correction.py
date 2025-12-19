import numpy as np

def dust_reddening(H1_integ_flux_obs, H1_integ_flux_obs_err, H2_integ_flux_obs, H2_integ_flux_obs_err):
    '''
    Eq (3) and (4) from Calzetti 2001 
    
    args:
        H1_integ_flux_obs (float): observed integrated line flux of Hα
        H2_integ_flux_obs (float): observed integrated line flux of Hβ
        line (str): identification of emission line to dust correct

    returns (float): E(B-V) reddening of source
    '''
    Ha_Hb_intrinsic_lum_ratio = 2.87
    Ha_Hb_extinction = 1.163  # k(Hβ) - k(Hα)
    observed_ratio = H1_integ_flux_obs / H2_integ_flux_obs
    
    if H1_integ_flux_obs > 0 and H2_integ_flux_obs > 0:

        reddening = np.log10(observed_ratio / Ha_Hb_intrinsic_lum_ratio) / (0.4 * Ha_Hb_extinction)

        sigma_R = observed_ratio * np.sqrt((H1_integ_flux_obs_err / H1_integ_flux_obs)**2 + (H2_integ_flux_obs_err / H2_integ_flux_obs)**2)
        reddening_err = (1 / (0.4 * Ha_Hb_extinction)) * (1 / np.log(10)) * (sigma_R / observed_ratio)
    
    else:
        reddening = 0.0
        reddening_err = 0.0

    # clamp negative E(B-V) to 0
    if reddening <= 0:
        print(f"Non-physical reddening value: {reddening:.3f}")
        reddening = 0.0
        reddening_err = 0.0

    return reddening, reddening_err

def extinction_k(wav_microns):
        '''
        Extinction curve: equations (8a) and (8b) from Calzetti 2001 
        '''
        if 0.12 <= wav_microns < 0.63:
            k_lambda = 1.17*(-2.156+(1.509/wav_microns)-(0.198/(wav_microns**2))+(0.011/(wav_microns**3)))+1.78
            return k_lambda
        elif 0.63 <= wav_microns <= 2.20:
            k_lambda = 1.17*(-1.857+(1.040/wav_microns))+1.78
            return k_lambda
        else:
            return np.nan
    
def dust_correct_flux(integ_flux_obs, integ_flux_obs_errs, reddening, reddening_err, line):
    '''
    Dust correct the integrated line flux using Calzetti et al. (2001) attenuation law.

    Args:
        integ_flux_obs (float): Observed integrated line flux [erg/s/cm^2]
        integ_flux_obs_errs (list): [mean err, 16% err, 84% err]
        reddening (float): E(B−V)
        reddening_err (float): 1-sigma error on E(B−V)
        line (str): 'Ha' or 'Hb'

    Returns:
        Tuple of (corrected flux, [mean_err, 16% err, 84% err])
    '''

    def extinction_k(wav_microns):
        '''
        Extinction curve: equations (8a) and (8b) from Calzetti 2001 
        '''
        if 0.12 <= wav_microns < 0.63:
            k_lambda = 1.17*(-2.156+(1.509/wav_microns)-(0.198/(wav_microns**2))+(0.011/(wav_microns**3)))+1.78
            return k_lambda
        elif 0.63 <= wav_microns <= 2.20:
            k_lambda = 1.17*(-1.857+(1.040/wav_microns))+1.78
            return k_lambda
        else:
            return np.nan

    if line == 'Ha':
        lambda_rest = 6562.819 * 1e-4  # µm
    elif line == 'Hb':
        lambda_rest = 4861.333 * 1e-4  # µm

    k_lambda = extinction_k(lambda_rest)
    A_lambda = reddening * k_lambda  # Eq. 7 Calzetti 2001

    # Dust-corrected flux
    corr_flux = integ_flux_obs * 10**(0.4 * A_lambda)

    # Error propagation
    dA_lambda = reddening_err * k_lambda
    flux_errs_corr = [
        np.sqrt((10**(0.4 * A_lambda) * integ_flux_obs_errs[i])**2 + 
                (0.4 * np.log(10) * dA_lambda * corr_flux)**2)
        for i in range(3)
    ]

    return corr_flux, flux_errs_corr