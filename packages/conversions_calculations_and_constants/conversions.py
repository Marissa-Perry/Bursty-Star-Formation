import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

def redshift_to_age(z):
    '''
    Converting z to t (age of the universe at given redshift)
    '''
    age = cosmo.age(z)
    return age.to(u.Gyr).value  

def flux_freq_to_flux_wav(flux_freq, wavelength):
    '''
    converts flux density per unit freq to flux denisty per unit wavelength 

    args:
        flux_freq (float): flux density per unit frequency [Jy] --> [ergs/(s*cm^2*Hz)]
        wavelength (float): wavelength at the measured flux [A]

    returns (float): flux density per unit wavelength [erg/(s*cm^2*A)]
    '''
    flux_freq = (10**-23) * flux_freq               # [Jy] --> [ergs/(s*cm^2*Hz)]
    
    c = 2.99702547*(10**18)                         # [A/s]
        
    flux_wav = (flux_freq * c) / (wavelength**2)    # [ergs/(s*cm^2*Hz)] --> [erg/(s*cm^2*A)]
    return flux_wav

def integrated_line_flux(A, sigma, A_err, sigma_err): 
    '''
    obtain integrated line flux by computing area of Gaussian

    args:
        A (float): measured flux density of line peak [erg/(s*cm^2*A)]
        sigma (float): stdev of gaussian
        A_err (float): associated error
        sigma_err (float): associated error
        
    returns (tuple): integrated line flux and associated error [erg/(s*cm^2)]
    '''
    integ_flux = A * sigma * np.sqrt(2 * np.pi) 
    integ_flux_err = integ_flux * np.sqrt((A_err / A)**2 + (sigma_err / sigma)**2)
    return integ_flux, integ_flux_err

def integrated_flux_to_luminosity(integ_flux, z_val):
    '''
    flux to luminosity conversion

    args:
        flux (float): integrated line flux [erg/(s*cm^2)]
        z_val (float): redshift of source
    
    returns (tuple): integrated line luminosity and associated error [erg/s]
    '''
    lum_dist = cosmo.luminosity_distance(z_val).to(u.cm).value  # [cm]
    integ_lum = 4 * np.pi * (lum_dist**2) * integ_flux
    return integ_lum

def m_AB_to_M_AB(m_AB, z):
    """
    Parameters
    ----------
    m_AB : float
        Apparent AB magnitude (e.g., near rest-frame 1500 Å)
    z : float
        Redshift of the galaxy

    Returns
    -------
    M_AB : float
        absolute rest-frame AB magnitude
    """

    d_lum = cosmo.luminosity_distance(z).to('pc').value  # luminosity distance in parsecs
    M_AB = m_AB - 5 * np.log10(d_lum / 10) + 2.5 * np.log10(1 + z)

    return M_AB

def M_AB_to_L_nu(M_ab):
    '''
    conversion of absolute AB magnitude to luminosity density per unit frequency
    
    args:
        M_ab (array): absolute magnitudes to convert [mag]

    returns:
        luminosities per unit frequency [ergs/s/Hz]
    '''
    # converting panda series to np.array
    M_ab = np.asarray(M_ab, dtype=np.float64)
    
    # constants -----------------------------------
    zero_magnitude_source_flux_density = 3631 * 1e-23  # [Jy]-->[erg/s/cm^2/Hz]
    D_10pc_cm2 = (10 * 3.086e18)**2                    # 10^2 [parsecs^2] converted to [cm^2]
    # ---------------------------------------------
    
    factor = 4 * np.pi * D_10pc_cm2 * zero_magnitude_source_flux_density

    L_nu = factor * 10**(-0.4 * M_ab)

    return L_nu

def f_lambda_to_luminosity(integ_flux, z_val):
    '''
    flux to luminosity conversion

    args:
        flux (float): integrated line flux [erg/(s*cm^2)]
        z_val (float): redshift of source
    
    returns (tuple): integrated line luminosity and associated error [erg/s]
    '''
    lum_dist = cosmo.luminosity_distance(z_val).to(u.cm).value  # [cm]
    integ_lum = 4 * np.pi * (lum_dist**2) * integ_flux
    return integ_lum
