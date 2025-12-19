import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

def restframe_to_obs(restframe_val, z):
    '''
    converts a restframe value into an observed value using inputted redshift
    
    args:
        z (float): redshift of source

    returns (float): observed spectral value
    '''
    obs_val = restframe_val * (1 + z)
    return obs_val

def SFR_Ha(lum_Ha):
    '''
    SFR from H-alpha luminosity using Kennicutt (1998) calibration,
    scaled for a Chabrier (2003) IMF.

    Args:
        lum_Ha (float): L_Ha [erg / s]

    Returns:
        SFR [Msun / yr]
    '''
    factor_sal = 7.9e-42  # Salpeter IMF
    chabrier_correction = 10**(-0.24)  # ≈ 0.575
    factor_chab = factor_sal * chabrier_correction

    return factor_chab * lum_Ha

def SFR_UV(lum_density_UV):
    '''
    SFR from UV (1500 Å) luminosity density using Kennicutt 1998 calibration
    scaled for a Chabrier (2003) IMF.
    
    Args:
        lum_density_UV (float): UV luminosity density [erg/s/Hz]

    Returns:
        SFR [Msun/yr]
    '''
    factor_sal = 1.4e-28  # Kennicutt (1998), Salpeter IMF
    chabrier_correction = 10**(-0.24)  # ≈ 0.575
    factor_chab = factor_sal * chabrier_correction

    return factor_chab * lum_density_UV

def log_SFR_mass_Speagle2014(log_mass,z):
    '''
    SFMS curve from Speagle et al. 2014 (eq. 28)
    Note, resulting log(SFR) is in terms of [log(solar mass / yr)]
    '''

    def redshift_to_age(z):
        '''
        Converting z to t (age of the universe at given redshift)
        '''
        age = cosmo.age(z)
        return age.to(u.Gyr).value 
    
    t = redshift_to_age(z)

    log_SFR = (0.84 - (0.026*t)) * log_mass - (6.51 - (0.11*t))
    return log_SFR

def log_SFR_mass_Popesso2023(log_mass,z):
    '''
    SFMS curve from Popesso et al. 2023 (eq. 10 and Table 2)
    Note, resulting log(SFR) is in terms of [log(solar mass / yr)]
    '''

    def redshift_to_age(z):
        '''
        Converting z to t (age of the universe at given redshift)
        '''
        age = cosmo.age(z)
        return age.to(u.Gyr).value 

    a_0 = 0.20
    a_1 = -0.034
    b_0 = -26.134
    b_1 = 4.722
    b_2 = -0.1925
    t = redshift_to_age(z)
    
    log_SFR = ((a_1 * t) + b_1) * log_mass + (b_2*(log_mass)**2) + (b_0 + (a_0 * t))
    return log_SFR