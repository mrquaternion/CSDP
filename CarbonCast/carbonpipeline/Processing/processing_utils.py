# carbonpipeline/processing_utils.py
import numpy as np
from carbonpipeline.Processing.constants import (
    ZERO_C_IN_K,
    DRY_AIR_MOLE_FRACTION_N2,
    DRY_AIR_MOLE_FRACTION_O2,
    DRY_AIR_MOLE_FRACTION_AR
)


# -------------- Conversion --------------
def kelvin_to_celsius(T_K):
    return T_K - ZERO_C_IN_K


def pa_to_kpa(p_pa):
    return p_pa / 1000


def kpa_to_pa(p_kpa):
    return p_kpa * 1000


def kpa_to_hpa(p_kpa):
    return p_kpa * 10


def volumetric_soil_water(SWC_decimal):
    return SWC_decimal * 100

# -------------- Variable processing --------------
def wind_speed_magnitude(u10, v10):
    return np.hypot(u10, v10)


def wind_speed_direction(u10, v10):
    theta = np.degrees(np.arctan2(u10, v10))
    return (theta + 360) % 360


def relative_humidity(t2m, d2m):
    """
    Source: https://arc.net/l/quote/lrazgyii
    """
    T_air_C = kelvin_to_celsius(t2m)
    T_dew_C = kelvin_to_celsius(d2m)
    a, b = 17.625, 243.04
    gamma_air = (a * T_air_C)  / (b + T_air_C)
    gamma_dew = (a * T_dew_C)  / (b + T_dew_C)
    return 100 * np.exp(gamma_dew - gamma_air)


def vapor_pressure_deficit(t2m, d2m):
    RH = relative_humidity(t2m, d2m)
    es_kpa = saturated_vapor_pressure(kelvin_to_celsius(t2m))
    vpd_kpa = es_kpa * (1 - (RH / 100))
    return kpa_to_hpa(vpd_kpa)


def saturated_vapor_pressure(t2m):
    """
    Tetens formula: https://en.wikipedia.org/wiki/Tetens_equation
    """
    a = np.where(t2m >= 0, 17.27, 21.875)
    b = np.where(t2m >= 0, 237.3, 265.5)
    return 0.61078 * np.exp(a * t2m / (t2m + b))


def shortwave_out(avg_sdswrf, fal):
    return avg_sdswrf * fal


def longwave_out(avg_sdlwrf, avg_snlwrf):
    return avg_snlwrf - avg_sdlwrf


def net_radiation(avg_sdswrf, avg_sdlwrf, avg_snlwrf, fal):
    return avg_sdswrf + avg_sdlwrf - shortwave_out(avg_sdswrf, fal) - longwave_out(avg_sdlwrf, avg_snlwrf)


def dry_to_wet_co2_fraction(t2m, d2m, sp, XCO2_dry):
    RH = relative_humidity(t2m, d2m)
    T_air_C = kelvin_to_celsius(t2m)
    es_pa = kpa_to_pa(saturated_vapor_pressure(T_air_C))

    xH2O_wet = (RH / 100) * es_pa / sp 
    xdry_wet = 1 - xH2O_wet
    xH2O_dry = xH2O_wet / xdry_wet

    n_tot = (DRY_AIR_MOLE_FRACTION_N2 
             + DRY_AIR_MOLE_FRACTION_O2 
             + DRY_AIR_MOLE_FRACTION_AR 
             + XCO2_dry / 1e6
             + xH2O_dry)
    
    return XCO2_dry / n_tot


def soil_heat_flux(avg_ishf, avg_slhtf, avg_sdswrf, avg_sdlwrf, avg_snlwrf, fal):
    NETRAD = net_radiation(avg_sdswrf, avg_sdlwrf, avg_snlwrf, fal)
    return NETRAD - avg_ishf - avg_slhtf


def photosynthesis_photo_flux_density(avg_sdswrf, fal=None):
    # SWintoPPFD.m mathlab document by Gabriel Hould Gosselin B.ing M.Sc
    return (
        (1.741 * avg_sdswrf + 1.45)
        if fal is None
        else
        (1.741 * avg_sdswrf * fal + 1.45)
    )              


PROCESSORS = {
    'RH': relative_humidity,
    'VPD': vapor_pressure_deficit,
    'TA': kelvin_to_celsius,
    'PA': pa_to_kpa,
    'SW_OUT': shortwave_out,
    'LW_OUT': longwave_out,
    'NETRAD': net_radiation,
    'WS': wind_speed_magnitude,
    'WD': wind_speed_direction,
    'G': soil_heat_flux,
    'TS_1': kelvin_to_celsius,
    'TS_2': kelvin_to_celsius,
    'TS_3': kelvin_to_celsius,
    'TS_4': kelvin_to_celsius,
    'TS_5': kelvin_to_celsius,
    'SWC_1': lambda x: x * 100,
    'SWC_2': lambda x: x * 100,
    'SWC_3': lambda x: x * 100,
    'SWC_4': lambda x: x * 100,
    'SWC_5': lambda x: x * 100,
    'PPFD_IN': photosynthesis_photo_flux_density,
    'PPFD_OUT': photosynthesis_photo_flux_density,
    'CO2': dry_to_wet_co2_fraction,
    'WTD': lambda x: x
}


AGG_SCHEMA = {
    # Temperature/pressure
    "TA": {
        "daily": {"TA_mean": "mean", "TA_std": "std", "TA_min": "min", "TA_max": "max"},
        "monthly": {"TA_mean": "mean", "TA_std": "std", "TA_min": "min", "TA_max": "max"},
    },
    "PA": {
        "daily": {"PA_mean": "mean"},
        "monthly": {"PA_mean": "mean"},
    },

    # Precipitation
    "P": {
        "daily": {"P_sum": "sum", "P_max": "max"},
        "monthly": {"P_sum": "sum", "P_max_daily": "max"},
    },

    # Humidity components
    "RH": {
        "daily": {"RH_mean": "mean", "RH_std": "std", "RH_max": "max", "RH_min": "min"},
        "monthly": {"RH_mean": "mean", "RH_std": "std"},
    },
    "VPD": {
        "daily": {"RH_mean": "mean", "VPD_std": "std", "VPD_max": "max"},
        "monthly": {"VPD_mean": "mean", "VPD_std": "std"},
    },

    # Wind components
    "WS": {
        "daily": {"WS_mean": "mean", "WS_std": "std", "WS_max": "max"},
        "monthly": {"WS_mean": "mean", "WS_std": "std"},
    },
    "WD": {
        "daily": "DROP",
        "monthly": "DROP",
    },

    # Radiation/energy
    "SW_IN": {
        "daily": {"SW_IN_mean": "mean", "SW_IN_std": "std", "SW_IN_total": "sum", "SW_IN_max": "max"},
        "monthly": {"SW_IN_mean": "mean", "SW_IN_std": "std", "SW_IN_total": "sum"},
    },
    "SW_IN_POT": {
        "daily": {"SW_IN_POT_total": "sum"},
        "monthly": {"SW_IN_POT_total": "sum"},
    },
    "SW_OUT": {
        "daily": {"SW_OUT_mean": "mean"},
        "monthly": {"SW_OUT_mean": "mean"},
    },
    "LW_IN": {
        "daily": {"LW_IN_mean": "mean"},
        "monthly": {"LW_IN_mean": "mean"},
    },
    "LW_OUT": {
        "daily": {"LW_OUT_mean": "mean"},
        "monthly": {"LW_OUT_mean": "mean"},
    },
    "NETRAD": {
        "daily": {"NETRAD_mean": "mean", "NETRAD_std": "std", "NETRAD_total": "sum"},
        "monthly": {"NETRAD_mean": "mean", "NETRAD_std": "std", "NETRAD_total": "sum"},
    },

    # Fluxes
    "LE": {
        "daily": {"LE_mean": "mean", "LE_total": "sum"},
        "monthly": {"LE_mean": "mean", "LE_total": "sum"},
    },
    "H": {
        "daily": {"H_mean": "mean", "H_total": "sum"},
        "monthly": {"H_mean": "mean", "H_total": "sum"},
    },
    "G": {
        "daily": {"G_mean": "mean", "G_total": "sum"},
        "monthly": {"G_mean": "mean", "G_total": "sum"},
    },

    # Turbulence/Light
    "USTAR": {
        "daily": {"USTAR_mean": "mean", "USTAR_max": "max"},
        "monthly": {"USTAR_mean": "mean"},
    },
    "PPFD_IN": {
        "daily": {"PPFD_IN_integral": "sum", "PPFD_IN_max": "max"},
        "monthly": "DROP",
    },
    "PPFD_OUT": {
        "daily": {"PPFD_OUT_integral": "sum"},
        "monthly": "DROP",
    },

    # Soil water content
    **{f"SWC_{k}": {
        "daily": {f"SWC_{k}_mean": "mean", f"SWC_{k}_min": "min",
                    f"SWC_{k}_delta": lambda s: s.iloc[-1] - s.iloc[0]},
        "monthly": {f"SWC_{k}_mean": "mean", f"SWC_{k}_min": "min",
                    f"SWC_{k}_delta": lambda s: s.iloc[-1] - s.iloc[0]},
    } for k in range(1, 6)},

    # Soil temperature
    **{f"TS_{k}": {
        "daily": {f"TS_{k}_mean": "mean", f"TS_{k}_min": "min", f"TS_{k}_max": "max"},
        "monthly": {f"TS_{k}_mean": "mean", f"TS_{k}_min": "min", f"TS_{k}_max": "max"},
    } for k in range(1, 6)},

    # CO2/WTD
    "CO2": {
        "daily": {"CO2_mean": "mean"},
        "monthly": {"CO2_mean": "mean"},
    },
    "WTD": {
        "daily": {"WTD_mean": "mean"},
        "monthly": {"WTD_mean": "mean"},
    },
}