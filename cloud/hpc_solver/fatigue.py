"""
Simplified fatigue-life estimation from a static stress result, using
Basquin's equation (the standard strain-life/stress-life power law):

    sigma_a = sigma_f' * (2*N_f)^b

This is the newest and least-validated capability in this solver --
there's no NASTRAN-95 fatigue rigid format to cross-check against (1976
NASTRAN didn't have one; fatigue solvers are a much later addition to
commercial codes). Real caveats, stated plainly rather than buried:

  - treats the static stress result as a fully-reversed amplitude
    (R = -1) -- no mean-stress correction (Goodman/Soderberg/Gerber)
  - uses representative, NOT certified, material fatigue constants --
    real fatigue design uses tested S-N data for the actual alloy/heat/
    surface finish, which vary substantially from these textbook values
  - a single constant-amplitude cycle, not a real load spectrum (no
    Miner's-rule cycle counting across a duty cycle)

This is an engineering ESTIMATE for relative comparison between design
variants, not a certification-grade fatigue prediction.
"""
import numpy as np

# Representative fatigue strength coefficient (sigma_f', MPa) and exponent
# (b) per material -- textbook-typical values (see Shigley's Mechanical
# Engineering Design / Boresi's Advanced Mechanics of Materials), not
# specific to any certified alloy/temper/surface condition.
FATIGUE_PRESETS = {
    "steel":    dict(sigma_f_prime=900.0, b=-0.095),
    "aluminum": dict(sigma_f_prime=470.0, b=-0.102),
    "titanium": dict(sigma_f_prime=1400.0, b=-0.080),
}

RUNOUT_CYCLES = 1.0e9  # reported life cap ("runout") instead of an absurd number for near-zero stress


def life_from_stress(stress_amplitude_mpa, sigma_f_prime, b):
    """
    Cycles to failure via Basquin's equation, vectorized over an array of
    stress amplitudes (MPa, must be positive). Stress at/below the level
    corresponding to RUNOUT_CYCLES is reported as RUNOUT_CYCLES rather
    than extrapolating the power law to an enormous, physically
    meaningless number.
    """
    stress_amplitude_mpa = np.asarray(stress_amplitude_mpa, dtype=float)
    floor = sigma_f_prime * (2 * RUNOUT_CYCLES) ** b
    safe_stress = np.maximum(stress_amplitude_mpa, floor)
    n2 = (safe_stress / sigma_f_prime) ** (1.0 / b)
    return np.minimum(n2 / 2.0, RUNOUT_CYCLES)
