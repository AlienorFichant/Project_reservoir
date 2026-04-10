import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL AND GEOMETRIC PARAMETERS
# All base units : SI (m, s, Pa, m³) unless stated otherwise
# ═══════════════════════════════════════════════════════════════════════════

# Core geometry
L   = 5.7    # core length [cm]
d   = 3.7    # core diameter [cm]
A   = np.pi * (d/2)**2   # cross-sectional area [cm²]
PV  = 13.87  # pore volume [cm³]

# Fluid properties
mu_o = 1.433e-3   # oil viscosity [Pa.s]
mu_w = 0.96e-3    # water viscosity [Pa.s]

# Rock properties
Kw   = 2.42783058e-13  # absolute permeability (from water endpoint) [m²]
phi  = 0.23            # porosity [-]

# Boundary saturations
Swi  = 0.19   # irreducible water saturation [-]
Sor  = 0.29   # residual oil saturation [-]

# Injection conditions
Brine_inj_rate = 1.0   # brine injection rate [cm³/min]

# ── Unit conversions (done once, used everywhere) ─────────────────────────
A_m2    = A * 1e-4                         # [cm²] → [m²]
L_m     = L * 1e-2                         # [cm]  → [m]
Q_m3s   = (Brine_inj_rate * 1e-6) / 60.0  # [cm³/min] → [m³/s]
u_darcy = Q_m3s / A_m2                     # Darcy velocity [m/s]

# ── CFL-safe timestep estimate ────────────────────────────────────────────
# CFL condition for explicit saturation update:
#   dt < phi * dx / (u * max(df/dSw))
# max(df/dSw) ≈ 4 is a conservative upper bound for Corey kr
N_CELLS    = 100
dx_m       = L_m / N_CELLS
dt_CFL     = phi * dx_m / (u_darcy / phi * 4.0)
DT_SAFE    = dt_CFL * 0.5   # safety factor 0.5

print(f"Darcy velocity  : {u_darcy:.4e} m/s")
print(f"CFL timestep    : {dt_CFL:.3f} s  →  using dt = {DT_SAFE:.3f} s")
print(f"Est. BT time    : {phi * L_m / u_darcy / 60:.2f} min\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL DATA
# Pressure in bar (experimental) → converted to Pa where needed
# Times in minutes
# Volumes in cm³
# ═══════════════════════════════════════════════════════════════════════════

# Measured pressure drop across core [bar]
t_dp_min = [-0.45, 0.72, 1.22, 2.22, 3.22, 4.22, 5.22, 6.22,
             7.22,  8.22, 9.22, 10.22, 11.22, 12.22, 13.22, 14.22]
dp_bar   = [0.04,  0.04, 0.04, 0.05,  0.08,  0.12,  0.14,  0.18,
            0.20,  0.21, 0.22, 0.22,  0.22,  0.22,  0.22,  0.22]

# Cumulative produced volumes [cm³] at experimental times [min]
t_exp_min    = [0.0, 0.0, 0.2, 1.2, 2.2, 3.2, 4.2,  5.2,  6.2,
                7.2, 8.2, 9.2, 10.2, 11.2, 13.2, 15.2, 16.2]
Vo_cumul_exp = [0.0, 0.0, 0.0, 1.42, 2.42, 3.12, 3.82, 4.62, 5.42,
                5.52, 6.62, 6.92, 7.02, 6.72, 6.82, 6.72, 6.82]
Vw_cumul_exp = [0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                0.80, 0.80, 1.50, 2.40, 3.50, 5.70, 7.80, 9.20]

# Convert experimental pressure drop : bar → Pa (for Darcy kr calculation)
dp_Pa_exp = np.array(dp_bar) * 1e5   # [Pa]


# ═══════════════════════════════════════════════════════════════════════════
# COREY RELATIVE PERMEABILITY CLASS
# Owns its own parameter copies → no lambda/self capture issues
# ═══════════════════════════════════════════════════════════════════════════

class CCoreyRelperms:
    """
    Corey relative permeability model for a two-phase wetting/non-wetting system.

    All parameters are stored as instance attributes and read at call time,
    so different instances are fully independent even when created in a loop.
    """

    def __init__(self, Nw, Nn, krwn, Swirr, Snr):
        """
        Parameters
        ----------
        Nw    : Corey exponent for wetting phase (water)
        Nn    : Corey exponent for non-wetting phase (oil)
        krwn  : endpoint relative permeability for water at Sw = 1 - Snr
        Swirr : irreducible water saturation
        Snr   : residual non-wetting (oil) saturation
        """
        self.Nw    = Nw
        self.Nn    = Nn
        self.krwn  = krwn
        self.Swirr = Swirr
        self.Snr   = Snr

    def _normSw(self, Sw):
        """Normalised water saturation in [0, 1]."""
        return np.clip(
            (Sw - self.Swirr) / (1.0 - self.Snr - self.Swirr),
            0.0, 1.0
        )

    def krw(self, Sw):
        """Water (wetting) relative permeability."""
        return self.krwn * self._normSw(Sw) ** self.Nw

    def kro(self, Sw):
        """Oil (non-wetting) relative permeability."""
        return (1.0 - self._normSw(Sw)) ** self.Nn


# ═══════════════════════════════════════════════════════════════════════════
# IMPES MODEL  — structure preserved from original code
# Only changes vs original:
#   1. fDeltat set to DT_SAFE (CFL-safe) instead of 60 s
#   2. fLeftDarcyVelocity uses corrected u_darcy [m/s]
#   3. kr functions delegated to CCoreyRelperms instance
# ═══════════════════════════════════════════════════════════════════════════

def normSw(fSw, fSwirr, fSnr):
    return (fSw - fSwirr) / (1.0 - fSnr - fSwirr)

class CCoreyWetting:
    """Wetting phase relative permeability (Corey)."""
    def __init__(self, fNw, fKrwn, fSwirr, fSnr):
        self.fNw    = fNw
        self.fKrwn  = fKrwn
        self.fSwirr = fSwirr
        self.fSnr   = fSnr
    def __call__(self, afSw):
        afnSw = np.clip(normSw(afSw, self.fSwirr, self.fSnr), 0.0, 1.0)
        return self.fKrwn * afnSw ** self.fNw

class CCoreyNonWetting:
    """Non-wetting phase relative permeability (Corey)."""
    def __init__(self, fNn, fSwirr, fSnr):
        self.fNn    = fNn
        self.fSwirr = fSwirr
        self.fSnr   = fSnr
    def __call__(self, afSw):
        afnSw = np.clip(normSw(afSw, self.fSwirr, self.fSnr), 0.0, 1.0)
        return (1.0 - afnSw) ** self.fNn


class CModel:
    """
    1D incompressible two-phase flow model (IMPES).
    Pressure BC at outlet, flow rate BC at inlet.
    """
    def __init__(self, Nw, Nn, krwn, fSwirr, fSnr):
        """
        Parameters
        ----------
        Nw, Nn   : Corey exponents for water and oil
        krwn     : water endpoint relative permeability
        fSwirr   : irreducible water saturation
        fSnr     : residual oil saturation
        """
        # Grid
        self.iNumCells = N_CELLS

        # Physical dimensions [SI]
        self.fLength = L_m        # core length [m]
        self.fTime   = 0.0        # current simulation time [s]
        self.fDeltat = DT_SAFE    # timestep [s]  ← CFL-safe

        # Rock properties
        self.fPoro = phi
        self.fPerm = Kw           # absolute permeability [m²]

        # Fluid viscosities [Pa.s]
        self.fNonWetViscosity = mu_o
        self.fWetViscosity    = mu_w

        # Boundary conditions
        self.fRightPressure     = 3.5e4   # outlet pressure [Pa]
        self.fLeftDarcyVelocity = u_darcy  # inlet Darcy velocity [m/s] ← corrected

        # Upwind weighting (1 = fully upwind)
        self.fMobilityWeighting = 1.0

        # Corey parameters
        self.fSwirr = fSwirr
        self.fSnr   = fSnr
        self.fKrwn  = krwn
        self.fNn    = Nn
        self.fNw    = Nw

        # Build grid, transmissibilities, initial fields and kr objects
        self.setParameters()

    def setParameters(self):
        """
        Compute derived grid quantities and initialise fields.
        Also instantiates kr objects with current Corey parameters.
        """
        self.fDeltaX    = self.fLength / self.iNumCells
        self.afxValues  = np.arange(self.fDeltaX / 2, self.fLength, self.fDeltaX)
        self.afPoro     = self.fPoro * np.ones(self.iNumCells)

        # Permeability and transmissibilities
        self._perm      = self.setPermeabilities(self.fPerm * np.ones(self.iNumCells))

        # Initial conditions
        self.afPressure   = self.fRightPressure * np.ones(self.iNumCells)
        self.afSaturation = self.fSwirr         * np.ones(self.iNumCells)

        # Instantiate kr objects with current parameter values
        # Each object owns its own copies → no shared-state risk across runs
        self.tRelpermWet    = CCoreyWetting(
            fNw=self.fNw, fKrwn=self.fKrwn,
            fSwirr=self.fSwirr, fSnr=self.fSnr
        )
        self.tRelpermNonWet = CCoreyNonWetting(
            fNn=self.fNn,
            fSwirr=self.fSwirr, fSnr=self.fSnr
        )

    def setPermeabilities(self, permVector):
        """
        Set cell permeabilities and compute face transmissibilities.
        Uses harmonic mean at internal faces.
        """
        self._perm      = permVector
        # Internal face transmissibilities [m²/(Pa.s.m) = m/Pa.s] via harmonic mean
        self._Tran      = (2.0 / (1.0/self._perm[:-1] + 1.0/self._perm[1:])) \
                          / self.fDeltaX**2
        # Right boundary transmissibility
        self._TranRight = self._perm[-1] / self.fDeltaX**2
        return permVector


class CSimulator1DIMPES:
    """
    IMPES simulator for CModel.
    Solves pressure implicitly, advances saturation explicitly.
    Unchanged from original code.
    """
    def __init__(self, InstanceOfCModel):
        self.model = InstanceOfCModel

    def dofTimestep(self):
        """One IMPES timestep : implicit pressure, explicit saturation."""
        m = self.model

        # ── Step 1 : compute phase mobilities at current saturation ───────
        # mobility = kr / viscosity  [1/(Pa.s)]
        afMobilityNonWet = m.tRelpermNonWet(m.afSaturation) / m.fNonWetViscosity
        afMobilityWet    = m.tRelpermWet(m.afSaturation)    / m.fWetViscosity

        # ── Step 2 : upwind-weighted face mobilities ──────────────────────
        afWeightedMobNonWet = (afMobilityNonWet[:-1] * m.fMobilityWeighting
                               + afMobilityNonWet[1:] * (1 - m.fMobilityWeighting))
        afWeightedMobWet    = (afMobilityWet[:-1]    * m.fMobilityWeighting
                               + afMobilityWet[1:]    * (1 - m.fMobilityWeighting))

        # ── Step 3 : phase and total transmissibilities at faces ──────────
        afTransNonWet  = m._Tran * afWeightedMobNonWet
        afTransWet     = m._Tran * afWeightedMobWet
        fNonWetTransR  = m._TranRight * afMobilityNonWet[-1]
        fTransWetRight = m._TranRight * afMobilityWet[-1]
        afTotalTrans   = afTransNonWet + afTransWet
        fTotalTransR   = fNonWetTransR + fTransWetRight

        # ── Step 4 : build pressure matrix A and RHS vector E ─────────────
        # Ax = E  gives cell pressures [Pa]
        matrixA = np.zeros((m.iNumCells, m.iNumCells))
        # First cell (inlet : Darcy velocity BC)
        matrixA[0, 0] = -afTotalTrans[0]
        matrixA[0, 1] =  afTotalTrans[0]
        # Interior cells
        for ii in range(1, m.iNumCells - 1):
            matrixA[ii, ii-1] =  afTotalTrans[ii-1]
            matrixA[ii, ii]   = -afTotalTrans[ii-1] - afTotalTrans[ii]
            matrixA[ii, ii+1] =  afTotalTrans[ii]
        # Last cell (outlet : pressure BC)
        matrixA[-1, -2] =  afTotalTrans[-1]
        matrixA[-1, -1] = -2*fTotalTransR - afTotalTrans[-1]

        vectorE = np.zeros(m.iNumCells)
        vectorE[0]  = -m.fLeftDarcyVelocity / m.fDeltaX   # inlet flux [Pa/m²... = 1/m]
        vectorE[-1] = -2.0 * fTotalTransR * m.fRightPressure  # outlet pressure [Pa]

        # ── Step 5 : solve for pressure [Pa] ─────────────────────────────
        afPressure = np.linalg.solve(matrixA, vectorE)

        # ── Step 6 : explicit saturation update ───────────────────────────
        # dSw/dt = -div(q_nw) using non-wetting (oil) fluxes
        dtOverPoro = m.fDeltat / m.afPoro

        m.afSaturation[1:-1] -= dtOverPoro[1:-1] * (
            afTransNonWet[1:]  * (afPressure[2:]  - afPressure[1:-1]) +
            afTransNonWet[:-1] * (afPressure[:-2] - afPressure[1:-1])
        )
        m.afSaturation[0]  -= dtOverPoro[0] * afTransNonWet[0] * (
            afPressure[1] - afPressure[0])
        m.afSaturation[-1] += dtOverPoro[-1] * (
            2 * fTransWetRight * (m.fRightPressure - afPressure[-1])
            - afTransWet[-1]   * (afPressure[-1]   - afPressure[-2])
        )

        # ── Step 7 : clip unphysical saturations ─────────────────────────
        m.afSaturation = np.clip(m.afSaturation,
                                 m.tRelpermNonWet.fSwirr,
                                 1.0 - m.tRelpermNonWet.fSnr)

        # ── Step 8 : update pressure and time ────────────────────────────
        m.afPressure = afPressure   # [Pa]
        m.fTime     += m.fDeltat    # [s]

    def simulateTo(self, fTargetTime):
        """Advance simulation to fTargetTime [s] using constant timesteps."""
        base_dt = self.model.fDeltat
        while self.model.fTime < fTargetTime - 1e-10:
            self.model.fDeltat = min(base_dt, fTargetTime - self.model.fTime)
            self.dofTimestep()
        self.model.fDeltat = base_dt
        self.model.fTime   = fTargetTime


# ═══════════════════════════════════════════════════════════════════════════
# run_simulation — creates a fresh CModel and runs it to completion
# Parameters are explicit arguments → easy to call with different Corey sets
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(Nw, Nn, krwn, fSwirr, fSnr,
                   t_max_min, report_interval_min):
    """
    Run one complete IMPES simulation and return time-series results.

    Parameters (all explicit, no defaults — caller decides)
    ----------
    Nw, Nn            : Corey exponents for water and oil
    krwn              : water endpoint kr
    fSwirr, fSnr      : irreducible water and residual oil saturations
    t_max_min         : total simulation time [min]
    report_interval_min : output interval [min]

    Returns
    -------
    dict with keys:
        t   [min], Vo [cm³], Vw [cm³],
        Sw  list of saturation profiles,
        P   list of pressure profiles [Pa],
        dp  list of pressure drops [Pa],
        model : final CModel instance
    """
    # Create a fresh model — guarantees independent state for each call
    model = CModel(Nw=Nw, Nn=Nn, krwn=krwn, fSwirr=fSwirr, fSnr=fSnr)
    sim   = CSimulator1DIMPES(model)

    t_max_s      = t_max_min        * 60.0   # [s]
    interval_s   = report_interval_min * 60.0   # [s]

    # Accumulators
    time_list, Vo_list, Vw_list = [], [], []
    Sw_profiles, P_profiles, dp_profiles = [], [], []
    cum_o = cum_w = 0.0
    next_report   = interval_s

    while model.fTime < t_max_s - 1e-10:

        # Advance to next report time
        target = min(next_report, t_max_s)
        sim.simulateTo(target)

        # ── Outlet fractional flow ────────────────────────────────────────
        mob_nw = model.tRelpermNonWet(model.afSaturation) / model.fNonWetViscosity
        mob_w  = model.tRelpermWet(model.afSaturation)    / model.fWetViscosity

        # Oil fraction at outlet (non-wetting = oil produced)
        f_oil  = mob_nw[-1] / (mob_nw[-1] + mob_w[-1] + 1e-30)
        q_tot  = u_darcy * A_m2   # total volumetric flow rate [m³/s]

        # Integrate over the reporting interval to get produced volumes
        dt_rep  = interval_s
        cum_o  += f_oil       * q_tot * dt_rep * 1e6   # m³ → cm³
        cum_w  += (1 - f_oil) * q_tot * dt_rep * 1e6

        # ── Pressure drop across core [Pa] — consistent with IMPES pressure ──
        # Both inlet and outlet pressures are in Pa (IMPES solves in Pa)
        dp_impes_Pa = model.afPressure[0] - model.afPressure[-1]   # [Pa]

        # Store results
        time_list.append(model.fTime / 60.0)      # s → min
        Vo_list.append(cum_o)
        Vw_list.append(cum_w)
        Sw_profiles.append(model.afSaturation.copy())
        P_profiles.append(model.afPressure.copy())   # [Pa]
        dp_profiles.append(dp_impes_Pa)               # [Pa]

        next_report += interval_s

    return {
        "t":     np.array(time_list),    # [min]
        "Vo":    np.array(Vo_list),      # [cm³]
        "Vw":    np.array(Vw_list),      # [cm³]
        "Sw":    Sw_profiles,
        "P":     P_profiles,             # [Pa]
        "dp":    np.array(dp_profiles),  # [Pa]
        "model": model,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL RELATIVE PERMEABILITY
# Uses measured dp [bar] → converted to Pa for Darcy's law
# ═══════════════════════════════════════════════════════════════════════════

def compute_exp_kr():
    """
    Compute krw and kro from experimental cumulative volumes and
    measured pressure drop using the unsteady-state Darcy method.

    Pressure unit : dp_bar [bar] → converted to [Pa] internally.
    Returns dict with Sw, krw, kro arrays.
    """
    t   = np.array(t_exp_min,    dtype=float)   # [min]
    Vo  = np.array(Vo_cumul_exp, dtype=float)   # [cm³]
    Vw  = np.array(Vw_cumul_exp, dtype=float)   # [cm³]
    tdp = np.array(t_dp_min,     dtype=float)   # [min]
    dp  = np.array(dp_bar,       dtype=float)   # [bar]

    # Interpolate measured dp onto experimental time grid
    dp_interp_fn = interp1d(tdp, dp, bounds_error=False,
                            fill_value=(dp[0], dp[-1]))
    dp_at_t_bar  = dp_interp_fn(t)              # [bar]
    dp_at_t_Pa   = dp_at_t_bar * 1e5           # [bar] → [Pa]

    # Mask near-zero pressures to avoid division instability
    valid = dp_at_t_Pa > 500.0   # threshold : 0.005 bar

    # Flow rates from cumulative volumes via finite differences
    dVo_dt = np.gradient(Vo, t)   # [cm³/min]
    dVw_dt = np.gradient(Vw, t)   # [cm³/min]

    # Convert to SI [m³/s]
    Qo_m3s = dVo_dt * 1e-6 / 60.0
    Qw_m3s = dVw_dt * 1e-6 / 60.0

    # Darcy's law : kr = Q * mu * L / (K * A * dP)
    # All quantities in SI → kr dimensionless
    kro = np.where(valid,
                   np.clip((Qo_m3s * mu_o * L_m) / (Kw * A_m2 * dp_at_t_Pa),
                           0, 1),
                   np.nan)
    krw = np.where(valid,
                   np.clip((Qw_m3s * mu_w * L_m) / (Kw * A_m2 * dp_at_t_Pa),
                           0, 1),
                   np.nan)

    # Water saturation from material balance
    Sw = Swi + Vo / PV

    return {
        "t":      t,
        "Sw":     Sw,
        "krw":    krw,
        "kro":    kro,
        "dp_Pa":  dp_at_t_Pa,    # [Pa] — kept for reference
        "dp_bar": dp_at_t_bar,   # [bar] — original experimental unit
    }


# ═══════════════════════════════════════════════════════════════════════════
# ERROR FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def error_function(params):
    """
    Compute normalised MSE between IMPES and experimental cumulative volumes.

    Parameters
    ----------
    params : list [Nw, Nn, krwn, fSwirr, fSnr]

    Returns
    -------
    float : combined normalised MSE (oil + water)
    """
    Nw, Nn, krwn, fSwirr, fSnr = params

    # Physical bounds check
    if (Nw < 0.3 or Nn < 0.3 or krwn <= 0 or krwn > 1
            or fSwirr < 0.01 or fSnr < 0.01
            or fSwirr + fSnr >= 0.99):
        return 1e6

    # Run simulation with these parameters
    res = run_simulation(
        Nw=Nw, Nn=Nn, krwn=krwn, fSwirr=fSwirr, fSnr=fSnr,
        t_max_min=max(t_exp_min),
        report_interval_min=0.5
    )
    if len(res["t"]) < 2:
        return 1e6

    t_arr  = np.array(t_exp_min,    dtype=float)
    Vo_arr = np.array(Vo_cumul_exp, dtype=float)
    Vw_arr = np.array(Vw_cumul_exp, dtype=float)

    # Only compare over the time range covered by the simulation
    mask = (t_arr >= res["t"].min()) & (t_arr <= res["t"].max())
    if mask.sum() < 2:
        return 1e6

    # Interpolate simulation onto experimental time grid
    Vo_sim = interp1d(res["t"], res["Vo"],
                      fill_value="extrapolate")(t_arr[mask])
    Vw_sim = interp1d(res["t"], res["Vw"],
                      fill_value="extrapolate")(t_arr[mask])

    # Normalise by max experimental volume so oil and water errors are comparable
    err = (np.mean((Vo_sim - Vo_arr[mask])**2) / (max(Vo_arr)**2 + 1e-9) +
           np.mean((Vw_sim - Vw_arr[mask])**2) / (max(Vw_arr)**2 + 1e-9))

    return err


# ═══════════════════════════════════════════════════════════════════════════
# STEEPEST DESCENT OPTIMISER
# ═══════════════════════════════════════════════════════════════════════════

def steepest_descent(params0, lr, max_iter, tol, h):
    """
    Minimise error_function w.r.t. Corey parameters using steepest descent
    with forward finite differences for the gradient.

    Parameters
    ----------
    params0  : initial [Nw, Nn, krwn, Swirr, Sor]
    lr       : learning rate (step size)
    max_iter : maximum number of iterations
    tol      : convergence threshold on gradient norm
    h        : finite difference step size

    Returns
    -------
    params   : optimised parameter array
    history  : list of dicts with iter, error, grad_norm, params
    """
    params  = np.array(params0, dtype=float)
    names   = ["Nw", "Nn", "krwn", "Swirr", "Sor"]
    history = []

    print(f"\n{'='*65}")
    print("Steepest descent — Corey parameter optimisation")
    print("Initial : " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, params)))
    print(f"{'='*65}")

    for it in range(max_iter):

        E0 = error_function(params)

        # Numerical gradient via forward finite differences
        grad = np.zeros(len(params))
        for j in range(len(params)):
            p_h      = params.copy()
            p_h[j]  += h
            grad[j]  = (error_function(p_h) - E0) / h

        grad_norm = np.linalg.norm(grad)

        # Gradient descent step
        params -= lr * grad

        history.append({
            "iter":      it,
            "error":     E0,
            "grad_norm": grad_norm,
            "params":    params.copy()
        })

        print(f"  it={it+1:3d}  err={E0:.5f}  |∇|={grad_norm:.5f}  "
              + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, params)))

        if grad_norm < tol:
            print(f"\n  Converged at iteration {it+1}")
            break

    print(f"\nFinal params : "
          + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, params)))
    return params, history


# ═══════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_cumulative(res, label):
    """Cumulative oil and water : IMPES vs EXP."""
    p   = res["model"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(res["t"], res["Vo"], "-", lw=2,
                 color="steelblue", label=f"Oil IMPES — {label}")
    axes[0].plot(t_exp_min, Vo_cumul_exp, "ro", ms=6, label="Oil EXP")
    axes[0].set(xlabel="Time [min]", ylabel="Volume [cm³]",
                title="Cumulative oil production")
    axes[0].legend(fontsize=8); axes[0].grid(True, ls="--", alpha=0.5)

    axes[1].plot(res["t"], res["Vw"], "-", lw=2,
                 color="darkorange", label=f"Water IMPES — {label}")
    axes[1].plot(t_exp_min, Vw_cumul_exp, "bo", ms=6, label="Water EXP")
    axes[1].set(xlabel="Time [min]", ylabel="Volume [cm³]",
                title="Cumulative water production")
    axes[1].legend(fontsize=8); axes[1].grid(True, ls="--", alpha=0.5)

    plt.suptitle(label, fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()


def plot_pressure(res, label):
    """
    Pressure drop comparison : IMPES [Pa converted to bar] vs EXP [bar].
    Both shown in bar so the user can compare on the same axis.
    """
    dp_impes_bar = res["dp"] / 1e5   # [Pa] → [bar]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res["t"],  dp_impes_bar, "-",  lw=2,
            color="steelblue", label="dP IMPES [bar]")
    ax.plot(t_dp_min,  dp_bar,       "rs", ms=6,
            label="dP EXP [bar]")
    ax.set(xlabel="Time [min]", ylabel="Pressure drop [bar]",
           title=f"Pressure drop — {label}")
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(); plt.show()


def plot_kr(res, exp, label):
    """
    Relative permeability curves : Corey model (IMPES) vs EXP.
    """
    m      = res["model"]
    Sw_vec = np.linspace(m.fSwirr, 1 - m.fSnr, 200)

    # Evaluate Corey kr on the saturation range
    krw_th = m.tRelpermWet(Sw_vec)
    kro_th = m.tRelpermNonWet(Sw_vec)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Sw_vec,   krw_th,      "b-",  lw=2,    label="krw Corey (IMPES)")
    ax.plot(Sw_vec,   kro_th,      "r-",  lw=2,    label="kro Corey (IMPES)")
    ax.plot(exp["Sw"], exp["krw"], "bo",  ms=5, alpha=0.7, label="krw EXP")
    ax.plot(exp["Sw"], exp["kro"], "rs",  ms=5, alpha=0.7, label="kro EXP")
    ax.set(xlabel="$S_w$ [-]", ylabel="Relative permeability [-]",
           title=f"Relative permeability — {label}",
           xlim=(Swi - 0.02, 1 - Sor + 0.02), ylim=(0, 1.05))
    ax.legend(); ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(); plt.show()


def plot_sat_profiles(res, label):
    """Saturation profiles along core at each report time."""
    cmap_ = plt.get_cmap("gnuplot")
    n     = len(res["Sw"])
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, sat in enumerate(res["Sw"]):
        ax.plot(res["model"].afxValues * 100, sat,
                color=cmap_(1 - i/n), label=f"{res['t'][i]:.1f} min")

    ax.set(xlim=(0, L), ylim=(Swi - 0.02, 1 - Sor + 0.02),
           xlabel="Distance [cm]", ylabel="$S_w$ [-]",
           title=f"Saturation profiles — {label}")
    ax.legend(title="Time [min]", fontsize=7, ncol=2)
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(); plt.show()


def plot_sensitivity(param_name, values, base_params):
    """
    Vary one Corey parameter and show influence on cumulative volumes.

    Parameters
    ----------
    param_name  : one of 'Nw', 'Nn', 'krwn', 'Swi', 'Sor'
    values      : list of values to test
    base_params : dict with keys Nw, Nn, krwn, Swi, Sor (baseline)
    """
    idx_map = {"Nw": 0, "Nn": 1, "krwn": 2, "Swi": 3, "Sor": 4}
    idx     = idx_map[param_name]
    cmap_   = plt.get_cmap("plasma")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(t_exp_min, Vo_cumul_exp, "ko", ms=6, zorder=5, label="EXP")
    axes[1].plot(t_exp_min, Vw_cumul_exp, "ko", ms=6, zorder=5, label="EXP")

    for i, val in enumerate(values):
        # Build parameter list, replace the varied parameter
        p = [base_params["Nw"],   base_params["Nn"],
             base_params["krwn"], base_params["Swi"], base_params["Sor"]]
        p[idx] = val

        err = error_function(p)
        res = run_simulation(
            Nw=p[0], Nn=p[1], krwn=p[2], fSwirr=p[3], fSnr=p[4],
            t_max_min=max(t_exp_min), report_interval_min=0.5
        )
        color = cmap_(i / max(len(values) - 1, 1))
        lbl   = f"{param_name}={val:.2f}  (err={err:.3f})"
        axes[0].plot(res["t"], res["Vo"], "-", color=color, lw=1.8, label=lbl)
        axes[1].plot(res["t"], res["Vw"], "-", color=color, lw=1.8, label=lbl)

    for ax, ylabel, title in zip(
            axes,
            ["Vo [cm³]", "Vw [cm³]"],
            [f"Oil — sensitivity to {param_name}",
             f"Water — sensitivity to {param_name}"]):
        ax.set(xlabel="Time [min]", ylabel=ylabel, title=title)
        ax.legend(fontsize=7); ax.grid(True, ls="--", alpha=0.5)

    plt.tight_layout(); plt.show()


def plot_convergence(history):
    """Error and gradient norm vs steepest descent iteration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].semilogy([h["error"]     for h in history], "b-o", ms=4)
    axes[1].semilogy([h["grad_norm"] for h in history], "r-o", ms=4)
    for ax, lbl in zip(axes, ["MSE error", "|gradient|"]):
        ax.set(xlabel="Iteration", ylabel=lbl, title=lbl)
        ax.grid(True, ls="--", alpha=0.5)
    plt.suptitle("Steepest descent convergence", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — calls with explicit parameters, no hidden defaults
# ═══════════════════════════════════════════════════════════════════════════

# Compute experimental kr (bar → Pa conversion handled inside)
exp = compute_exp_kr()

# ── 1. Baseline simulation ────────────────────────────────────────────────
res_base = run_simulation(
    Nw=2.0, Nn=2.0, krwn=0.4, fSwirr=Swi, fSnr=Sor,
    t_max_min=max(t_exp_min), report_interval_min=1.0
)
plot_cumulative(res_base,         label="Baseline  Nw=2  Nn=2  krwn=0.4")
plot_pressure(res_base,           label="Baseline  Nw=2  Nn=2  krwn=0.4")
plot_sat_profiles(res_base,       label="Baseline  Nw=2  Nn=2  krwn=0.4")
plot_kr(res_base, exp,            label="Baseline  Nw=2  Nn=2  krwn=0.4")

# ── 2. Sensitivity analysis ───────────────────────────────────────────────
base = {"Nw": 2.0, "Nn": 2.0, "krwn": 0.4, "Swi": Swi, "Sor": Sor}

plot_sensitivity("Nw",   [1.0, 2.0, 3.0, 4.0, 5.0], base)
plot_sensitivity("Nn",   [1.0, 2.0, 3.0, 4.0, 5.0], base)
plot_sensitivity("krwn", [0.1, 0.2, 0.4, 0.6, 0.8], base)
plot_sensitivity("Swi",  [0.10, 0.15, 0.19, 0.25],  base)
plot_sensitivity("Sor",  [0.15, 0.20, 0.29, 0.35],  base)

# ── 3. Steepest descent optimisation ─────────────────────────────────────
params_opt, history = steepest_descent(
    params0=[2.0, 2.0, 0.4, Swi, Sor],
    lr=3e-3,
    max_iter=80,
    tol=1e-6,
    h=1e-3
)
plot_convergence(history)

# ── 4. Optimised result ───────────────────────────────────────────────────
Nw_o, Nn_o, krwn_o, Swi_o, Sor_o = params_opt
res_opt = run_simulation(
    Nw=Nw_o, Nn=Nn_o, krwn=krwn_o, fSwirr=Swi_o, fSnr=Sor_o,
    t_max_min=max(t_exp_min), report_interval_min=1.0
)
opt_label = (f"Optimised  Nw={Nw_o:.3f}  Nn={Nn_o:.3f}  "
             f"krwn={krwn_o:.3f}  Swi={Swi_o:.3f}  Sor={Sor_o:.3f}")
plot_cumulative(res_opt,   label=opt_label)
plot_pressure(res_opt,     label=opt_label)
plot_sat_profiles(res_opt, label=opt_label)
plot_kr(res_opt, exp,      label=opt_label)