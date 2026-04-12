import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# ═══════════════════════════════════════════════════════════════════════════
# RESERVOIR PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
#Core geometry
L = 5.7 #lenght (cm)
L_m = L*1e-2 #(m)
d = 3.7 #diameter (cm)
A = np.pi*(d/2)**2 #cross-section area (cm2)
A_m2 = A * 1e-4 
V_tot = 61.3  #cm3
PV = 13.87    #pore volume (cm3)

#Fluid properties
mu_o= 1.433E-3  #oil viscosity (Pa.s)
mu_w= 0.96E-3   #water viscosity (Pa.s)

Brine_inj_rate=1 #cm/min
Q_m3s = (Brine_inj_rate * 1e-6) / 60.0
u_darcy = Q_m3s / A_m2 

#Rock properties
phi=0.23 #Porosity
Kw=2.42783058000E-13 #m2

#Fixed Corey saturations
Swi=0.19
Sor=0.29

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL DATA
# ═══════════════════════════════════════════════════════════════════════════

# Pressure drop measured across the core [bar] at times [min]
t_dp_min = [0.72, 1.22, 2.22, 3.22, 4.22, 5.22, 6.22,
             7.22,  8.22, 9.22, 10.22, 11.22, 12.22, 13.22, 14.22]
dp_bar   = [0.04, 0.04, 0.05,  0.08,  0.12,  0.14,  0.18,
            0.20,  0.21, 0.22, 0.22,  0.22,  0.22,  0.22,  0.22]

# Cumulative produced volumes [cm³] at experimental times [min]
t_exp_min    = [0.0, 0.2, 1.2, 2.2, 3.2,  4.2,  5.2,  6.2,
                7.2, 8.2, 9.2, 10.2, 11.2, 13.2, 15.2, 16.2]
Vo_cumul_exp = [0.0, 0.0, 1.42, 2.42, 3.12, 3.82, 4.62, 5.42,
                5.52, 6.62, 6.92, 7.02, 6.72, 6.82, 6.72, 6.82]
Vw_cumul_exp = [0.0, 0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
                0.80, 0.80, 1.50, 2.40, 3.50, 5.70, 7.80, 9.20]

Sw_mean_exp = Swi + Vo_exp_arr / PV

# Convert experimental pressure drop to Pa
dp_Pa_exp = np.array(dp_bar) * 1e5  


t_exp_arr  = np.array(t_exp_min,    dtype=float)
Vo_exp_arr = np.array(Vo_cumul_exp, dtype=float)
Vw_exp_arr = np.array(Vw_cumul_exp, dtype=float)
t_dp_arr   = np.array(t_dp_min,     dtype=float)
dp_bar_arr = np.array(dp_bar,       dtype=float)

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL KR
# ═══════════════════════════════════════════════════════════════════════════
def compute_exp_kr():
    t  = t_exp_arr.copy()
    Vo = Vo_exp_arr.copy()
    Vw = Vw_exp_arr.copy()
    dp_fn = interp1d(t_dp_arr,dp_bar_arr,bounds_error=False,
                     fill_value=(dp_bar_arr[0],dp_bar_arr[-1]))
    dp_Pa = dp_fn(t)*1e5
    valid = dp_Pa > 500.0
    dVo   = np.gradient(Vo,t)*1e-6/60.0
    dVw   = np.gradient(Vw,t)*1e-6/60.0
    kro   = np.where(valid,np.clip((dVo*mu_o*L_m)/(Kw*A_m2*dp_Pa),0,1),np.nan)
    krw   = np.where(valid,np.clip((dVw*mu_w*L_m)/(Kw*A_m2*dp_Pa),0,1),np.nan)
    Sw    = Swi + Vo/PV
    return {"Sw":Sw,"krw":krw,"kro":kro,"dp_Pa":dp_Pa}

exp_kr = compute_exp_kr()


# ═══════════════════════════════════════════════════════════════════════════
# IMPES MODEL
# ═══════════════════════════════════════════════════════════════════════════

def normSw(fSw, fSwirr, fSnr):
    """Normalise water saturation to [0,1] mobile range."""
    return (fSw - fSwirr) / (1.0 - fSnr - fSwirr)


class CCoreyWetting:
    """
    Wetting phase (water) relative permeability — Corey model.
    krw(Sw) = krwn * ((Sw - Swirr) / (1 - Snr - Swirr))^Nw
    """
    def __init__(self, fNw, fKrwn, fSwirr, fSnr):
        # Store all parameters as instance attributes so each instance
        # is fully independent — no risk of sharing state between runs
        self.fNw    = fNw
        self.fKrwn  = fKrwn
        self.fSwirr = fSwirr
        self.fSnr   = fSnr

    def __call__(self, afSw):
        # Clip normalised saturation to [0,1] to avoid unphysical kr values
        afnSw = np.clip(normSw(afSw, self.fSwirr, self.fSnr), 0.0, 1.0)
        return self.fKrwn * afnSw ** self.fNw


class CCoreyNonWetting:
    """
    Non-wetting phase (oil) relative permeability — Corey model.
    kro(Sw) = (1 - (Sw - Swirr)/(1 - Snr - Swirr))^Nn
    """
    def __init__(self, fNn, fSwirr, fSnr):
        self.fNn    = fNn
        self.fSwirr = fSwirr
        self.fSnr   = fSnr

    def __call__(self, afSw):
        afnSw = np.clip(normSw(afSw, self.fSwirr, self.fSnr), 0.0, 1.0)
        return (1.0 - afnSw) ** self.fNn


class CModel:
    """
    1D incompressible two-phase flow model.
    Pressure BC at outlet (right), Darcy velocity BC at inlet (left).
    Follows OPM Flow template structure.

    CHANGE vs original : Corey params (Nw, Nn, krwn) passed as arguments
    so that multiple independent instances can be created with different
    parameters. Swi and Sor are fixed (not optimised).
    """
    def __init__(self, Nw, Nn, krwn):
        self.iNumCells = 100
        self.fLength   = L_m          # [m]
        self.fTime     = 0.0          # [s]
        self.fDeltat   = 1.0    # [s]

        self.fPoro             = phi
        self.fPerm             = Kw   # [m²]
        self.fNonWetViscosity  = mu_o # [Pa.s]
        self.fWetViscosity     = mu_w # [Pa.s]

        # Boundary conditions
        self.fRightPressure     = 3.5e4   # outlet pressure [Pa]
        self.fLeftDarcyVelocity = u_darcy # inlet Darcy velocity [m/s]
        self.fMobilityWeighting = 1.0     # 1 = fully upwind

        # Corey saturation endpoints
        self.fSwirr = Swi
        self.fSnr   = Sor

        # Corey shape parameters
        self.fKrwn = krwn
        self.fNn   = Nn
        self.fNw   = Nw

        self.setParameters()

    def setParameters(self):
        """
        Compute grid geometry, initial fields, and kr objects.
        Called once at construction. kr objects are instantiated with
        the current values of self.fNw etc. — each instance owns its
        own kr objects, so there is no cross-contamination between runs.
        """
        self.fDeltaX    = self.fLength / self.iNumCells
        self.afxValues  = np.arange(self.fDeltaX / 2, self.fLength, self.fDeltaX)
        self.afPoro     = self.fPoro * np.ones(self.iNumCells)
        self._perm      = self.setPermeabilities(self.fPerm * np.ones(self.iNumCells))
        self.afPressure   = self.fRightPressure * np.ones(self.iNumCells)
        self.afSaturation = self.fSwirr         * np.ones(self.iNumCells)

        # Instantiate kr objects with current parameter values
        # Because CCoreyWetting stores its own copies of fNw, fKrwn etc.,
        # two CModel instances with different Nw will have independent kr objects
        self.tRelpermWet    = CCoreyWetting(
            self.fNw, self.fKrwn, self.fSwirr, self.fSnr)
        self.tRelpermNonWet = CCoreyNonWetting(
            self.fNn, self.fSwirr, self.fSnr)

    def setPermeabilities(self, permVector):
        """
        Set cell permeabilities and compute face transmissibilities.
        Internal faces use harmonic mean (standard in reservoir simulation).
        """
        self._perm      = permVector
        # Harmonic mean transmissibility at internal faces [m/Pa.s / m²] = [1/Pa.s.m]
        self._Tran      = (2.0 / (1.0/self._perm[:-1] + 1.0/self._perm[1:])) \
                          / self.fDeltaX**2
        self._TranRight = self._perm[-1] / self.fDeltaX**2
        return permVector


class CSimulator1DIMPES:
    """
    1D IMPES simulator — OPM Flow template, unchanged.
    Solves pressure implicitly each timestep, then advances
    saturation explicitly using the computed pressure field.
    """
    def __init__(self, InstanceOfCModel):
        self.model = InstanceOfCModel

    def dofTimestep(self):
        """
        One IMPES timestep of length self.model.fDeltat [s].

        Step 1 : compute phase mobilities  lambda = kr/mu  [1/Pa.s]
        Step 2 : upwind-weight mobilities at faces
        Step 3 : compute phase and total transmissibilities
        Step 4 : assemble pressure matrix A and RHS vector E
        Step 5 : solve A*P = E for cell pressures [Pa]
        Step 6 : update saturations explicitly using oil-phase fluxes
        Step 7 : clip saturations to physical range [Swi, 1-Sor]
        Step 8 : store pressure and advance time
        """
        m = self.model

        # Step 1 — mobilities
        afMobilityNonWet = m.tRelpermNonWet(m.afSaturation) / m.fNonWetViscosity
        afMobilityWet    = m.tRelpermWet(m.afSaturation)    / m.fWetViscosity

        # Step 2 — upwind face mobilities
        afWeightedMobNonWet = (afMobilityNonWet[:-1] * m.fMobilityWeighting
                               + afMobilityNonWet[1:] * (1 - m.fMobilityWeighting))
        afWeightedMobWet    = (afMobilityWet[:-1]    * m.fMobilityWeighting
                               + afMobilityWet[1:]    * (1 - m.fMobilityWeighting))

        # Step 3 — transmissibilities
        afTransNonWet  = m._Tran * afWeightedMobNonWet
        afTransWet     = m._Tran * afWeightedMobWet
        fNonWetTransR  = m._TranRight * afMobilityNonWet[-1]
        fTransWetRight = m._TranRight * afMobilityWet[-1]
        afTotalTrans   = afTransNonWet + afTransWet
        fTotalTransR   = fNonWetTransR + fTransWetRight

        # Step 4 — pressure matrix
        matrixA = np.zeros((m.iNumCells, m.iNumCells))
        matrixA[0, 0] = -afTotalTrans[0]
        matrixA[0, 1] =  afTotalTrans[0]
        for ii in np.arange(1, m.iNumCells - 1):
            matrixA[ii, ii-1] =  afTotalTrans[ii-1]
            matrixA[ii, ii]   = -afTotalTrans[ii-1] - afTotalTrans[ii]
            matrixA[ii, ii+1] =  afTotalTrans[ii]
        matrixA[-1, -2] =  afTotalTrans[-1]
        matrixA[-1, -1] = -2*fTotalTransR - afTotalTrans[-1]

        vectorE = np.zeros(m.iNumCells)
        vectorE[0]  = -m.fLeftDarcyVelocity / m.fDeltaX
        vectorE[-1] = -2.0 * fTotalTransR * m.fRightPressure

        # Step 5 — solve for pressure [Pa]
        matrixAInv = np.linalg.inv(matrixA)
        afPressure = np.dot(matrixAInv, vectorE)

        # Step 6 — explicit saturation update using oil-phase fluxes
        # dSw/dt = -div(q_oil) : oil leaving a cell increases Sw there
        dtOverafPoro = m.fDeltat / m.afPoro

        m.afSaturation[1:-1] = (m.afSaturation[1:-1]
            - dtOverafPoro[1:-1] * (
                afTransNonWet[1:]  * (afPressure[2:]  - afPressure[1:-1]) +
                afTransNonWet[:-1] * (afPressure[:-2] - afPressure[1:-1])))

        m.afSaturation[0] = (m.afSaturation[0]
            - dtOverafPoro[0] * afTransNonWet[0] * (afPressure[1] - afPressure[0]))

        m.afSaturation[-1] = (m.afSaturation[-1]
            + dtOverafPoro[-1] * (
                2 * fTransWetRight * (m.fRightPressure - afPressure[-1])
                - afTransWet[-1]  * (afPressure[-1]   - afPressure[-2])))

        # Step 7 — clip to physical range
        maxsat = 1.0 - m.tRelpermNonWet.fSnr
        minsat = m.tRelpermNonWet.fSwirr
        m.afSaturation[m.afSaturation > maxsat] = maxsat
        m.afSaturation[m.afSaturation < minsat] = minsat

        # Step 8 — store pressure and advance time
        m.afPressure = afPressure   # [Pa]
        m.fTime      = m.fTime + m.fDeltat

    def simulateTo(self, fTargetTime):
        """
        Advance simulation to fTargetTime [s].
        Uses constant timestep self.model.fDeltat, shortening the last
        step if needed to land exactly on fTargetTime.
        """
        basefDeltat = self.model.fDeltat
        while self.model.fTime < fTargetTime:
            if self.model.fTime + basefDeltat >= fTargetTime:
                self.model.fDeltat = fTargetTime - self.model.fTime
                self.dofTimestep()
                self.model.fDeltat = basefDeltat
                self.model.fTime   = fTargetTime
            else:
                self.dofTimestep()


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_impes(Nw, Nn, krwn, t_max_min=20.0, report_interval_min=2.0):
    """
    Run one complete IMPES simulation with given Corey shape parameters.
    Swi and Sor are fixed globally — only Nw, Nn, krwn are varied.

    Creates a fresh CModel instance each call → fully independent runs.

    Parameters
    ----------
    Nw, Nn, krwn      : Corey shape parameters to test
    t_max_min         : simulation duration [min]
    report_interval_min : output interval [min]

    Returns
    -------
    dict with t [min], Vo [cm³], Vw [cm³], Sw profiles, P profiles [Pa],
    dp [Pa], and the CModel instance
    """
    #Compute independent runs
    model = CModel(Nw=Nw, Nn=Nn, krwn=krwn)
    sim   = CSimulator1DIMPES(model)

    t_max_s    = t_max_min        * 60.0
    interval_s = report_interval_min * 60.0

    t_list, Vo_list, Vw_list = [], [], []
    Sw_list, P_list, dp_list = [], [], []
    cum_o = cum_w = 0.0
    next_report = interval_s

    while model.fTime < t_max_s - 1e-10:
        target    = min(next_report, t_max_s)
        t_before  = model.fTime
        sim.simulateTo(target)
        dt_actual = model.fTime - t_before   # elapsed time this step [s]

        # Outlet fractional flow
        mob_nw = model.tRelpermNonWet(model.afSaturation) / model.fNonWetViscosity
        mob_w  = model.tRelpermWet(model.afSaturation)    / model.fWetViscosity
        f_oil  = mob_nw[-1] / (mob_nw[-1] + mob_w[-1] + 1e-30)
        q_tot  = model.fLeftDarcyVelocity * A_m2   # [m³/s]

        cum_o += f_oil       * q_tot * dt_actual * 1e6   # → cm³
        cum_w += (1 - f_oil) * q_tot * dt_actual * 1e6

        # Pressure drop inlet → outlet [Pa] — both in Pa, consistent
        dp_Pa = model.afPressure[0] - model.afPressure[-1]

        t_list.append(model.fTime / 60.0)
        Vo_list.append(cum_o)
        Vw_list.append(cum_w)
        Sw_list.append(model.afSaturation.copy())
        P_list.append(model.afPressure.copy())
        dp_list.append(dp_Pa)

        next_report += interval_s

    return {
        "t":     np.array(t_list),    # [min]
        "Vo":    np.array(Vo_list),   # [cm³]
        "Vw":    np.array(Vw_list),   # [cm³]
        "Sw":    Sw_list,
        "P":     P_list,              # [Pa]
        "dp":    np.array(dp_list),   # [Pa]
        "model": model,
    }


# ═══════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTION
#
# Minimises discrepancy on Vo and Vw only (most directly controlled by
# Corey parameters). 
#
# Formula per variable : (sum |exp - sim|)² / (sum |exp| + eps)²
#   → dimensionless, balanced regardless of the magnitude of each variable
#   → equivalent to a normalised L1 distance squared

#Potenntial changes : 
#L1 or MSE (quadratic) error type ?
#Add error on dp or not ?
# ═══════════════════════════════════════════════════════════════════════════
def error_function(sim):
    """
    Dimensionless objective function on cumulative oil and water volumes.

    Returns err_Vo + err_Vw where each term is :
        (sum_t |V_exp(t) - V_sim(t)|)² / (sum_t |V_exp(t)| + eps)²

    This is the L1-squared discrepancy normalised by the L1 norm of the
    experimental data — fully dimensionless, both terms on the same scale.
    """
    mask=(t_exp_arr>=sim["t"].min())&(t_exp_arr<=sim["t"].max())
    if mask.sum()<2:
        return 1e6

    Vo_sim=interp1d(sim["t"],sim["Vo"],fill_value="extrapolate")(t_exp_arr[mask])
    Vw_sim=interp1d(sim["t"],sim["Vw"],fill_value="extrapolate")(t_exp_arr[mask])

    eps=1e-9
    # Normalised L1-squared error — dimensionless
    err_Vo=(np.sum(np.abs(Vo_exp_arr[mask]-Vo_sim))**2) \
           /(np.sum(np.abs(Vo_exp_arr[mask]))+eps)**2
    err_Vw=(np.sum(np.abs(Vw_exp_arr[mask]-Vw_sim))**2) \
           /(np.sum(np.abs(Vw_exp_arr[mask]))+eps)**2

    return err_Vo + err_Vw


# ═══════════════════════════════════════════════════════════════════════════
# STEEPEST DESCENT
#
# Parameters :
#   Nw0, Nn0, krwn0 : starting point 
#   lr   : learning rate — size of each gradient step in parameter space
#           too large → diverges; too small → very slow
#           typical range : 0.01 – 0.1 for these normalised Corey params
#   max_iter : maximum iterations (50 is usually sufficient)
#   tol  : stop when |∇E| < tol (gradient is small → near a minimum)
#           1e-3 is appropriate — below this we are in numerical noise
#   h    : finite difference step for gradient estimation
#           must be large enough to avoid numerical noise but small enough
#           to be accurate — h=0.05 is well suited for params of order 1
# ═══════════════════════════════════════════════════════════════════════════
def steepest_descent(Nw0, Nn0, krwn0, lr, max_iter, tol, h):
    """
    Gradient descent minimisation of error_function w.r.t. (Nw, Nn, krwn).

    At each iteration :
      1. Compute E0 = error_function at current params
      2. Estimate gradient by forward finite differences :
            dE/dNw ≈ (E(Nw+h,...) - E0) / h
      3. Update params : p ← p - lr × ∇E / |∇E|
         (normalised gradient → unit step scaled by lr)
      4. Clip to physical bounds
      5. Stop if |∇E| < tol

    Using the normalised gradient (step 3) avoids explosion when the
    gradient is very large, which caused divergence in previous versions.
    """
    params=np.array([Nw0,Nn0,krwn0],dtype=float)
    names =["Nw","Nn","krwn"]
    history=[]

    print(f"\n{'='*65}")
    print("Steepest descent — minimising error on Vo and Vw")
    print(f"lr={lr}  h={h}  tol={tol}  max_iter={max_iter}")
    print("Start : "+"  ".join(f"{n}={v:.4f}" for n,v in zip(names,params)))
    print(f"{'='*65}")

    for it in range(max_iter):
        # Step 1 : current error
        sim0=run_impes(Nw=params[0],Nn=params[1],krwn=params[2])
        E0  =error_function(sim0)

        # Step 2 : numerical gradient (forward finite differences)
        grad=np.zeros(3)
        for j in range(3):
            ph=params.copy(); ph[j]+=h
            sim_h=run_impes(Nw=ph[0],Nn=ph[1],krwn=ph[2])
            grad[j]=(error_function(sim_h)-E0)/h

        gn=np.linalg.norm(grad)

        # Step 3 : normalised gradient step — avoids explosion
        # p ← p - lr × grad/|grad|  (unit direction, step size = lr)
        if gn > 1e-12:
            params=params - lr*(grad/gn)
        
        # Step 4 : physical bounds
        params[0]=np.clip(params[0],0.5,6.0)   # Nw  ∈ [0.5, 6]
        params[1]=np.clip(params[1],0.5,6.0)   # Nn  ∈ [0.5, 6]
        params[2]=np.clip(params[2],0.05,0.95) # krwn∈ [0.05, 0.95]

        history.append({"iter":it+1,"error":E0,"grad_norm":gn,
                        "params":params.copy()})
        print(f"  it={it+1:3d}  err={E0:.5f}  |∇|={gn:.5f}  "
              +"  ".join(f"{n}={v:.4f}" for n,v in zip(names,params)))

        # Step 5 : convergence check
        if gn<tol:
            print(f"  Converged at iteration {it+1}"); break

    print(f"\nOptimal : "+"  ".join(f"{n}={v:.4f}" for n,v in zip(names,params)))
    return params[0],params[1],params[2],history

# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_volumes(sim):
    plt.figure(figsize=(10,5))

    # Oil
    plt.subplot(1,2,1)
    plt.plot(t_exp_arr, Vo_exp_arr, '-', label="Oil EXP")
    plt.plot(sim["t"], sim["Vo"], '--', label="Oil IMPES")
    plt.xlabel("Time [min]"); plt.ylabel("Volume [cm³]")
    plt.title("Cumulative Oil"); plt.legend(); plt.grid()

    # Water
    plt.subplot(1,2,2)
    plt.plot(t_exp_arr, Vw_exp_arr, '-', label="Water EXP")
    plt.plot(sim["t"], sim["Vw"], '--', label="Water IMPES")
    plt.xlabel("Time [min]"); plt.ylabel("Volume [cm³]")
    plt.title("Cumulative Water"); plt.legend(); plt.grid()

    plt.tight_layout(); plt.show()


def plot_pressure(sim):
    plt.figure()
    plt.plot(t_dp_arr, dp_bar_arr, '-', label="EXP")
    plt.plot(sim["t"], sim["dp"]/1e5, '--', label="IMPES")
    plt.xlabel("Time [min]"); plt.ylabel("ΔP [bar]")
    plt.title("Pressure drop"); plt.legend(); plt.grid()
    plt.show()


def plot_kr(sim):
    m = sim["model"]
    Sw = np.linspace(Swi, 1-Sor, 200)

    plt.figure()
    plt.plot(Sw, m.tRelpermWet(Sw), '--', label="krw IMPES")
    plt.plot(Sw, m.tRelpermNonWet(Sw), '--', label="kro IMPES")

    plt.scatter(exp_kr["Sw"], exp_kr["krw"], s=20, label="krw EXP")
    plt.scatter(exp_kr["Sw"], exp_kr["kro"], s=20, label="kro EXP")

    plt.xlabel("Sw"); plt.ylabel("kr")
    plt.title("Relative permeability")
    plt.legend(); plt.grid()
    plt.show()


def plot_saturation_map(sim):
    m = sim["model"]
    x = m.afxValues*100
    Sw = np.array(sim["Sw"])

    plt.figure(figsize=(8,5))
    plt.pcolormesh(x, sim["t"], Sw, shading='auto')
    plt.colorbar(label="Sw")
    plt.xlabel("Distance [cm]")
    plt.ylabel("Time [min]")
    plt.title("Saturation map")
    plt.show()

def plot_saturation_comparison(sim, label=""):
    """
    Left  : simulated 2D saturation map Sw(x,t).
    Right : mean saturation vs time — simulated (spatial average)
            vs experimental (from material balance : Sw_mean = Swi + Vo/PV).

    The experimental data does not provide a spatial profile Sw(x),
    only the mean saturation through cumulative volume measurements.
    The comparison is therefore made on the spatially averaged saturation.
    """
    m  = sim["model"]
    x  = m.afxValues*100    # [cm]
    Sw = np.array(sim["Sw"])  # shape (n_times, n_cells)

    # Mean simulated saturation at each report time
    Sw_mean_sim=Sw.mean(axis=1)

    # Experimental mean saturation — interpolated onto simulation time grid
    Sw_mean_exp_interp=interp1d(t_exp_arr,Sw_mean_exp,
                                 bounds_error=False,
                                 fill_value=(Sw_mean_exp[0],Sw_mean_exp[-1])
                                )(sim["t"])

    fig,axes=plt.subplots(1,2,figsize=(14,5))

    # Left : 2D saturation map (simulation only — no spatial exp data)
    im=axes[0].pcolormesh(x,sim["t"],Sw,cmap="RdYlBu_r",
                          vmin=Swi,vmax=1-Sor,shading="auto")
    fig.colorbar(im,ax=axes[0],label="$S_w$ [-]")
    axes[0].set(xlabel="Distance along core [cm]",ylabel="Time [min]",
                title="Simulated saturation map $S_w(x,t)$")

    # Right : mean saturation comparison
    axes[1].plot(sim["t"],Sw_mean_sim,"-",lw=2.5,color="crimson",
                 label="IMPES — spatial mean $S_w$")
    axes[1].plot(t_exp_arr,Sw_mean_exp,"o--",lw=2,ms=6,color="steelblue",
                 label="EXP — $S_{w,mean} = S_{wi} + V_o/PV$")
    axes[1].set(xlabel="Time [min]",ylabel="Mean $S_w$ [-]",
                title="Mean water saturation — IMPES vs EXP",
                ylim=(Swi-0.02,1-Sor+0.05))
    axes[1].legend(fontsize=10); axes[1].grid(True,ls="--",alpha=0.5)

    plt.suptitle(f"Saturation — {label}",fontsize=12,fontweight="bold")
    plt.tight_layout(); plt.show()


def plot_convergence(history):
    # history est une liste de dicts : {"iter":..., "error":..., "grad_norm":...}
    iters=[h["iter"] for h in history]
    errs =[h["error"] for h in history]
    grads=[h["grad_norm"] for h in history]

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].semilogy(iters,errs,"b-o",ms=5,lw=2)
    axes[0].set(xlabel="Iteration",ylabel="Objective function",
                title="Convergence of objective function")
    axes[0].grid(True,ls="--",alpha=0.5)

    axes[1].semilogy(iters,grads,"r-o",ms=5,lw=2)
    axes[1].set(xlabel="Iteration",ylabel="|∇E|",
                title="Gradient norm")
    axes[1].grid(True,ls="--",alpha=0.5)

    plt.suptitle("Steepest descent convergence",fontsize=12,fontweight="bold")
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

Nw_opt, Nn_opt, krwn_opt, history = steepest_descent(
    Nw0=2.0, Nn0=2.0, krwn0=0.4,
    lr=1e-3,
    max_iter=200,
    tol=1e-4,
    h=0.05
)

plot_convergence(history)

sim_opt=run_impes(Nw=Nw_opt,Nn=Nn_opt,krwn=krwn_opt)
opt_label=f"Nw={Nw_opt:.3f}  Nn={Nn_opt:.3f}  krwn={krwn_opt:.3f}"

print(f"\nOptimal parameters : {opt_label}")
print(f"Final objective    : {error_function(sim_opt):.5f}")

plot_volumes(sim_opt)
plot_pressure(sim_opt)
plot_kr(sim_opt)
plot_saturation_comparison(sim_opt,label=opt_label)
