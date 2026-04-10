import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


###Initialization

time_list = []
Sw_impes = []
P_impes=[]
Qo_impes=[]
Vo_cumul_impes=[]
Vw_cumul_impes = []

cum_oil = 0.0
cum_water = 0.0

###Reservoir Parameters

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

#Corey Parameters
Swi=0.19
Sor=0.29
krwn=0.4
Nn=2.0
Nw=2.0

###Experimental data

#Pressure drop data (bar)

t_dp_min = [-0.45, 0.72, 1.22, 2.22, 3.22, 4.22, 5.22, 6.22, 7.22, 8.22, 9.22, 10.22, 11.22, 12.22, 13.22, 14.22]

dp_bar = [0.04, 0.04, 0.04, 0.05, 0.08, 0.12, 0.14, 0.18, 0.20, 0.21, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22]

#Cumulative volumes (m3)

t_exp_min = [0.0, 0.0, 0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 13.2, 15.2, 16.2]

Vo_cumul_exp = [0.0, 0.0, 0.0, 1.42, 2.42, 3.12, 3.82, 4.62, 5.42, 5.52, 6.62, 6.92, 7.02, 6.72, 6.82, 6.72, 6.82]
Vw_cumul_exp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.80, 1.50, 2.40, 3.50, 5.70, 7.80, 9.20]

#Pressure drop conversion to Pa

dp_Pa_exp = np.array(dp_bar) * 1e5 


###IMPES solution

def normSw(fSw,fSwirr,fSnr):
    return (fSw-fSwirr)/(1.0-fSnr-fSwirr)

class CCoreyWetting:
    '''
    Wetting phase relative permeability
    '''
    def __init__(self,fNw,fKrwn,fSwirr,fSnr):
        '''
        Args:
        fNw: Exponent
        fKrwn: Relperm at 1-S_{nrw} for wetting phase
        Swirr: S_{wi} Irreducible wetting saturation
        Snrw: S_{nrw} Residual non-wetting saturation
        '''
        self.fNw = fNw
        self.fKrwn = fKrwn
        self.fSwirr = fSwirr
        self.fSnr = fSnr
    def __call__(self,afSw):
        afnSw = normSw(afSw,self.fSwirr,self.fSnr)
        return self.fKrwn*afnSw**self.fNw
    
class CCoreyNonWetting:
    '''
    Non-wetting phase relative permeability
    '''
    def __init__(self,fNn,fSwirr,fSnr):
        '''
        Args:
        Nn: Exponent
        fSwirr: S_{wi} Irreducible wetting saturation
        fSnr: S_{nr} Residual non-wetting saturation
        '''
        self.fNn = fNn
        self.fSwirr = fSwirr
        self.fSnr = fSnr
    def __call__(self,afSw):
        afnSw = normSw(afSw,self.fSwirr,self.fSnr)
        return (1.0-afnSw)**self.fNn
    
class CfractionalFlowCorey:
    '''
    Fractional flow function for Corey relative permeabilities
    '''
    def __init__(self,InstanceOfCModel):
        self.model=InstanceOfCModel
    def __call__(self,afSw):
        afnSw = normSw(afSw,self.model.fSwirr,self.model.fSnr)
        return 1.0/(1.0+(1-afnSw)**self.model.fNn*self.model.fWetViscosity/(self.model.fKrwn*afnSw**self.model.fNw*self.model.fNonWetViscosity))


class CModel:
    '''
    A 1D model for incompressible two-phase flow
    afPressure boundary at outlet and flow boundary at inlet
    '''
    def __init__(self):
        '''
        Args:
        iNumCells: Number of cells
        fLength: Total fLength [m]
        '''
        self.iNumCells = 100
        self.fLength = L_m #m
        self.fTime = 0.0
        self.fDeltat = 10 #s
        self.fPoro=phi
        self.fPerm=Kw 
        self.fNonWetViscosity = mu_o
        self.fWetViscosity = mu_w
        
        # Boundary conditions
        self.fRightPressure = 3.5E4 #Pa
        self.fLeftDarcyVelocity = u_darcy
        self.fMobilityWeighting = 1.0
        
        # Corey parameters for relative permeability
        self.fSwirr=Swi
        self.fSnr=Sor
        
        self.fKrwn=krwn
        self.fNn=Nn
        self.fNw=Nw
        
        # Set generated parameters
        self.setParameters()
        
    def setParameters(self):
        '''
        Calculate generated parameters for the model
        '''
        self.fDeltaX = self.fLength/self.iNumCells
        self.afxValues = np.arange(self.fDeltaX/2,self.fLength,self.fDeltaX)
        self.afPoro = self.fPoro*np.ones(self.iNumCells)
        
        # This next line will also define the transmissibilities
        self._perm = self.setPermeabilities(self.fPerm*np.ones(self.iNumCells))
        
        # Note that the pressures will not be used since they are 
        #solved implicitly. We include them here for completeness
        self.afPressure = self.fRightPressure*np.ones(self.iNumCells)
        self.afSaturation = self.fSwirr*np.ones(self.iNumCells)
        self.tRelpermWet = CCoreyWetting(self.fNw,self.fKrwn,self.fSwirr,self.fSnr)
        self.tRelpermNonWet = CCoreyNonWetting(self.fNn,self.fSwirr,self.fSnr)
        
    def setPermeabilities(self,permVector):
        '''
        Set permeabilities
        Args:
        permVector: A numpy array of fLength
        self.iNumCells with perm values
        '''
        self._perm = permVector
        self._Tran = (2.0/(1.0/self._perm[:-1]+1.0/self._perm[1:]))/self.fDeltaX**2
        self._TranRight = self._perm[-1]/self.fDeltaX**2
        

class CSimulator1DIMPES:
    '''
    A 1D IMPES simulator for incompressible two-phase flow
    afPressure boundary at outlet and flow boundary at inlet
    ''' 
    def __init__(self,InstanceOfCModel):
        '''
        Loading in an object of the model CModel
        '''
        self.model=InstanceOfCModel
        
    def dofTimestep(self):
        '''
        Do one fTime step of fLength self.fDeltat
        '''
        # Calculate the mobilities for the current saturation state
        afMobilityNonWet = self.model.tRelpermNonWet(self.model.afSaturation)/self.model.fNonWetViscosity
        afMobilityWet = self.model.tRelpermWet(self.model.afSaturation)/self.model.fWetViscosity
        
        # Calculate the weighted mobilities
        afWeightedMobNonWet = afMobilityNonWet[:-1]*self.model.fMobilityWeighting + afMobilityNonWet[1:]*(1-self.model.fMobilityWeighting)
        afWeightedMobWet = afMobilityWet[:-1]*self.model.fMobilityWeighting + afMobilityWet[1:]*(1-self.model.fMobilityWeighting)
        
        # Calculate the product of total mobilites and transmissibilities
        afTransNonWet = self.model._Tran*afWeightedMobNonWet
        afTransWet = self.model._Tran*afWeightedMobWet
        fNonWetTransRight = self.model._TranRight*afMobilityNonWet[-1]
        fTransWetRight = self.model._TranRight*afMobilityWet[-1]
        afTotalTrans = afTransNonWet + afTransWet
        fTotalTransRight = fNonWetTransRight + fTransWetRight
    
        # --- Build matrixA:
        matrixA = np.zeros((self.model.iNumCells,self.model.iNumCells))
        # First row
        matrixA[0,0] = -afTotalTrans[0]
        matrixA[0,1] = afTotalTrans[0]
        # Middle rows
        for ii in np.arange(1,self.model.iNumCells-1):
            matrixA[ii,ii-1] = afTotalTrans[ii-1]
            matrixA[ii,ii] = -afTotalTrans[ii-1]-afTotalTrans[ii]
            matrixA[ii,ii+1] = afTotalTrans[ii]
        # Last row
        matrixA[-1,-2] = afTotalTrans[-1]
        matrixA[-1,-1] = -2*fTotalTransRight - afTotalTrans[-1]
        
        # --- Build vectorE:
        vectorE = np.zeros(self.model.iNumCells)
        vectorE[0] = -self.model.fLeftDarcyVelocity/self.model.fDeltaX
        vectorE[-1] = -2.0*fTotalTransRight*self.model.fRightPressure
        
        #Solving for the pressures
        matrixAInv = np.linalg.inv(matrixA)
        afPressure = np.dot(matrixAInv,vectorE)
        
        dtOverafPoro = self.model.fDeltat/self.model.afPoro
        self.model.afSaturation[1:-1] = self.model.afSaturation[1:-1] - dtOverafPoro[1:-1]*(afTransNonWet[1:]*(afPressure[2:]-afPressure[1:-1]) + afTransNonWet[:-1]*(afPressure[:-2]-afPressure[1:-1]))
        self.model.afSaturation[0] = self.model.afSaturation[0] - dtOverafPoro[0]*afTransNonWet[0]*(afPressure[1]-afPressure[0])
        self.model.afSaturation[-1] = self.model.afSaturation[-1] + dtOverafPoro[-1]*(2*fTransWetRight*(self.model.fRightPressure-afPressure[-1])-afTransWet[-1]*(afPressure[-1]-afPressure[-2]))
        # Numerical check for unphysical saturations:
        maxsat = 1.0-self.model.tRelpermNonWet.fSnr
        minsat = self.model.tRelpermNonWet.fSwirr
        self.model.afSaturation[ self.model.afSaturation>maxsat ] = maxsat
        self.model.afSaturation[ self.model.afSaturation<minsat ] = minsat
        # Update pressures and time in model
        self.model.afPressure = afPressure
        self.model.fTime = self.model.fTime + self.model.fDeltat
        
        
    def simulateTo(self,fTargetTime):
        '''
        Progress simulation to specific fTarget with a constant fTimestep self.model.fDelta.Args:
        fTargetTime: fTargetTime to advance to [s]
        '''
        basefDeltat = self.model.fDeltat
        while self.model.fTime < fTargetTime:
            if self.model.fTime + basefDeltat >= fTargetTime:
                self.model.fDeltat = fTargetTime - self.model.fTime
                self.dofTimestep()
                self.model.fDeltat = basefDeltat
                self.model.fTime = fTargetTime
            else:
                self.dofTimestep()  


Imodel=CModel()
ISimIMPES = CSimulator1DIMPES(Imodel)
fMaxtime = 20*60
fReportInterval = 2*60
cmap = plt.get_cmap('gnuplot')

while ISimIMPES.model.fTime<fMaxtime:
    color = color=cmap(1-ISimIMPES.model.fTime/fMaxtime)
    ISimIMPES.simulateTo(ISimIMPES.model.fTime + fReportInterval)
    plt.plot(ISimIMPES.model.afxValues*100,ISimIMPES.model.afSaturation,color=color,label=str(int(ISimIMPES.model.fTime)/60))
    
    #data for cumulative volume production calculations
    afMobilityNonWet = ISimIMPES.model.tRelpermNonWet(ISimIMPES.model.afSaturation) / Imodel.fNonWetViscosity
    afMobilityWet = ISimIMPES.model.tRelpermWet(ISimIMPES.model.afSaturation) / Imodel.fWetViscosity

    lambda_o = afMobilityNonWet[-1]
    lambda_w = afMobilityWet[-1]
    f_o = lambda_o / (lambda_o + lambda_w)
    
    # --- Total flow (m³/s)
    q_tot = Imodel.fLeftDarcyVelocity * A * 1e-4

    # --- Phase rates
    q_o = f_o * q_tot
    q_w = (1 - f_o) * q_tot
    

    # --- Cumulative volumes (m³)
    cum_oil += q_o * ISimIMPES.model.fDeltat
    cum_water += q_w * ISimIMPES.model.fDeltat

    # --- Convert to cm³ for plotting
    Vo_cumul_impes.append(cum_oil * 1e6)
    Vw_cumul_impes.append(cum_water * 1e6)

    # --- Time in minutes for plotting
    time_list.append(ISimIMPES.model.fTime / 60)
    
    # store pressure at all cells
    P_impes.append(ISimIMPES.model.afPressure.copy())  #Pa

    # store saturation
    Sw_impes.append(ISimIMPES.model.afSaturation.copy())
    
plt.xlim([0.0,ISimIMPES.model.fLength*100])
plt.legend(title='Time [min]')
plt.xlabel('Distance [cm]')
plt.ylabel('Saturation $s_w$')
plt.show()


###Plotting V_water and V_oil in functiun of time, in the reservoir

V_cell = Imodel.fDeltaX*100 * A

water_volumes = []
oil_volumes = []

for sat in Sw_impes:
    V_w = np.sum(sat * V_cell)
    V_o = V_tot - V_w

    water_volumes.append(V_w)
    oil_volumes.append(V_o)
    
plt.plot(time_list, water_volumes, label="Water volume")
plt.plot(time_list, oil_volumes, label="Oil volume")
plt.xlabel("Time [s]")
plt.ylabel("Volume [cm³]")
plt.legend()
plt.show()

print(Vo_cumul_exp, Vo_cumul_impes)

###Plotting cumulative oil production
plt.figure()

# --- Oil
plt.plot(time_list, Vo_cumul_impes, label="Oil IMPES")
plt.plot(t_exp_min, Vo_cumul_exp, 'o', label="Oil EXP")

plt.xlabel("Time [min]")
plt.ylabel("Cumulative oil volume [cm³]")
plt.legend()
plt.grid()

plt.show()


###Comparison plots of pressure drop and real perm
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
    Sw = Vw / PV

    return {
        "t":      t,
        "Sw":     Sw,
        "krw":    krw,
        "kro":    kro,
        "dp_Pa":  dp_at_t_Pa,    # [Pa] — kept for reference
        "dp_bar": dp_at_t_bar,   # [bar] — original experimental unit
    }

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

def plot_saturation_vs_time(Sw_impes, time_list, exp):
    """
    Compare average water saturation vs time (IMPES vs EXP)
    """
    # --- Saturation moyenne IMPES
    Sw_avg_impes = [np.mean(sw) for sw in Sw_impes]

    # --- Saturation expérimentale
    Sw_exp = exp["Sw"]
    t_exp = exp["t"]

    plt.figure(figsize=(8,5))
    plt.plot(time_list, Sw_avg_impes, 'b-', lw=2, label="Sw IMPES (avg)")
    plt.plot(t_exp, Sw_exp, 'ro', label="Sw EXP")

    plt.xlabel("Time [min]")
    plt.ylabel("Water saturation [-]")
    plt.title("Average water saturation vs time")
    plt.legend()
    plt.grid()
    plt.show()

def build_impes_results(Imodel, time_list, P_impes):
    """
    Build dictionary compatible with plotting functions
    """
    # pression différentielle (entrée - sortie)
    dp_impes = [p[0] - p[-1] for p in P_impes]

    return {
        "t": np.array(time_list),
        "dp": np.array(dp_impes),
        "model": Imodel
    }

#Exp Kr
exp_results = compute_exp_kr()

#IMPES Results
impes_results = build_impes_results(Imodel, time_list, P_impes)

#Plots
plot_pressure(impes_results, "IMPES vs EXP")
plot_kr(impes_results, exp_results, "IMPES vs EXP")
plot_saturation_vs_time(Sw_impes, time_list, exp_results)