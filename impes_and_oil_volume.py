import matplotlib.pyplot as plt
import numpy as np


time_list = []
sat_list = []
q_oil_list=[]
cum_oil_list=[]
cumul=0.0

#Reservoir Parameters

L=5.7 #cm
d=3.7 #cm
A = np.pi*(d/2)**2 #cm2
V_tot=61.3 #cm3

Brine_inj_rate=1 #cm/min
Swi=0.19
Sor=0.29

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
        self.fLength = L/100 #m
        self.fTime = 0.0
        self.fDeltat = 60
        self.fPoro=0.23
        self.fPerm=2.42783058000E-13 #m²
        self.fNonWetViscosity = 1.433E-3
        self.fWetViscosity = 0.96E-3
        
        # Boundary conditions
        self.fRightPressure = 3.5E4 #Pa
        self.fLeftDarcyVelocity = Brine_inj_rate/(A*60)
        self.fMobilityWeighting = 1.0
        
        # Corey parameters for relative permeability
        self.fSwirr=Swi
        self.fSnr=Sor
        
        self.fKrwn=0.4
        self.fNn=2.0
        self.fNw=2.0
        
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
fMaxtime = 2*3600
fReportInterval = 15*60
cmap = plt.get_cmap('gnuplot')

while ISimIMPES.model.fTime<fMaxtime:
    color = color=cmap(1-ISimIMPES.model.fTime/fMaxtime)
    ISimIMPES.simulateTo(ISimIMPES.model.fTime + fReportInterval)
    plt.plot(ISimIMPES.model.afxValues,ISimIMPES.model.afSaturation,color=color,label=str(int(ISimIMPES.model.fTime)/60))
    
    afMobilityNonWet = ISimIMPES.model.tRelpermNonWet(ISimIMPES.model.afSaturation) / Imodel.fNonWetViscosity
    afMobilityWet = ISimIMPES.model.tRelpermWet(ISimIMPES.model.afSaturation) / Imodel.fWetViscosity

    lambda_o = afMobilityNonWet[-1]
    lambda_w = afMobilityWet[-1]
    
    q_total=Imodel.fLeftDarcyVelocity * A
    f_o = lambda_o / (lambda_o + lambda_w)

    q_oil = f_o*q_total
    cumul+=q_oil*fReportInterval

    time_list.append(ISimIMPES.model.fTime)
    sat_list.append(ISimIMPES.model.afSaturation.copy())
    q_oil_list.append(q_oil)
    cum_oil_list.append(cumul)
    
plt.xlim([0.0,ISimIMPES.model.fLength])
plt.legend(title='Time [min]')
plt.xlabel('Distance [m]')
plt.ylabel('Saturation $s_w$')
plt.show()


#V_water and V_oil in functiun of time


V_cell = Imodel.fPoro * Imodel.fDeltaX * A
V_tot = Imodel.fPoro * Imodel.fLength * A

water_volumes = []
oil_volumes = []

for sat in sat_list:
    V_w = np.sum(sat * V_cell)
    V_o = V_tot - V_w

    water_volumes.append(V_w)
    oil_volumes.append(V_o)

print(time_list, water_volumes, oil_volumes)
    
plt.plot(time_list, water_volumes, label="Water volume")
plt.plot(time_list, oil_volumes, label="Oil volume")

plt.xlabel("Time [s]")
plt.ylabel("Volume [m³]")
plt.legend()
plt.show()

#cumulative oil production

plt.plot(time_list, cum_oil_list, label="Cumulative oil volume")

plt.xlabel("Time [s]")
plt.ylabel("Produced oil volume [m³]")
plt.legend()
plt.show()