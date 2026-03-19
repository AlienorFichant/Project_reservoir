import numpy as np
import matplotlib.pyplot as plt
def normSw(fSw,fSwirr,fSnr):
    return (fSw-fSwirr)/(1.0-fSnr-fSwirr)

class CCoreyNonWetting:

    def __init__(self,fNn,fSwirr,fSnr):
        self.fNn = fNn
        self.fSwirr = fSwirr
        self.fSnr = fSnr
    def __call__(self,afSw):
        afnSw = normSw(afSw,self.fSwirr,self.fSnr)
        return (1.0-afnSw)**self.fNn
class CCoreyWetting:

    def __init__(self,fNw,fKrwn,fSwirr,fSnr):
        self.fNw = fNw
        self.fKrwn = fKrwn
        self.fSwirr = fSwirr
        self.fSnr = fSnr
    def __call__(self,afSw):
        afnSw = normSw(afSw,self.fSwirr,self.fSnr)
        return self.fKrwn*afnSw**self.fNw
class CModel:
    def __init__(self):
        self.fNn = 2
        self.fNw = 2
        self.fKrwn = 0.4
        self.fSwirr = 0.2
        self.fSnr = 0.29 #Sor
        self.fphi = 0.23 #porosity
        self.fPerm = 2.43e-13 #liquid permeability
        self.fqt = 1/(100*60) #injection rate?
        self.fSwi = 0.19 #from sheet
        self.fL = 0.057 #length of core sample
        self.fTime = 0.0
        self.iNumCells = 100
        self.fDeltaT = 3*60*60 #3hours
        self.fRightPressure = 1.0E7
        self.fMobilityWeighting = 1.0
        self.fNonWetViscosity = 1.03e-3 
        self.fWetViscosity = 0.96e-3
        self.setParameters()
    
    def krw(self,x):
        return self.fKrwn*x*self.fNw
    def krn(self,x):
        return (1-x)**self.fNn
    def fw(self,x):
        lw = self.krw(x) / self.fWetViscosity
        ln = self.krn(x) / self.fNonWetViscosity
        return lw / (lw + ln)
    
    def numDerivativeCentered(self,f, x, dx=1e-6):
        return (f(x + dx) - f(x - dx)) / (2*dx)
    
    def normSw(self,fSw):
        return (fSw-self.fSwirr)/(1.0-self.fSnr-self.fSwirr)
    def setPermeabilities(self,permVector):
        self._perm = permVector
        self._Tran = (2.0/(1.0/self._perm[:-1]+1.0/self._perm[1:]))/self.fDeltaX**2
        self._TranRight = self._perm[-1]/self.fDeltaX**2
   
    def setParameters(self):
        self.fDeltaX = self.fL/self.iNumCells
        self.afxValues = np.arange(self.fDeltaX/2,self.fL,self.fDeltaX)
        self.afPoro = self.fphi*np.ones(self.iNumCells)

        self._perm = self.setPermeabilities(self.fPerm*np.ones(self.iNumCells))

        self.afPressure = self.fRightPressure*np.ones(self.iNumCells)
        self.afSaturation = self.fSwirr*np.ones(self.iNumCells)
        self.tRelpermWet = CCoreyWetting(self.fNw,self.fKrwn,self.fSwirr,self.fSnr)
        self.tRelpermNonWet = CCoreyNonWetting(self.fNn,self.fSwirr,self.fSnr)


    def fractionalFlowCorey(self,fSw):
        return 1/(1+(1-fSw)**self.fNn*self.fWetViscosity/(self.fKrwn*fSw**self.fNw*self.fNonWetViscosity))

    def findFrontSaturation(self):
        from scipy import optimize
        def negativeSlopeFFCorey(fSw):
            return -self.fractionalFlowCorey(self.normSw(fSw))/(fSw-self.fSwirr)
        fSwf=optimize.fmin(negativeSlopeFFCorey,((1.0-self.fSnr)-self.fSwirr)/2.0)
        return fSwf[0]
    
    def buckleyLeverettSolution(self,fTime):
        fSwf=self.findFrontSaturation()
        def funcFFCorey(fSw):
            return self.fractionalFlowCorey(self.normSw(fSw))
        afSw=np.arange(fSwf,1.0-self.fSnr,0.001)
        afDerivativeFFCorey=self.numDerivativeCentered(funcFFCorey,afSw,dx=1E-6)
        afDerFFDarcyVeloPhi=afDerivativeFFCorey*self.fqt/self.fphi
        return afDerFFDarcyVeloPhi*fTime,afSw
    
    def plot(self):
        cmap = plt.get_cmap('gnuplot')
        fTime=0.0
        fDeltaT=60*60*24*100
        fMaxTime=60*60*24*1000
        while(fTime<fMaxTime):
            fTime=fTime+fDeltaT
            afDistance,afSw=self.buckleyLeverettSolution(fTime)
            plt.plot(afDistance,afSw,color=cmap(1-fTime/fMaxTime))
            plt.plot([afDistance[0],afDistance[0]],[afSw[0],self.fSwirr],color= cmap(1-fTime/fMaxTime))
            plt.plot([afDistance[0],self.fL],[self.fSwirr,self.fSwirr],color=cmap(1-fTime/fMaxTime))
        plt.title("Buckley-Leverett")
        plt.xlabel(r'Distance [m]')
        plt.ylabel(r'Saturation $s_w$')
        plt.show()
run = CModel()
run.setParameters()
run.plot()
