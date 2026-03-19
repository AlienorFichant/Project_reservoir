import numpy as np
import matplotlib.pyplot as plt
from bl_oving import CModel   

class CSimulator1DIMPES:
    def __init__(self, model):
        self.model = model

    def dofTimestep(self):
        afMobilityNonWet = self.model.tRelpermNonWet(self.model.afSaturation)/self.model.fNonWetViscosity
        afMobilityWet = self.model.tRelpermWet(self.model.afSaturation)/self.model.fWetViscosity

        afWeightedMobNonWet = afMobilityNonWet[:-1]*self.model.fMobilityWeighting + afMobilityNonWet[1:]*(1-self.model.fMobilityWeighting)
        afWeightedMobWet = afMobilityWet[:-1]*self.model.fMobilityWeighting + afMobilityWet[1:]*(1-self.model.fMobilityWeighting)

        afTransNonWet = self.model._Tran*afWeightedMobNonWet
        afTransWet = self.model._Tran*afWeightedMobWet
        fNonWetTransRight = self.model._TranRight*afMobilityNonWet[-1]
        fTransWetRight = self.model._TranRight*afMobilityWet[-1]
        afTotalTrans = afTransNonWet + afTransWet
        fTotalTransRight = fNonWetTransRight + fTransWetRight

        matrixA = np.zeros((self.model.iNumCells,self.model.iNumCells))

        matrixA[0,0] = -afTotalTrans[0]
        matrixA[0,1] = afTotalTrans[0]

        for ii in np.arange(1,self.model.iNumCells-1):
            matrixA[ii,ii-1] = afTotalTrans[ii-1]
            matrixA[ii,ii] = -afTotalTrans[ii-1]-afTotalTrans[ii]
            
            matrixA[ii,ii+1] = afTotalTrans[ii]

        matrixA[-1,-2] = afTotalTrans[-1]
        matrixA[-1,-1] = -2*fTotalTransRight - afTotalTrans[-1]

        vectorE = np.zeros(self.model.iNumCells)
        vectorE[0] = -self.model.fqt/self.model.fDeltaX
        vectorE[-1] = -2.0*fTotalTransRight*self.model.fRightPressure

        afPressure = np.linalg.solve(matrixA, vectorE)

        dtOverafPoro = self.model.fDeltaT/self.model.afPoro
        self.model.afSaturation[1:-1] = self.model.afSaturation[1:-1] - dtOverafPoro[1:-1]*(afTransNonWet[1:]*(afPressure[2:]-afPressure[1:-1]) + afTransNonWet[:-1]*(afPressure[:-2]-afPressure[1:-1]))
        self.model.afSaturation[0] = self.model.afSaturation[0] - dtOverafPoro[0]*afTransNonWet[0]*(afPressure[1] - afPressure[0])
        self.model.afSaturation[-1] = self.model.afSaturation[-1] + dtOverafPoro[-1]*(2*fTransWetRight*(self.model.fRightPressure-afPressure[-1])-afTransWet[-1]*(afPressure[-1]-afPressure[-2]))

        maxsat = 1.0-self.model.tRelpermNonWet.fSnr
        minsat = self.model.tRelpermNonWet.fSwirr
        self.model.afSaturation[ self.model.afSaturation>maxsat ] = maxsat
        self.model.afSaturation[ self.model.afSaturation<minsat ] = minsat

        self.model.afPressure = afPressure
        self.model.fTime = self.model.fTime + self.model.fDeltaT

    def simulateTo(self,fTargetTime):

        basefDeltaT = self.model.fDeltaT
        while self.model.fTime < fTargetTime:
            if self.model.fTime + basefDeltaT >= fTargetTime:
                self.model.fDeltaT = fTargetTime - self.model.fTime
                self.dofTimestep()
                self.model.fDeltaT = basefDeltaT
                self.model.fTime = fTargetTime
            else:
                self.dofTimestep() 


Imodel=CModel()
ISimIMPES = CSimulator1DIMPES(Imodel)
fMaxtime = 500*24*60*60
fReportInterval = 100*24*60*60
cmap = plt.get_cmap('gnuplot')
while ISimIMPES.model.fTime<fMaxtime:
    color = color=cmap(1-ISimIMPES.model.fTime/fMaxtime)
    ISimIMPES.simulateTo(ISimIMPES.model.fTime + fReportInterval)
    plt.plot(ISimIMPES.model.afxValues,ISimIMPES.model.afSaturation,color=color,label=str(int(ISimIMPES.model.fTime/(24*60*60))))
    plt.xlim([0.0,ISimIMPES.model.fL])
plt.title("IMPES")
plt.legend(title='Time [days]')
plt.xlabel('Distance [m]')
plt.ylabel('Saturation $s_w$')
plt.show()






