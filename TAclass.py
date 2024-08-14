# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 20:46:34 2021

@author: Bozark
"""
#Things To do for OTA class: store wavelength, time, intensity, statistics for
#given wavlengths? What about noise filtering? Anything that can be done with 
#a single dataset, so no dataset or std dev, but GVD correction, t0 alignment,
#bad time point removal, kinetic and spectral traces
import numpy as np
import math, re
from numpy.lib.scimath import sqrt
import KineticFitFunctions as kff
from copy import deepcopy

class TAdata():
    def __init__(self, TAFile, WLFile, WLuFile):
        self.Ave=False
    
        TAData=np.loadtxt(TAFile,skiprows=11)
        TimeArray= np.around(TAData[0,1:],3)
        #SortTIndex = np.argsort(TimeArray)
        self.Time=TimeArray
        self.Wavelength= np.around(TAData[1:,0],2)
        self.Intensity = np.array(TAData[1:,1:])
        if re.search('\.',TAFile)!=None:
            self.name = re.split('/|\.',TAFile)[-2]
        else:
            self.name = re.split('/',TAFile)[-1]
        self.Kinetic = {}
        self.T_slice = {}
        self.FitResults = {}
        #SortOTA = np.zeros(np.shape(self.Intensity))
        #np.put_along_axis(SortOTA,(0,100),SortTIndex,axis=0)    
        if WLFile != "":
            WlData=np.loadtxt(WLFile,skiprows=11)
            self.WLTime= np.sort(np.around(WlData[0,1:],3))
            self.WLWavelength= np.around(WlData[1:,0],2)
            self.WLIntensity = WlData[1:,1:]            
            
        if WLuFile != "":
            WluData=np.loadtxt(WLuFile,skiprows=11)
            self.WLUTime= np.sort(np.around(WluData[0,1:],3))
            self.WLUWavelength= np.around(WluData[1:,0],2)
            self.WLUIntensity = WluData[1:,1:]   
        
    def KineticTrace(self,Wvlength,binW):
        WL = []
        label  = str(Wvlength)
        WL = [i for i,x in enumerate(self.Wavelength) if math.isclose(x, Wvlength,rel_tol=(binW/Wvlength))]
        self.Kinetic[label]=np.sum(self.Intensity[WL,:].copy(),axis=0)/len(WL)
    
    def SpectralTrace(self,Tpoint,binW):
        Tslice = []
        label = str(Tpoint)
        Tslice = [i for i,x in enumerate(self.Time) if math.isclose(x, Tpoint,rel_tol=(np.abs(binW/Tpoint)))]
        self.T_slice[label] = np.sum(self.Intensity[:,Tslice].copy(),axis=1)/len(Tslice)

    def TimeZero(self,wavelength):
        ZeroCorrect = kff.t_zero(self,wavelength)
        self.Time -= ZeroCorrect
    
    def BckgSub(self):
        TmpArr = self.Intensity[:,0:10]
        TmpMean = np.mean(TmpArr,axis=1)
        self.Intensity = self.Intensity - TmpMean[:,None]
        
class AveTAdata():
    def __init__(self,TAdata,Probe):
        self.Ave=True
        self.Intensity = np.zeros(TAdata[0].Intensity.shape)
        
        Probe_i = []
        Probe_i.append(np.where(TAdata[0].Wavelength>Probe[0])[0][0])
        Probe_i.append(np.where(TAdata[0].Wavelength<Probe[1])[0][-1])
        
        self.Wavelength = TAdata[0].Wavelength[Probe_i[0]:Probe_i[1]]
        self.Time = np.zeros(TAdata[0].Time.shape)
        for i,data in enumerate(TAdata):
            try:
                self.Time = np.add(data.Time,self.Time)
            except:
                self.Error = "Improper, or unequal, OTA data"
        
        TmpStdArray = np.dstack((TAdata[i].Intensity[Probe_i[0]:Probe_i[1],:] for i in range(len(TAdata))))
        TmpArr = np.nanstd(TmpStdArray,axis = 0)
        StdMean = np.nanmean(TmpArr)
        Outlier = np.broadcast_to(TmpArr[None,:,:], TmpStdArray.shape)
        TmpStdArray = np.where(Outlier>3*StdMean, np.nan,TmpStdArray)
        
        self.Intensity = np.nanmean(TmpStdArray,axis=2)
        self.Std = np.nanstd(TmpStdArray,axis = 2)
        self.Time /= len(TAdata)
        self.Kinetic = {}
        self.T_slice = {}
        self.KineticStd = {}
        self.T_sliceStd = {}
        self.FitResults = {}
        self.KineticFit = {}
        self.name = TAdata[0].name
    
    def KineticTrace(self,Wvlength,binW):
        WL = []
        label  = str(Wvlength)
        WL = [i for i,x in enumerate(self.Wavelength) if math.isclose(x, Wvlength,rel_tol=(binW/Wvlength))]
        self.Kinetic[label]=np.sum(self.Intensity[WL,:].copy(),axis=0)/len(WL)
        self.KineticStd[label]=np.sqrt(np.sum(np.square(self.Std[WL,:].copy()),axis=0))/sqrt(len(WL))
    
    def SpectralTrace(self,Tpoint,binW):
        Tslice = []
        label = str(Tpoint)
        Tslice = [i for i,x in enumerate(self.Time) if math.isclose(x, Tpoint,rel_tol=(np.abs(binW/Tpoint)))]
        self.T_slice[label] = np.sum(self.Intensity[:,Tslice].copy(),axis=1)/len(Tslice)
        self.T_sliceStd[label]=np.sqrt(np.sum(np.square(self.Std[:,Tslice].copy()),axis=1))/sqrt(len(Tslice))
        
    def TimeZero(self,wavelength):
        ZeroCorrect = kff.t_zero(self,wavelength)
        self.Time -= ZeroCorrect
        
class GVDdata():
    def __init__(self,Intensity,Std,TAdata):
        self.Ave = True
        self.Intensity = Intensity
        self.Wavelength = TAdata.Wavelength
        self.Time = TAdata.Time
        self.Std = Std
        self.Kinetic = {}
        self.T_slice = {}
        self.KineticStd = {}
        self.T_sliceStd = {}
        self.FitResults = {}
        self.KineticFit = {}
        self.name = "GVD" + TAdata.name
    
    def KineticTrace(self,Wvlength,binW):
        WL = []
        label  = str(Wvlength)
        WL = [i for i,x in enumerate(self.Wavelength) if math.isclose(x, Wvlength,rel_tol=(binW/Wvlength))]
        self.Kinetic[label]=np.sum(self.Intensity[WL,:].copy(),axis=0)/len(WL)
        self.KineticStd[label]=np.sqrt(np.sum(np.square(self.Std[WL,:].copy()),axis=0))/sqrt(len(WL))
    
    def SpectralTrace(self,Tpoint,binW):
        Tslice = []
        label = str(Tpoint)
        Tslice = [i for i,x in enumerate(self.Time) if math.isclose(x, Tpoint,rel_tol=(binW/Tpoint))]
        self.T_slice[label] = np.sum(self.Intensity[:,Tslice].copy(),axis=1)/len(Tslice)
        self.T_sliceStd[label]=np.sqrt(np.sum(np.square(self.Std[:,Tslice].copy()),axis=1))/sqrt(len(Tslice))
        
class FitOTAdata():
     def __init__(self,TAdata,fit_type='spectral'):
         self.Ave = True
         self.Intensity = deepcopy(TAdata.Intensity)
         self.Wavelength = TAdata.Wavelength
         self.Time = TAdata.Time
         self.Std = deepcopy(TAdata.Std)
         self.Kinetic = {}
         self.T_slice = {}
         self.KineticStd = {}
         self.T_sliceStd = {}
         self.FitResults = {}
         self.KineticFit = {}
         self.fit_type = fit_type
         self.fit_params = []
     
     def KineticTrace(self,Wvlength,binW):
         WL = []
         label  = str(Wvlength)
         WL = [i for i,x in enumerate(self.Wavelength) if math.isclose(x, Wvlength,rel_tol=(binW/Wvlength))]
         self.Kinetic[label]=np.sum(self.Intensity[WL,:].copy(),axis=0)/len(WL)
         self.KineticStd[label]=np.sqrt(np.sum(np.square(self.Std[WL,:].copy()),axis=0))/sqrt(len(WL))
     
     def SpectralTrace(self,Tpoint,binW):
         Tslice = []
         label = str(Tpoint)
         Tslice = [i for i,x in enumerate(self.Time) if math.isclose(x, Tpoint,rel_tol=(np.abs(binW/Tpoint)))]
         self.T_slice[label] = np.sum(self.Intensity[:,Tslice].copy(),axis=1)/len(Tslice)
         self.T_sliceStd[label]=np.sqrt(np.sum(np.square(self.Std[:,Tslice].copy()),axis=1))/sqrt(len(Tslice))
         
     def SpectralFit(self,Tpoint,binW):
         Tslice = []
         Tslice = [i for i,x in enumerate(self.Time) if math.isclose(x, Tpoint,rel_tol=(binW/Tpoint))]
         signal = np.sum(self.Intensity[:,Tslice].copy(),axis=1)/len(Tslice)
         std =np.sqrt(np.sum(np.square(self.Std[:,Tslice].copy()),axis=1))/sqrt(len(Tslice))
         return signal,std