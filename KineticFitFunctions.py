# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:26:57 2022

@author: Bozark
"""
#module holding kinetic fits
import sympy as sym
import sympy.utilities.lambdify as lambdify
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters,fit_report
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.stats import lognorm
from scipy.integrate import quad
from numpy.lib.scimath import sqrt,log
import io
import TAplotter as TAplt
import inspect as ins

def fitInit(comp=1):
    """
    Parameters
    ----------
    comp : TYPE, optional
        The default is 1, and describes the sum of exponential fitting funct.
        Pass in comp=0 for the t_zero function.

    Returns
    -------
    fncDict : TYPE
        A dictionary of the default values for any given fitting function.
        Can be amended and passed in to manyFitKinetic in place of kwargs with "**fncDict"

    """
    if comp==0:
        signature = ins.signature(t_zero)
    elif comp==1:
        signature = ins.signature(oneExpFit)
    elif comp==2:
        signature = ins.signature(twoExpFit)
    elif comp==3:
        signature = ins.signature(threeExpFit)
    elif comp==4:
        signature = ins.signature(fourExpFit)
    elif comp==5:
        signature = ins.signature(stretchExpFit)

    fncDict =  {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not ins.Parameter.empty
    }
    
    fncDict.pop('quiet')
    
    return fncDict

def oneExpFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,A1 = -0.0005,
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True, quiet = False,
              time_mask=[[]]):
    
    label = str(wvlngth)
    if isinstance(time_mask[0],list) and len(time_mask[0])>0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    data = data[nan_idx]
    if OTAData.Ave:
        fit_std = fit_std[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value=A1, vary=varyA1)
    params.add('t0',value=t0, vary=varyt0)
    params.add('rise',value=rise,vary=False)
        
    if OTAData.Ave:
        results = minimize(oneExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(oneExpResidual,params,args=(time,data),nan_policy='omit')

    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = OTAData.Time
    if rise:
        OTAData.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))
    else:
        OTAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time)
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def oneExpResidual(params,time,data,std=[]):
    parvals = params.valuesdict()
    
    # time_arr = np.linspace(-50,100,9001)
    # arr_0 = np.where(time_arr==0)[0][0]
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model = A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))
    else:
        model = A1*erfexp(C1,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def twoExpFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,\
              A1 = -0.0005,C2 = 10,A2 = -0.0005, time_mask = [[]],
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True,
              varyC2=True,varyA2=True,quiet = False):
    
    label = str(wvlngth)
    if isinstance(time_mask[0],list) and len(time_mask[0])>0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('t0',value= t0, vary=varyt0)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value= A1, vary=varyA1)
    params.add('C2', value= C2, min=0.1, vary=varyC2)
    params.add('A2',value= A2, vary=varyA2)
    params.add('rise',value=rise,vary=False)
        
    #resid = Residual(params,A,B,C, ZnOkinetic,UVTime,QDkinetic,VisTime)
    if OTAData.Ave:
        results = minimize(twoExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(twoExpResidual,params,args=(time,data),nan_policy='omit')

    if not quiet:
       print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = OTAData.Time
    if rise:
        OTAData.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))
    else:
        OTAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def twoExpResidual(params,time,data,std = []):
    parvals = params.valuesdict()
    
    # time_arr = np.linspace(-50,100,9001)
    # arr_0 = np.where(time_arr==0)[0][0]
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model = A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))+A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))
    else:
        model = A1*erfexp(C1,IRF,t0,time)+A2*erfexp(C2,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def threeExpFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,\
              A1 = -0.0005,C2 = 10,A2 = -0.0005, C3 = 500,A3 = -0.0005,time_mask = [[]], 
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True,
              varyC2=True,varyA2=True,varyC3=True,varyA3=True,quiet=False):
    
    label = str(wvlngth)
    if isinstance(time_mask[0],list) and len(time_mask[0])>0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('t0',value= t0, vary=varyt0)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('C2', value= C2, min=0.1, vary=varyC2)
    params.add('C3', value= C3, min=0.1, vary=varyC3)
    if  (type(A1)==str):
        params.add('A1',expr= A1, vary=varyA1)
    else:
        params.add('A1',value= A1, vary=varyA1)
    if  (type(A2)==str):
        params.add('A2',expr= A2, vary=varyA2)
    else:
        params.add('A2',value= A2, vary=varyA2)
    if  (type(A3)==str):
        params.add('A3',expr= A3, vary=varyA3)
    else:
        params.add('A3',value= A3, vary=varyA3)
    
    params.add('rise',value=rise,vary=False)
        
    if OTAData.Ave:
        results = minimize(threeExpResidual,params,args=(time,data,fit_std))
        # results = minimize(twoplusTPAResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(threeExpResidual,params,args=(time,data),nan_policy='omit')

    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    C3 = parvals['C3']
    A3 = parvals['A3']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = OTAData.Time
    if rise:
        OTAData.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time)) 
    else:
        OTAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time)
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def threeExpResidual(params,time,data,std = []):
    parvals = params.valuesdict()
    
    # time_arr = np.linspace(-50,100,9001)
    # arr_0 = np.where(time_arr==0)[0][0]
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    C3 = parvals['C3']
    A3 = parvals['A3']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time))
    else:
        model= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data
    
def fourExpFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,
              A1 = -0.0005,C2 = 10,A2 = -0.0005, C3 = 500,A3 = -0.0005,
              A4 = -0.001,C4 = 1000, time_mask = [[]], quiet=False):
    
    label = str(wvlngth)
    if isinstance(time_mask[0],list) and len(time_mask[0])>0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=False)
    params.add('t0',value= t0, vary=False)
    params.add('C1', value= C1, min=0.1, vary=False)
    params.add('A1',value= A1,max=0, vary=True)
    params.add('C2', value= C2, min=0.1, vary=True)
    params.add('A2',value= A2,max=0, vary=True)
    params.add('C3', value= C3, min=0.1, vary=True)
    params.add('A3',value= A3,max=0, vary=True)
    params.add('C4', value= C4, min=0.1, vary=True)
    params.add('A4',value= A4,min=0, vary=True)
    params.add('rise',value=rise,vary=False)
        
    if OTAData.Ave:
        results = minimize(twoplusTPAResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(threeExpResidual,params,args=(time,data),nan_policy='omit')

    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    C3 = parvals['C3']
    A3 = parvals['A3']
    C4 = parvals['C4']
    A4 = parvals['A4']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = OTAData.Time
    if rise:
        OTAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time)\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time))\
                    +A4*(erfexp(1e6,IRF,t0,time) -erfexp(C4, IRF, t0, time))
    else:
        OTAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time)
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results
    
def twoplusTPAResidual(params,time,data,std = []):
    parvals = params.valuesdict()
    
    # time_arr = np.linspace(-50,100,9001)
    # arr_0 = np.where(time_arr==0)[0][0]
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    C3 = parvals['C3']
    A3 = parvals['A3']
    C4 = parvals['C4']
    A4 = parvals['A4']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model= A1* erfexp(C1,IRF,t0,time)\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time))\
                    +A4*(erfexp(1e6,IRF,t0,time) -erfexp(C4, IRF, t0, time))
    else:
        model= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(std),1,std)
        return np.divide((model-data),tmpStd)
    else:
        return model-data    

def stretchExpFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,A1 = -0.0005,B = 1,
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True, quiet = False,varyB=True,
              time_mask=[]):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value=A1, vary=varyA1)
    params.add('t0',value=t0, vary=varyt0)
    params.add('B',value=B, min=0.01,vary=varyB)
    params.add('rise',value=rise,vary=False)
        
    if OTAData.Ave:
        results = minimize(stretchExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(stretchExpResidual,params,args=(time,data),nan_policy='omit')

    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = OTAData.Time
    if rise:
        OTAData.KineticFit[label] = np.where(time<0,A1*(erfexp(1e6, IRF, t0, time)-erfexp(C1,IRF,t0,time)),
                         A1*(stretcherfexp(1e6,IRF,t0,B,time) - stretcherfexp(C1,IRF,t0,B,time)))
    else:
        OTAData.KineticFit[label] = np.where(time<0,A1*erfexp(C1,IRF,t0,time),
                         A1*stretcherfexp(C1,IRF,t0,B,time))
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def stretchExpResidual(params,time,data,std=[]):
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    B = parvals['B']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model = np.where(time<0,A1*(erfexp(1e6, IRF, t0, time)-erfexp(C1,IRF,t0,time)),
                         A1*(stretcherfexp(1e6,IRF,t0,B,time) - stretcherfexp(C1,IRF,t0,B,time)))
    else:
        model = np.where(time<0,A1*erfexp(C1,IRF,t0,time),
                         A1*stretcherfexp(C1,IRF,t0,B,time))

    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def distExpFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0,
               varyIRF=False,varyt0=False,quiet = False,time_mask=[]):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    C_arr = np.logspace(0.5, 5000,num=20)
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    for idx in range(len(C_arr)):
        params.add('A'+str(idx),value=1/len(C_arr),vary=True)
    params.add('t0',value=t0, vary=varyt0)
    params.add('rise',value=rise,vary=False)
        
    if OTAData.Ave:
        results = minimize(distExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(distExpResidual,params,args=(time,data),nan_policy='omit')

    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    model = np.zeros(time.shape)
    
    time = OTAData.Time
    OTAData.KineticFit[label] = np.zeros(time.shape)
    if rise:
        for idx,tau in enumerate(C_arr):
            OTAData.KineticFit[label].add(parvals['A'+str(idx)]*(erfexp(1e6,IRF,t0,time) - erfexp(tau,IRF,t0,time)))
    else:
        for idx,tau in enumerate(C_arr):
            OTAData.KineticFit[label].add(parvals['A'+str(idx)]*erfexp(tau,IRF,t0,time))
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def distExpResidual(params,time,data,std=[]):
    parvals = params.valuesdict()
    
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    C_arr = np.logspace(0.5, 5000,num=20)
    model = np.zeros(time.shape)
    
    if rise:
        for idx,tau in enumerate(C_arr):
            model.add(parvals['A'+str(idx)]*(erfexp(1e6,IRF,t0,time) - erfexp(tau,IRF,t0,time)))
    else:
        for idx,tau in enumerate(C_arr):
            model.add(parvals['A'+str(idx)]*erfexp(tau,IRF,t0,time))
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(std),1,std)
        return np.divide((model-data),tmpStd)
    else:
        return model-data
    
def lognormFit(OTAData,wvlngth, rise = False,IRF = 0.13, t0 = 0, C1 = 1,A1 = -0.0005,
               sigma=1,varysigma=True,plotInit=False,
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True, quiet = False,
              time_mask=[]):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    data = data[nan_idx]
    if OTAData.Ave:
        fit_std = fit_std[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value=A1, vary=varyA1)
    params.add('sigma',value=sigma,min=0.1,max=10,vary=varysigma)
    params.add('t0',value=t0, vary=varyt0)
    params.add('rise',value=rise,vary=False)
        
    if plotInit:
        dist_k = np.linspace(0.1,1000.1,5000)
        dist_tau = lognorm.pdf(dist_k,sigma,loc=C1)
        norm_k=dist_tau.sum()
        dist_tau/=norm_k
        
        if rise:
            model = np.zeros(time.shape)
            for c,k in zip(dist_k,dist_tau):
                tmp_arr = k*(erfexp(1e6,IRF,t0,time)-erfexp(c,IRF,t0,time))
                model += tmp_arr
        else:
            model = np.zeros(time.shape)
            for c,k in zip(dist_k,dist_tau):
                tmp_arr = k*erfexp(c,IRF,t0,time)
                model += tmp_arr
                
        model*=A1
        fig,ax = plt.subplots()
        ax.plot(time,model)
        ax.plot(time,data)
    
    if OTAData.Ave:
        results = minimize(lognormResid,params,args=(time,data,fit_std),nan_policy='raise')
    else: 
        results = minimize(lognormResid,params,args=(time,data),nan_policy='raise')

    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    sigma=parvals['sigma']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    dist_k = np.linspace(0.1,1000.1,5000)
    dist_tau = lognorm.pdf(dist_k,sigma,loc=C1)
    norm_k=dist_tau.sum()
    dist_tau/=norm_k
    
    time = OTAData.Time
    
    if rise:
        model = np.zeros(time.shape)
        for c,k in zip(dist_k,dist_tau):
            tmp_arr = k*(erfexp(1e6,IRF,t0,time)-erfexp(c,IRF,t0,time))
            model += tmp_arr
    else:
        model = np.zeros(time.shape)
        for c,k in zip(dist_k,dist_tau):
            tmp_arr = k*erfexp(c,IRF,t0,time)
            model += tmp_arr
            
    OTAData.KineticFit[label] = A1*model
    OTAData.FitResults[label]=results
    
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    fig,ax = plt.subplots()
    ax.plot(dist_k,dist_tau)
    
    return results

def lognormResid(params,time,data,std=[]):
    
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    # k = parvals['k']
    sigma = parvals['sigma']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    dist_k = np.linspace(0.1,1000.1,5000)
    dist_tau = lognorm.pdf(dist_k,sigma,loc=C1)
    norm_k=dist_tau.sum()
    dist_tau/=norm_k
    
    if rise:
        model = np.zeros(time.shape)
        for c,k in zip(dist_k,dist_tau):
            tmp_arr = k*(erfexp(1e6,IRF,t0,time)-erfexp(c,IRF,t0,time))
            model += tmp_arr
    else:
        model = np.zeros(time.shape)
        for c,k in zip(dist_k,dist_tau):
            tmp_arr = k*erfexp(c,IRF,t0,time)
            model += tmp_arr
            
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((A1*model-data),tmpStd)
    else:
        return A1*model-data

def t_zero(OTAData,wvlngth,A1=-0.005,IRF = 0.13 , t0 = 0, C1 = 0.1,rise = False,quiet=False,
              varyC1=True,varyIRF=False,varyt0=True,varyA1=True,time_mask=[[]]):
    
    label = str(wvlngth)
    if isinstance(time_mask[0],list) and len(time_mask[0])>0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label]
    
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    data = data[nan_idx]
    
    if OTAData.Ave:
        fit_std = fit_std[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value= A1, vary=varyA1)
    params.add('t0',value= t0, vary=varyt0)
    params.add('rise',value= rise, vary=False)
    
    if OTAData.Ave:    
        results = minimize(oneExpResidual,params,args=(time,data,fit_std),nan_policy='raise')
    else:
        results = minimize(oneExpResidual,params,args=(time,data),nan_policy='raise')
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    single_fit = A1*erfexp(C1,IRF,t0,time)
    
    if not quiet:
        fig,ax = plt.subplots()
        ax.plot(time,single_fit,'b',linewidth = 2.0,label='t0 fit')
        ax.plot(time,data,'c',linewidth = 2.0,label='data')
        ax.legend(loc=3)
        plt.show()
    
    return t0

def convolve_ex(params,time,data):
    parvals = params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    
    time_arr = np.linspace(-50,100,9001)
    arr_0 = np.where(time_arr==0)[0][0]
    
    gauss =1/sqrt(np.pi*2)*IRF/std_FWHM_const* np.exp(-time**2/2/(IRF/std_FWHM_const)**2)
    gauss =1/sqrt(np.pi*2)/(IRF/std_FWHM_const)* np.exp(-time_arr**2/2/(IRF/std_FWHM_const)**2)
    tindex=[arr_0, arr_0+time_arr.size]
    heavi=np.heaviside(time,0.5)
    KB = np.split(np.convolve(heavi, gauss),tindex)[1]
    KB =KB /KB.max()
    
    fig = plt.figure()
    fig,ax = plt.subplots()
    # # ax = bax.brokenaxes(xlims=((-5,5),(20,7000)),wspace=.05,width_ratios=[2,1])
    # ax.plot(time_arr,model,'b',linewidth = 2.0,label='ERFEXP')
    ax.plot(time_arr,KB,'c-',linewidth = 2.0,label='testConvolve')
    
    return KB

def stretchCustFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,A1 = -0.0005,B = 1,
                   C2=1000,A2=0.001,varyC2=True,varyA2=True,
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True, quiet = False,varyB=True,
              time_mask=[]):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value=A1, vary=varyA1)
    params.add('C2', value= C2, min=0.1, vary=varyC2)
    params.add('A2',value=A2, vary=varyA2)
    params.add('t0',value=t0, vary=varyt0)
    params.add('B',value=B, min=0.01,vary=varyB)
    params.add('rise',value=rise,vary=False)
        
    if OTAData.Ave:
        results = minimize(stretchCustResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(stretchCustResidual,params,args=(time,data),nan_policy='omit')

    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    B = parvals['B']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = OTAData.Time
    if rise:
        OTAData.KineticFit[label] = np.where(time<0,A1*(erfexp(1e6, IRF, t0, time)-erfexp(C1,IRF,t0,time)),
                         A1*(stretcherfexp(1e6,IRF,t0,B,time) - stretcherfexp(C1,IRF,t0,B,time)))+ \
            A2*(erfexp(1e6,IRF,t0,time)-erfexp(C2, IRF, t0, time))
    else:
        OTAData.KineticFit[label] = np.where(time<0,A1*erfexp(C1,IRF,t0,time),
                         A1*stretcherfexp(C1,IRF,t0,B,time)) +\
            A2*erfexp(C2, IRF, t0, time)
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def stretchCustResidual(params,time,data,std=[]):
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    B = parvals['B']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model = np.where(time<0,A1*(erfexp(1e6, IRF, t0, time)-erfexp(C1,IRF,t0,time)),
                         A1*(stretcherfexp(1e6,IRF,t0,B,time) - stretcherfexp(C1,IRF,t0,B,time))) +\
                A2*(erfexp(1e6,IRF,t0,time)-erfexp(C2, IRF, t0, time))
    else:
        model = np.where(time<0,A1*erfexp(C1,IRF,t0,time),
                         A1*stretcherfexp(C1,IRF,t0,B,time)) +\
                A2*erfexp(C2, IRF, t0, time)

    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(std),1,std)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def stretcherfexp(tau,IRF,t0, B,t):
    
    val = np.where((t-t0)>-5, np.exp((IRF/1.65511/2/tau)**2-((t-t0)/tau)**B)*\
           (erf((t-t0)/IRF*1.65511-(IRF/1.65511/2/tau))+1)/2,0)
    return val

def erfexp(tau,IRF,t0, t):
    
    val = np.where((t-t0)>-5, np.exp((IRF/1.65511/2/tau)**2-(t-t0)/tau)*\
           (erf((t-t0)/IRF*1.65511-(IRF/1.65511/2/tau))+1)/2,0)
    return val