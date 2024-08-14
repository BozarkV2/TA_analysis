# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:53:57 2023

@author: Bozark
"""
import sympy as sym
import sympy.utilities.lambdify as lambdify
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters,fit_report
from lmfit.models import GaussianModel
# import brokenaxes as bax
from scipy.interpolate import interp1d
from scipy.special import erf
from numpy.lib.scimath import sqrt,log
import math
import io
import TAplotter as TAplt
from TAclass import FitOTAdata
import copy as cp

def FitFullSpectral(OTA,wvl_mask=[[]],time_mask=[[]],quiet=0,model=None):
    
    Time = OTA.Time 
    Fit_data = FitOTAdata(OTA)
    
    if model is not None:
        fit_params = model.make_params(center=365,sigma=5,amplitude=-0.001)
    else:
        model = GaussianModel(nan_policy='omit')
        fit_params = model.make_params(center=365,sigma=5,amplitude=-0.001)
        
    wvl = OTA.Wavelength
    
    for idx,t in enumerate(Time):
        results = FitSingleSpectral(Fit_data, t, fit_params,
                                         wvl_mask=wvl_mask,quiet=1,full_fit = True)
        
        if len(wvl_mask)>0:
            Fit_data.Intensity[:,idx] = results.eval(results.params,wvl)
            Fit_data.Std[:,idx] = results.residual
        else:
            Fit_data.Intensity[:,idx] = results.best_fit
            Fit_data.Std[:,idx] = results.residual
            
        Fit_data.fit_params.append(results.params)
       
    gcenter=[]
    gamp=[]
    gwidth=[]
    for params in Fit_data.fit_params:
        gcenter.append(params['center'].value)
        gamp.append(params['amplitude'].value)
        gwidth.append(params['sigma'].value)
    
    if not quiet:
        TAplt.plotTAdata(Fit_data)
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
        ax1.plot(Time,gcenter,"r-",label="GaussCenter")
        ax2.plot(Time,gamp,"g-",label="GaussAmp")
        ax3.plot(Time,gwidth,"b-",label="GaussWidth")
        plt.legend()
    return Fit_data

def FitSingleSpectral(OTA,T_slice,params,wvl_mask=[],quiet = False,
                      full_fit = False, model=None):
    
    label = str(T_slice)
    if len(wvl_mask) >0:
        wvl_index = np.zeros(0,dtype=int)
        regions = len(wvl_mask)
        for r in range(regions):
            x,y = wvl_mask[r]
            wvl_index = np.concatenate((wvl_index,
                                        [i for i,z in enumerate(OTA.Wavelength) if z < x or z>y]))
        
        idx,cnt = np.unique(wvl_index,return_counts=True)
        wvl_index = idx[cnt == regions] 
        wvl = OTA.Wavelength[wvl_index]
    else:
        wvl_index = range(0,len(OTA.Wavelength))
        wvl = OTA.Wavelength[wvl_index]
    
    if label in OTA.T_slice:
        data = OTA.T_slice[label][wvl_index]
    elif full_fit:
        data,fit_std = OTA.SpectralFit(T_slice, T_slice/10)
    else:
        OTA.SpectralTrace(T_slice,T_slice/10)
        data = OTA.T_slice[label][wvl_index]
    
    if OTA.Ave and not full_fit:
        fit_std = OTA.T_sliceStd[label][wvl_index]

    if model is not None:
        fit_params = model.make_params(center=350,sigma=5,amplitude=-0.001)
    else:
        model = GaussianModel(nan_policy='omit')
        model.set_param_hint('center',min=300,max=400)
        model.set_param_hint('amplitude', min=-1,max=1)
        fit_params = model.make_params(center=350,sigma=5,amplitude=-0.001)
    
    results = model.fit(data,fit_params,x=wvl)
    
    if OTA.Ave and (0 not in fit_std):
        # results = model.fit(data,fit_params,weights=1/fit_std,nan_policy='omit',x=wvl)
        results = model.fit(data,fit_params,nan_policy='omit',x=wvl)
        # results = minimize(SpectralResid,params,args=(wvl,data,fit_std))
    else: 
        results = model.fit(data,fit_params,nan_policy='omit',x=wvl)
        # results = minimize(SpectralResid,params,args=(wvl,data))    
        
    if not quiet:
        print(results.fit_report())
    
    # parvals = fit_params.valuesdict()
    # gwidth = parvals['gwidth']
    # gcenter = parvals['gcenter']
    # gamp = parvals['gamp']
    # offset = parvals['offset']
    
    wvl = OTA.Wavelength
    
    if not full_fit:
        OTA.SpectralFit[label]= results.best_fit
        OTA.FitResults[label]=results
    
    if not quiet:
        plt.plot(wvl,data,'o')
        plt.plot(wvl,results.best_fit,'-')
        
    return results

def SpectralResid(params,wvl,data,std=[]):
    parvals = params.valuesdict()
    functions = parvals['functions']
    gwidth = parvals['gwidth']
    gcenter = parvals['gcenter']
    gamp = parvals['gamp']
    offset = parvals['offset']
    
    model = offset + gamp * np.exp(-(wvl - gcenter) ** 2 / (2 * gwidth ** 2))
    
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        return np.divide((model-data),tmpStd)
    else:
        return model - data
    