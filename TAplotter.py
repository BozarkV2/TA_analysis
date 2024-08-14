# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:53:05 2021

@author: Bozark
"""
import TAclass,matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import math

def plotTAdata(TAdata,color_min=-0.02,color_max=0.02,fromGUI=False):
    """
    Plots a single TAdata with variable color. If its an averaged dataset, will also plot standard deviation.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdataset.
    color_min : float, optional
        Minimum to use for color map. The default is -0.0005.
    color_max : float, optional
        Maximum to use for color map. The default is 0.0005.

    Returns
    -------
    fig : pyplot fig class
        Can be used for further alterations to figure.

    """
    Wavelength = TAdata.Wavelength
    TimeAxis = TAdata.Time
    Intensity = TAdata.Intensity
    
    xticks = range(0,len(TimeAxis),10)
    yticks = range(0,len(Wavelength),20)
    xticklabel = ["{:6.2f}".format(i) for i in TimeAxis[xticks]]
    yticklabel = ["{:6.2f}".format(i) for i in Wavelength[yticks]]
    
    if TAdata.Ave:
        fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticklabel)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(yticklabel)
    else:
        fig,ax1 = plt.subplots(1,1)
    
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabel)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabel)
    
    # if not fromGUI:
    cax = ax1.imshow(Intensity,interpolation=None,
               vmin=color_min,vmax=color_max,
               cmap='bwr',aspect='auto')
    
    fig.colorbar(cax,ax=ax1)
    
    if TAdata.Ave: #and not fromGUI:
        cax2 = ax2.imshow(TAdata.Std,vmin=0,vmax=np.max(TAdata.Std),
                   cmap='bwr',interpolation=None,aspect='auto')
        fig.colorbar(cax2,ax=ax2)
    
    return fig
    
def PlotKinetic(TAdata,wvlngth,plotFit = True, norm = False,normT = 500,plotError = True):
    """
    Takes a single TAdata, or a list of TAdata, and plots a kinetic trace. Plots any fit and error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    wvlngth : Integer
        specific wavelength to plot kinetics for.
    plotFit : Boolean, optional
        Whether to plot a fitted line, if available. The default is True.
    norm : Boolean, optional
        Whether to normalize the kinetic line to 1, at some time. The default is False.
    normT : Integer, optional
        The time in ps to normalize a kinetic trace to. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.

    """
    
    if type(TAdata)==list:
        gcf = PlotMK(TAdata, wvlngth,plotFit,norm,normT,plotError)
    else:
        label= str(wvlngth)
        
        if label in TAdata.Kinetic:
            Intensity = TAdata.Kinetic[label]
        else:
            TAdata.KineticTrace(wvlngth,1)
            Intensity = TAdata.Kinetic[label]
        Time = TAdata.Time
        
        if norm:
            norm_i = [i for i,x in enumerate(Time) if math.isclose(x, normT,rel_tol=(0.2))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
        
        if label in TAdata.FitResults and plotFit:
            fit_wvl = TAdata.KineticFit[label]
        
        if TAdata.Ave and plotError:
            error = TAdata.KineticStd[label]
        
        gcf, ax = plt.subplots()
        if TAdata.Ave and plotError:
            ax.errorbar(Time,Intensity/normFact,yerr=error/normFact,capsize=0.5,label = TAdata.name)
        else:
            ax.plot(Time,Intensity/normFact,'.',label = TAdata.name)
            
        if label in TAdata.FitResults and plotFit:
            ax.plot(Time[0:len(fit_wvl)],fit_wvl/normFact,label = TAdata.name+"fit")
        
        plt.legend()
        
    return gcf
    
def PlotSpectral(TAdata,time_slice,norm = False,normW = 500,plotError = False):
    """
    Takes a single TAdata, or a list of TAdata, and plots a spectral trace. Plots error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    time_slice : float
        time to plot spectral traces at.
    norm : Boolean, optional
        Whether to normalize the spectral trace to 1, at some wavelength. The default is False.
    normW : integer, optional
        The wavelength to normalize a spectral trace to 1 at. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.
    """
    if type(TAdata)==list:
        gcf = PlotMS(TAdata, time_slice, norm, normW,plotError)
    else:
        label= str(time_slice)
        
        if label in TAdata.T_slice:
            Intensity = TAdata.T_slice[label]
        else:
            TAdata.SpectralTrace(time_slice,time_slice/10)
            Intensity = TAdata.T_slice[label]
        
        if(TAdata.Ave):
            error = TAdata.T_sliceStd[label]
    
        Wavelengths = TAdata.Wavelength
        gcf, ax = plt.subplots()
        
        if norm:
            norm_i = [i for i,x in enumerate(Wavelengths) if math.isclose(x, normW,rel_tol=(0.01))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
            
        if TAdata.Ave and plotError:
            error = TAdata.T_sliceStd[label]
            
        if TAdata.Ave and plotError:
            ax.errorbar(Wavelengths,Intensity/normFact,yerr=error/normFact,capsize=0.5,label =  TAdata.name)
        else:
            ax.plot(Wavelengths,Intensity/normFact,'r.',label = TAdata.name)
            
    return gcf

def PlotMK(TAdata,wvlngth,plotFit = False, norm = False,normT = 500,plotError = False):
    """
    Takes a list of TAdata and plots a kinetic trace. Plots any fit and error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    wvlngth : Integer
        specific wavelength to plot kinetics for.
    plotFit : Boolean, optional
        Whether to plot a fitted line, if available. The default is True.
    norm : Boolean, optional
        Whether to normalize the kinetic line to 1, at some time. The default is False.
    normT : Integer, optional
        The time in ps to normalize a kinetic trace to. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.
    """
    label= str(wvlngth)
    
    gcf, ax = plt.subplots()
    cmap = get_cmap(len(TAdata)+1)
    
    for i,data in enumerate(TAdata):
        Intensity = data.Kinetic[label]
        Time = data.Time
        
        if (normT > Time[-1]) or (normT<Time[0]):
            print('Time to normalize to is outside of range')
            norm=False
        
        if norm:
            norm_i = [i for i,x in enumerate(Time) if math.isclose(x, normT,rel_tol=(0.2))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
        
        if label in data.FitResults and plotFit:
            fit_wvl = data.KineticFit[label]
        
        if data.Ave and plotError:
            error = data.KineticStd[label]
        
        if data.Ave and plotError:
            ax.errorbar(Time,Intensity/normFact,yerr=error/normFact,fmt='.',capsize=1,label = data.name,color = cmap(i))
        else:
            ax.plot(Time,Intensity/normFact,'.',label = data.name,color = cmap(i))
            
        if label in data.FitResults and plotFit:
            ax.plot(Time,fit_wvl/normFact,label = data.name+"fit",c=cmap(i))
        
        plt.legend()
        
    return gcf

def PlotMS(TAdata,time_slice,norm = False,normW = 500,plotError = False):
    """
    Takes a list of TAdata, and plots a spectral trace. Plots error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    time_slice : float
        time to plot spectral traces at.
    norm : Boolean, optional
        Whether to normalize the spectral trace to 1, at some wavelength. The default is False.
    normW : integer, optional
        The wavelength to normalize a spectral trace to 1 at. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.
    """
    
    label= str(time_slice)
    
    gcf, ax = plt.subplots()
    cmap = get_cmap(len(TAdata)+1)
    
    for i,data in enumerate(TAdata):
        Intensity = data.T_slice[label]
        Wavelengths = data.Wavelength
        
        if norm:
            norm_i = [i for i,x in enumerate(Wavelengths) if math.isclose(x, normW,rel_tol=(0.01))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
            
        if data.Ave and plotError:
            error = data.T_sliceStd[label]
            
        if data.Ave and plotError:
            ax.errorbar(Wavelengths,Intensity/normFact,yerr=error/normFact,capsize=0.5,label = str(i))
        else:
            ax.plot(Wavelengths,Intensity/normFact,'r.',markevery =None,label = data.name,c=cmap(i))
            
        plt.legend()
            
    return gcf

def PlotMSS(TAdata,time_points,norm = False,normW = 500,plotError = False):
    """
    Takes a single TAdata and plots a range of spectral traces. Can plot error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    time_points : List
        Plots multiple spectral traces at these times.
    norm : Boolean, optional
        Whether to normalize the spectral trace to 1, at some wavelength. The default is False.
    normW : integer, optional
        The wavelength to normalize a spectral trace to 1 at. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.
    """
    gcf, ax = plt.subplots()
    cmap = get_cmap(len(time_points)+1)
    
    for i,t_pnts in enumerate(time_points):
        label=str(t_pnts)
        
        if label in TAdata.T_slice:
            data = TAdata.T_slice[label][:]
        else:
            TAdata.SpectralTrace(t_pnts,t_pnts/10)
            data = TAdata.T_slice[label][:]
        
        Intensity = TAdata.T_slice[label]
        Wavelengths = TAdata.Wavelength
        
        if norm:
            norm_i = [i for i,x in enumerate(Wavelengths) if math.isclose(x, normW,rel_tol=(0.01))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
            
        if TAdata.Ave and plotError:
            error = TAdata.T_sliceStd[label]
            
        if TAdata.Ave and plotError:
            ax.errorbar(Wavelengths,Intensity/normFact,yerr=error/normFact,capsize=0.5,label = label)
        else:
            ax.plot(Wavelengths,Intensity/normFact,'-',markevery =None, label = label,c=cmap(i))
            
        #future todo when a robust spectral fitting is implemented    
        # if plotFit: #label in TAdata.SpectralFit and
        #     fit_data = TAdata.SpectralFit[label]
        #     ax.plot(Wavelengths,fit_data/normFact,label = "fit"+str(i),c=cmap(i))
            
        plt.legend()
            
    return gcf

def get_cmap(i,name = 'hsv'):
    return plt.cm.get_cmap(name,i)
