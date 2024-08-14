# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:50:53 2021

@author: Bozark
"""

from TAclass import TAdata,AveTAdata,GVDdata
import TAplotter as TAplt
import Import as imp
import matplotlib.pyplot as plt
import KineticFitFunctions as kff
import SpectralFitFunctions as sff
import os,time
import numpy as np
from scipy.interpolate import interp1d
from lmfit import Parameters
from mpl_point_clicker import clicker 

REIMPORT = True
DIRECTORY = r'C:\Users\Samth\Box\VWGroupShared\Data\MRL_108_20240515_OTA_AQDS\20240514_AQDS\water_pH5_27AQDS_magic_angle\Test'
EXP_DIR = "/105E_UV/105E_19p25x9p25y_30uJ"
FluenceDir = DIRECTORY + "/105E_Vis/FluenceDep"
all_subdirs=True
PROBE = [320,700]

def main(directory, all_subdirs,reimport):
    """"This is the main function for importing TA data. It will automatically find
    files with _TA, _WL, and _WLu based on the naming convention of the MRL laser.
    It requires the directory, whether you want to reimport the data as True or False.
    The output is a list, with the data loaded as a TAdata structure.
    
    If you set all_subdirs to True, then the function will loop through all subfolders,
    and return a list of lists with all the data in each subfolder.
    """
    plt.close('all')

    if all_subdirs:
        TAdata=[]
        for entry in os.scandir(directory):
            if(entry.is_dir()==True):
                print("importing: " + entry.path)
                TAdata.append(imp.load_files(entry.path, reimport))

    else:
        TAdata = imp.load_files(directory, reimport)
    
        for data in TAdata:
            data.BckgSub()
            TAplt.plotTAdata(data)
            
    return TAdata

def AveTA(TAdata, probe, t0_meth = 'auto', kineticWVL="",**kwargs):
    """Average a set of TAdata. Requires a list of TAdata, a range of wavelengths to average over,
    and a wavelength to align time 0."""
    
    if kineticWVL != "":
        if t0_meth == 'auto':
            for data in TAdata:
                data.KineticTrace(kineticWVL, 1)
                # CleanKin(data,kineticWVL,factor=3)
                data.Time -= kff.t_zero(data,kineticWVL,**kwargs)
        else:
            for data in TAdata:
                data.KineticTrace(kineticWVL,1)
                # CleanKin(data,kineticWVL,factor=3)
                data.Time -= getT0pos(data, kineticWVL)
            
    TAave = AveTAdata(TAdata,probe)
    TAplt.plotTAdata(TAave)
    
    return TAave

def manyKinetic(TAdata,wvlngth,binw, norm = True,normT = 500, clean = True,cleanFactor=3):
    """Take a list of TAdata and extract kinetic traces for each, then plot together. Can bin over a wavelength range, along with
    options to normalize the plotting. Clean is an option to remove outliers."""
    
    for data in TAdata:
        data.KineticTrace(wvlngth,binw)
        if clean and data.Ave:
            CleanKin(data,wvlngth,factor=cleanFactor)
            
    TAplt.PlotMK(TAdata, wvlngth, norm = norm,normT = normT)
    
def manySpectral(TAdata,Time,binw,norm = True,normW = 500):
    """Take a list of TAdata and extract spectral traces for each, then plot together. Can bin over a wavelength range, along with
    options to normalize the plotting."""
  
    for data in TAdata:
        data.SpectralTrace(Time,binw)
            
    TAplt.PlotMS(TAdata,Time,norm = norm,normW = normW )

def manyFitKinetic(TAdata,wvlngth,comp,**kwargs):
    """
    Function to take a list of datasets, and fit a kinetic trace to the same exp model.

    Parameters
    ----------
    TAdata : list
        List of TAdata.
    wvlngth : int
        Wavelength to fit for each dataset.
    comp : int between 1 and 5
        Numbers 1 through 4 are # of exponential components to fit. 
        5 components is actually a stretched exponential fit.
    **kwargs : dictionary
         Dictionary of parameters to pass to exponential fitting. Can include amplitude (A1...) or time (C1...).
    Note: time_mask is a list of lists. Ex: [[0,10]] would exclude all time points between
        0 and 10 ps. [[0,10],[1000,7000]] would exclude all time points between 0 and 10 ps,
        and 1 and 7 ns. Passing [0,10] will not work correctly.
    Returns
    -------
    results : list
        List of lmfit minimizer results. Can be passed to saveFits function to export results.

    """
    
    results = []
    for data in TAdata:
        if comp ==1:
            results.append(kff.oneExpFit(data, wvlngth,quiet = True,**kwargs))
        elif comp ==2:
            results.append(kff.twoExpFit(data, wvlngth,quiet = True,**kwargs))
        elif comp ==3:
            results.append(kff.threeExpFit(data, wvlngth,quiet = True,**kwargs))
        elif comp ==4:
            results.append(kff.fourExpFit(data, wvlngth,quiet = True,**kwargs))
        elif comp == 5:
            results.append(kff.stretchExpFit(data, wvlngth, quiet=True, **kwargs))    
        else:
            results.append(kff.lognormFit(data, wvlngth, quiet=True, **kwargs))    
            
    for i in results:
        params = i.params.valuesdict()
        print("aic: " + str(i.aic))
        print("bic: " + str(i.bic))
        print("chi2: " + str(i.chisqr))
        print("C1: " + str(params['C1']))
            
    return results

def mainFluence(directory, probe, kineticWVL, reimport, plotAll=False):
    """
    Main function to import fluence dependent data all at once. Give it a directory with a subdirectory for each fluence.

    Parameters
    ----------
    directory : str
        Directory, where each fleunce is in a subdirectory.
    probe : list
        range of wavelengths to use for averaging.
    kineticWVL : float
        Wavelength to use for aligning t0.
    reimport : Boolean
        Whether to look for pickled data or not.
    plotAll : Boolean, optional
        Whether to plot the individual TAdatasets. Can make a lot of figures. The default is False.

    Returns
    -------
    TAavelist : list of averaged TAdatasets.
    """
    
    plt.close('all')
    TAlist = []
    TAavelist = []
    
    for entry in os.scandir(directory):
        if(entry.is_dir()==True):
            print("importing: " + entry.path)
            TAlist.append(imp.load_files(entry.path, reimport))
        
    for data in TAlist:
        for measure in data:
            print("Background subtraction and plotting for:\n")
            print(measure.name)
            measure.BckgSub()
            if plotAll==True:
                TAplt.plotTAdata(measure)
                
        TAavelist.append(AveTA(data, probe, kineticWVL))
        
    return TAavelist

def manyFitSpectral(TAdata,t_slice,params,wvl_mask=[],**kwargs):
    """
    Takes a list of TAdata and extracts and plots a spectral slice. Unfinished functioin

    Parameters
    ----------
    TAdata : List
        List of TAdata.
    t_slice : time
        DESCRIPTION.
    params : Fitting parameters
        DESCRIPTION.
    wvl_mask : list of wavelengths to exclude, optional
        DESCRIPTION. The default is [].
    **kwargs : dictionary 
        Used to pass arguments to the spectral fitting function..

    Returns
    -------
    results : list of lmfit results.

    """
    results=[]
    for i in TAdata:
        results.append(sff.FitSingleSpectral(i, t_slice, params,wvl_mask=wvl_mask,**kwargs))
        
    return results

def undoBckSub(TAdata,indexrange):
    """"Use a custom index range to fix background correction. Takes and returns a single TAdata.
    Indexrange is the custom background correction, default background uses the first 10 points."""
    
    TmpArr = TAdata.Intensity[:,indexrange]
    TmpMean = np.mean(TmpArr,axis=1)
    TAdata.Intensity -= TmpMean[:,None]

def GVDcorr(TAdata):
    """
    This is a function to take either a single average dataset, or a list of 
    datasets to apply the same GVD correction to.

    Parameters
    ----------
    TAdata : single average TAdata, or a list of TAdata

    Returns
    -------
    Corrected : GVD corrected TAdata either single or as a list.

    """
    if type(TAdata)!=list:
        GVDenergy,GVDtime = getGVDpos(TAdata)
    else:
        GVDenergy,GVDtime = getGVDpos(TAdata[0])
        
    GVD_fit = np.polyfit(GVDenergy,GVDtime,3)
    GVD_correct = []
    for e in GVDenergy:
        GVD_correct.append(np.polyval(GVD_fit, e))
        
    fig,ax = plt.subplots()
    ax.plot(GVDenergy,GVDtime,'ro')
    ax.plot(GVDenergy,GVD_correct)
    
    if type(TAdata)!=list:
        GVD_data = np.zeros(np.shape(TAdata.Intensity))
        GVD_std = np.zeros(np.shape(TAdata.Intensity))
        for idx,wvl in enumerate(TAdata.Wavelength):
            wvl_interp = interp1d(TAdata.Time,TAdata.Intensity[idx],
                                  kind='linear',fill_value='extrapolate')
            std_interp = interp1d(TAdata.Time,TAdata.Std[idx],
                                  kind='linear',fill_value='extrapolate')
            
            GVD_data[idx][:] = wvl_interp(TAdata.Time[:]+np.polyval(GVD_fit,1239/wvl))
            # GVD_data[idx][:] = wvl_interp(TAdata.Time[:]+np.polyval(GVD_fit,wvl))
            GVD_std[idx][:] = std_interp(TAdata.Time[:]+np.polyval(GVD_fit,1239/wvl))
            
        Corrected = GVDdata(GVD_data,GVD_std,TAdata)
        TAplt.plotTAdata(Corrected)  
    else:
        Corrected = []
        for data in TAdata:
            GVD_data = np.zeros(np.shape(data.Intensity))
            GVD_std = np.zeros(np.shape(data.Intensity))
            for idx,wvl in enumerate(data.Wavelength):
                wvl_interp = interp1d(data.Time,data.Intensity[idx],
                                      kind='linear',fill_value='extrapolate')
                std_interp = interp1d(data.Time,data.Std[idx],
                                      kind='linear',fill_value='extrapolate')
                
                GVD_data[idx][:] = wvl_interp(data.Time[:]+np.polyval(GVD_fit,1239/wvl))
                # GVD_data[idx][:] = wvl_interp(TAdata.Time[:]+np.polyval(GVD_fit,wvl))
                GVD_std[idx][:] = std_interp(data.Time[:]+np.polyval(GVD_fit,1239/wvl))
                
            Corrected.append(GVDdata(GVD_data,GVD_std,data))  
        manySpectral(Corrected, 0.1, 0.05)
        TAplt.plotTAdata(Corrected[0]) 
    
    return Corrected

def LinearPCorr(TAlist,factor=0.1):
    """
    Apply a linear rescaling to TAdata to correct for a power drift.

    Parameters
    ----------
    TAlist : list
        TAdata to adjust for a drift in power.
    factor : float, optional
        Percent correction to apply for TAdata. If power drifted by -10% between the first and last scan, use a factor of 0.1. 
        The default is 0.1, or 10%.

    Returns
    -------
    None.

    """
    for i,data in enumerate(TAlist):
        data.Intensity = data.Intensity *(1+(i/len(TAlist))*factor)

def CleanKin(TAdata,wvlngth,factor = 3):
    """
    Remove outliers from kinetic trace. Takes a single TAdata and wavelength, along with a factor for the number of standard deviations
    to consider an outlier.

    Parameters
    ----------
    TAdata : TAdata
        A single dataset to correct a kinetic trace for.
    wvlngth : Int
        The wavelength at which to remove outliers from.
    factor : integer, optional
        The number of standard deviations away from the mean to consider a point as an outlier. The default is 3.

    Returns
    -------
    None.

    """
    label = str(wvlngth)
    
    stdMean = np.median(TAdata.KineticStd[label])
    if np.mean(TAdata.KineticStd[label])<stdMean:
        stdMean = np.mean(TAdata.KineticStd[label])

    for i,pnt in enumerate(TAdata.Kinetic[label]):
        if i ==0:
            mean = 0
        elif i > (len(TAdata.Kinetic[label])-3):
            mean = sum(TAdata.Kinetic[label][i-4:i-1])/4
        else:
            mean = (TAdata.Kinetic[label][i-1]+
                    TAdata.Kinetic[label][i+1]+
                    TAdata.Kinetic[label][i-2]+
                    TAdata.Kinetic[label][i+2])/4
        if abs(pnt) > (abs(mean)+stdMean*factor):
            TAdata.Kinetic[label][i] = mean
            TAdata.KineticStd[label][i] = stdMean

def onclick(event):
    global offset
    offset = 1
    return 

def getGVDpos(TAdata):
    """"Function to use to extract points for GVDcorrection. Only call on it's own if you want the points used for GVDcorrection.
    Returns a tuple of GVD points in energy and time"""
    GVDimg = TAplt.plotTAdata(TAdata,
                              color_min=np.min(TAdata.Intensity),
                              color_max=np.max(TAdata.Intensity))
    GVDaxs = GVDimg.axes
    klicker = clicker(GVDaxs[0],['GVDpos'],markers=["x"],linestyle="-",colors=["red"])
    plt.show()
        
    cid = GVDimg.canvas.mpl_connect('close_event', onclick)
    while not 'offset' in globals():
        plt.pause(2)
    
    if 'offset' in globals():
        global offset
        del offset

    GVDimg.canvas.mpl_disconnect(cid)
    
    GVDpos = klicker.get_positions()['GVDpos']
    GVDidx = np.array(np.round(GVDpos[:,1],0),dtype='int')
    GVDenergy = 1239/TAdata.Wavelength[GVDidx]
    GVDidx = np.array(np.round(GVDpos[:,0],0),dtype='int')
    GVDtime = TAdata.Time[GVDidx]
    
    return GVDenergy,GVDtime
    
def getT0pos(TAdata,kinData):
    """"Function to use to extract points for GVDcorrection. Only call on it's own if you want the points used for GVDcorrection.
    Returns a tuple of GVD points in energy and time"""
    T0img = TAplt.PlotKinetic(TAdata,kinData)
    plt.title('Select a single time point to shift t0 to')
    T0axs = T0img.axes
    klicker = clicker(T0axs[0],['T0pos'],markers=["x"],linestyle="-",colors=["red"])
    plt.show()
        
    cid = T0img .canvas.mpl_connect('close_event', onclick)
    while not 'offset' in globals():
        plt.pause(2)
    
    if 'offset' in globals():
        global offset
        del offset

    T0img.canvas.mpl_disconnect(cid)
    
    T0pos = klicker.get_positions()['T0pos']
    T0time = np.array(T0pos[:,0])
    
    return T0time[0]

def autoFitGVD(TAdata,fit_region,method):
    """Currently broken function but is the barebones meothod
    for gving it a whole TA dataset and having the GVD corrected automatically"""
    threshold = 0.0005
    Zero_list = []
    Zero_list_arg = []
    GVD_energy=[]
    
    if(len(fit_region)>0):
        GVD_wvl = np.arange(fit_region[0],fit_region[1],2)
    else:
        GVD_wvl = np.arange(TAdata.Wavelength[0],TAdata.Wavelength[-1],2)
    
    for wvl in GVD_wvl:
        #probably need to include some CPM-shaped-fit, a basic t0 fit doesn't give 
        #good, consistent results where there is little signal.
        idx = next(i for i,_ in enumerate(TAdata.Wavelength) if np.isclose(_,wvl,0.005))
        
        if method ==1:
            tmpVal = np.argmax(np.abs(TAdata.Intensity[idx][:])>threshold)
            while tmpVal<len(TAdata.Wavelength):
                if (np.abs(TAdata.Intensity[idx][tmpVal])>threshold) and \
                    (np.abs(TAdata.Intensity[idx][tmpVal])/TAdata.Std[idx][tmpVal]>2):
                    # (np.abs(TAdata.Intensity[idx][tmpVal+1])>threshold) and
                    #     (np.abs(TAdata.Intensity[idx][tmpVal-1])>threshold) and
                    Zero_list_arg.append(tmpVal)
                    # GVD_energy.append(1239/wvl)
                    GVD_energy.append(wvl)
                    break
                tmpVal +=1
        else:
            wvl_interp = interp1d(TAdata.Time,TAdata.Intensity[idx],
                                  kind='cubic')
            tmpVal = np.argmax(np.abs(wvl_interp(TAdata.Time[:]))>threshold)
            while tmpVal<len(TAdata.Wavelength):
                if (np.abs(TAdata.Intensity[idx][tmpVal-1])>threshold) and \
                    (np.abs(TAdata.Intensity[idx][tmpVal+1])>threshold) and\
                        (np.abs(TAdata.Intensity[idx][tmpVal])>threshold) and\
                            (np.abs(TAdata.Intensity[idx][tmpVal])/TAdata.Std[idx][tmpVal]>2):
                    Zero_list_arg.append(tmpVal)
            # GVD_energy.append(1239/wvl)
                    GVD_energy.append(wvl)
                    break
                tmpVal+=1

    return GVD_energy

# if __name__ == '__main__':
#     TAdata = main()