# -*- coding: utf-8 -*-

import os, shutil, re
import numpy as np
import math , pickle
import sys
import TAclass #TAdata,AveTAdata

def load_files(directory, reimport):
    """
    General utility for taking a directory and loading all TA files. Files must include
    _TA_ in the filename.    

    Parameters
    ----------
    directory : str
        Directory that holds all ta data.
    reimport : Boolean
        Whether to reload all raw data, or pull data from pickled data.

    Returns
    -------
    List with all TAdata.

    """
    if (directory == ""):
        return None

    np.set_printoptions(threshold=sys.maxsize,linewidth=15000)

    TAfilename=[];
    WLfilename=[];
    WLufilename=[];
    TAlist=[]
    pickled = 0;
    
    for entry in os.scandir(directory):
        if(entry.is_file()==True and bool(re.search('[^var]_TA_\d*.txt',entry.name))):
        # filename.append(entry.name)
            TAfilename.append(entry.path)
        elif (bool(re.search('[^var]_WLPump_\d*.txt',entry.name))):
            WLfilename.append(entry.path)
        elif (bool(re.search('[^var]_WLUnPump_\d*.txt',entry.name))):
            WLufilename.append(entry.path)  
        elif(bool(re.search('Pickle',entry.name)) and not reimport):
            with open(entry.path,'r+b') as pickleFile:
                TAlist=pickle.load(pickleFile)
                print("Found pickled object, loading...")
                pickled = 1
                pickleFile.close()
    
    if pickled:
        return TAlist
    else:
        print("No pickled object, loading " + str(len(TAfilename)))
        if (len(TAfilename) == len (WLfilename) and len(TAfilename)==len(WLufilename)):      
            for i in range(len(TAfilename)):
                TAdata_i = TAclass.TAdata(TAfilename[i], WLfilename[i], WLufilename[i]) 
                TAlist.append(TAdata_i)
        else:
            for i in range(len(TAfilename)):
                TAdata_i = TAclass.TAdata(TAfilename[i],"","") 
                TAlist.append(TAdata_i)
        
        with open(directory + "/PythonPickle",'w+b') as TAfile:
            print("Pickling...")
            pickle.dump(TAlist,TAfile)
            TAfile.close()
        
        return TAlist
    
def saveKin(TAdata,directory,basename,wvlngth):
    """
    Utility to save all kinetic traces, either for a list of data or a single dataset.
    Will also save fits to the kinetic data, if available.

    Parameters
    ----------
    TAdata : Single TAdata or list of TAdata
        The data to save kinetic traces from.
    directory : str
        Where to save all the kinetic traces.
    basename : str
        Name to use for header and filename.
    wvlngth : int
        The wavelength to save kinetic traces from.

    Returns
    -------
    None.

    """
    label = str(wvlngth)
    if re.search('/\Z', directory)==None:
        directory +='/'
        
    if isinstance(TAdata,list):
        #TAdata is not an individual object, but a list of objects
        for i,data in enumerate(TAdata):
            hdr = "Time" +str(i)+", kin" + basename+str(i) + ", Std" + basename +str(i) +", fit" \
                + basename+str(i)
            np.savetxt(directory+basename+str(i)+".txt",
            np.stack([data.Time,data.Kinetic[label],data.KineticStd[label],
                      data.KineticFit[label]],axis=1),header = hdr)
    else:
        hdr = "Time" +", kin" + basename + ", Std" + basename 
        np.savetxt(directory+basename+".txt",
        np.stack([TAdata.Time,TAdata.Kinetic[label],
                  TAdata.KineticStd[label]],axis=1),header = hdr)
        
def saveSpectral(TAdata,directory,basename,time):
    """
    Utility to save all spectral traces, either for a list of data or a single dataset.

    Parameters
    ----------
    TAdata : Single TAdata or list of TAdata
        The data to save kinetic traces from.
    directory : str
        Where to save all the kinetic traces.
    basename : str
        Name to use for header and filename.
    time : int
        The time to save spectral traces from.

    Returns
    -------
    None.

    """
    
    label = str(time)
    
    if re.search('/\Z', directory)==None:
        directory +='/'
    
    if isinstance(TAdata,list):
        #TAdata is not an individual object, but a list of objects
        for i,data in enumerate(TAdata):
            hdr = "Wavelength" +str(i)+", T_slice" + basename+str(i) + ", Std" + basename +str(i)
            np.savetxt(directory+basename+str(i)+".txt",
            np.stack([data.Wavelength,data.T_slice[label],
                      data.T_sliceStd[label]],axis=1),header = hdr)
    else:
        hdr = "Wavelength" +", T_slice" + basename + ", Std" + basename 
        np.savetxt(directory+basename+".txt",
        np.stack([TAdata.Wavelength,TAdata.T_slice[label],
                  TAdata.T_sliceStd[label]],axis=1),header = hdr)
    
def saveFits(FitList,directory,basename, components):
    """
    Utility to save parameters from fits from multiple fits, such as AIC and time constants/amplitudes.
    Amplitudes will be saved as normalized amplitudes, may not be as meaningful when the amplitudes have different signs.

    Parameters
    ----------
    FitList : List
        This should be a list of fit results, as outputted by manyFitKinetic, for example.
    directory : str
        Where to save the fits.
    basename : str
        Name to use for file name and headers.
    components : int
        How many components are in the fitted results.

    Returns
    -------
    None.

    """
    
    label = str(components)
    AIC = []
    BIC = []
    outArr1 =np.zeros((4*components))
    hdr = "AIC" +basename+", BIC" + basename
    
    if re.search('/\Z', directory)==None:
        directory +='/'
        
    for n in range(1,components+1):
        hdr+= ", C"+str(n) + basename + ", C" +str(n)+"Std_" + basename
        hdr+= ", A"+str(n) + basename + ", A" +str(n)+"Std_" + basename
        
    for i,data in enumerate(FitList):
        params = data.params.valuesdict()
        AIC.append(data.aic)
        BIC.append(data.bic)
        
        normA=1
        # for n in range(1,components+1): 
        #     label= "A"+str(n)
        #     normA += params[label]
            
        tmparr = []
        for n in range(1,components+1):                        
            label = "C"+str(n)
            tmparr.append(params[label])
            if data.params[label].stderr is None:
                tmparr.append(0)
            else:
                tmparr.append(data.params[label].stderr)
            label= "A"+str(n)
            tmparr.append(params[label])
            if data.params[label].stderr is None:
                tmparr.append(0)
            else:
                tmparr.append(data.params[label].stderr/abs(tmparr[-1]))
            tmparr[-2] /= normA
            
        outArr1 = np.vstack((outArr1,tmparr))
    
    outArr2 = np.stack([AIC,BIC])
    outArr = np.delete(outArr1,0,axis=0)
    outArr = np.concatenate([outArr2.transpose(),outArr],axis=1)
    np.savetxt(directory+basename+".txt",outArr,header = hdr)
    
def exportAllTA(Directory,TAlist,FileNameList):
    """
    Utility to save everything for a list of TAdata sets, including spectral traces,
    kinetic traces, time, wavelengths, 2D matrix, and standard deviation matrix.

    Parameters
    ----------
    Directory : str
        Where to save data.
    TAlist : list
        List of TAdata to export.
    FileNameList : List
        List of names as a string to save each dataset to.

    Returns
    -------
    None.

    """
    if re.search('/\Z', Directory)==None:
        Directory +='/'
       
    if isinstance(TAlist,list):     
        for data,file in zip(TAlist,FileNameList):
            for kin in data.Kinetic:
                saveKin(data,Directory,file+kin,kin)
                
            for spec in data.T_slice:
                saveSpectral(data,Directory,file+spec+"ps",spec)
                
            np.savetxt(Directory+file+".txt",data.Intensity,header = file)
            np.savetxt(Directory+file+"Std.txt",data.Std,header = file)
            np.savetxt(Directory+file+"Time.txt",data.Time,header = file)
            np.savetxt(Directory+file+"Wavelength.txt",data.Wavelength,header = file)
    else:
        if isinstance(FileNameList,list):
            file = FileNameList[0]
        else:
            file = FileNameList
            
        for kin in TAlist.Kinetic:
            saveKin(TAlist,Directory,file+kin,kin)
            
        for spec in TAlist.T_slice:
            saveSpectral(TAlist,Directory,file+spec+"ps",spec)
            
        np.savetxt(Directory+file+".txt",TAlist.Intensity,header = file)
        np.savetxt(Directory+file+"Std.txt",TAlist.Std,header = file)
        np.savetxt(Directory+file+"Time.txt",TAlist.Time,header = file)
        np.savetxt(Directory+file+"Wavelength.txt",TAlist.Wavelength,header = file)

 # fancy way to activate the main() function
# if __name__ == '__main__':
#     load_files('C:/Users/Bozark/Documents/Samples/OTA/April 2021 ZnOCdSe/CdSe_84B_5uW_visProbe')