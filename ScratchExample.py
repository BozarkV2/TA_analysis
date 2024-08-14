# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:16:44 2024

@author: Bozark
"""

import main as mn
import TAplotter as TAplt
import Import as imp

DIRECTORY = r'C:\MRL_108\water_pH5_27AQDS_magic_angle'
all_subdir=False
KineticWvl = 500 #nm
binw=1 #nm
outDir = r'C:\MRL_108_20240515_OTA_AQDS\water_pH5_27AQDS_magic_angle\Out'
PROBE = [350,700]

AQDOTA = mn.main(DIRECTORY,all_subdir,reimport = True)
Ave_AQD = mn.AveTA(AQDOTA,PROBE,t0_meth=Auto,kineticWVL=KineticWvl)
GVD_AQD = mn.GVDcorr(Ave_AQD)

GVD_AQD.KineticTrace(KineticWvl, binw)
mn.manyKinetic(AQDOTA,KineticWvl,binw)

TAplt.plotTAdata(GVD_AQD,color_max=0.005,color_min=-0.005)

num_comp=1
results = mn.manyFitKinetic(AQDOTA, KineticWvl, num_comp)
imp.saveFits(results, outDir, 'AQDFit500nm', num_comp)
imp.saveKin(AQDOTA, outDir, 'AQD500nm', KineticWvl)
