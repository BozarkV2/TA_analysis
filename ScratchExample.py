# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:16:44 2024

@author: Samth
"""

import main as mn
import TAplotter as TAplt
import Import as imp

DIRECTORY = r'C:\Users\Samth\Box\VWGroupShared\Data\MRL_108_20240515_OTA_AQDS\20240514_AQDS\water_pH5_27AQDS_magic_angle\Test'
outDir = r'C:\Users\Samth\Box\VWGroupShared\Data\MRL_108_20240515_OTA_AQDS\20240514_AQDS\water_pH5_27AQDS_magic_angle\Test'
PROBE = [350,700]

AQDOTA = mn.main(DIRECTORY,False,reimport = True)
Ave_AQD = mn.AveTA(AQDOTA,PROBE)
GVD_AQD = mn.GVDcorr(Ave_AQD)

GVD_AQD.KineticTrace(500, 1)
mn.manyKinetic(AQDOTA,500,1)

TAplt.plotTAdata(GVD_AQD,color_max=0.005,color_min=-0.005)

results = mn.manyFitKinetic(AQDOTA, 500, 1)
imp.saveFits(results, outDir, 'AQDFit500nm', 1)
imp.saveKin(AQDOTA, outDir, 'AQD500nm', 500)