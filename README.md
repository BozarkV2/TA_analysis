# TA_analysis
A collection of classes, methods, and scripts for working with OTA data. Specifically data produced at UIUC, but expanding to include more generic collection schemes.

You can install the dependencies of this code base by downloading the requirements file and running the following:
pip install -r requirements.txt

Designed to work with multiple sets of optical transient absorption files and quickly process them, including:
- importing multiple datasets from a single directory (main) or from multiple (main, or mainFluence)
- averaging those datasets (AveTA)
- taking kinetic traces (manyKinetic) or spectral traces (manySpectral) of multiple datasets
- fitting multiple kinetic traces to a kinetic model (manyFitKinetic) Note: support for global fitting of many datasets is in progress
- plotting 2D data (TAplt.plotTAdata)
- plotting many kinetic traces (TAplt.PlotKinetic)
- plotting many spectral traces (TAplt.PlotSpectral)
- Saving the results of many kinetic traces (imp.SaveKinetic) or spectral traces (imp.SaveSpectral)
- Saving the results of many kinetic fits (imp.SaveFits)

