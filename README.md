# pCO2Residual_Testbed
Code to implement Bennington et al. 2022 JAMES using CMIP6 testbed.

Brief summary on each notebook:
- 00_regridding_members :
- 01_create_ML_inputs :
- 02_xgboost :
- 03_add_back_pco2-T :
- 04_calculate_flux :

If you want to follow our process from beginning to end (including making a regridded testbed using the same CMIP6 ESMs as we did, and running it through XGBoost), start with notebook 00 and work through notebook 03. Otherwise, if you want to use our already-made testbed, download it from Zenodo (LINK HERE), and start from notebook 01 and work through notebook 03. You will need to change the paths at the top of the notebook.

If you want only to use our already-made testbed for your own ML algorithm, **all** you need to do is download the testbed from Zenodo. The process for making it is in notebook 00. 
