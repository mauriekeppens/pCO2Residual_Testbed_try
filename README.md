# pCO2Residual_Testbed
Code to implement Bennington et al. 2022 JAMES using the CMIP6 testbed.
Code and documentation in progress!

Brief summary on each notebook:
- residual_utils: Files with supporting functions
- 00_regridding_members : CMIP6 member selection and re-gridding 
- 01_create_ML_inputs : Creates a dataframe with all variables needed to run notebook 02
- 02_xgboost : Reconstructs pCO2-Residual using XGBoost
- 03_add_back_pco2-T : Adds pCO2-T to pCO2-Residual to get pCO2

If you want to follow our process from beginning to end (including making a regridded testbed using the same CMIP6 ESMs as we did, and running it through XGBoost), start with notebook 00 and work through notebook 03. Otherwise, if you want to use our already-made testbed, download it from Zenodo (LINK HERE), and work through notebook 02 and 03. You will need to change the paths at the top of the notebook.

If you want only to use our already-made testbed for your own ML algorithm, **all** you need to do is download the testbed from Zenodo. The process for making it is in notebook 00. 

Folder "Figures_Heimdal_et_al_2025" includes notebooks for figures and calculations presented in Heimdal et al. (2025): https://eartharxiv.org/repository/view/8958/
