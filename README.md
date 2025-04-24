# pCO2Residual_Testbed
Code to implement Bennington et al. 2022 JAMES using the CMIP6 testbed.

Code and documentation in progress!

Brief summary on each notebook:
- residual_utils: Files with supporting functions
- 00_regridding_members : CMIP6 member selection and re-gridding 
- 01_create_ML_inputs : Creates a dataframe with all variables needed to run notebook 02
- 02_xgboost : Reconstructs pCO2-Residual using XGBoost
- 03_add_back_pco2-T : Adds pCO2-T to pCO2-Residual to get pCO2

If you want to follow our process from beginning to end, start with notebook 00 and work through notebook 03. Otherwise, if you want to use our already-made testbed, work through notebook 02 and 03. If you are using the LEAP Pangeo computing platform, you can use the provided path in notebook 02. If not using LEAP Pangeo, processed testbed files (produced after running notebook 00 and 01), are publicly accessible on the LEAP OSN pod; switch to the following path: https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs.   

The folder named "Figures_Heimdal_et_al_2025" includes notebooks for figures and calculations presented in Heimdal et al. (2025): https://eartharxiv.org/repository/view/8958/
