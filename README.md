# pCO2Residual_Testbed

!Code and documentation in progress!

Code to implement Bennington et al. 2022 JAMES using the CMIP6 testbed. This code is set up to be run on the LEAP Pangeo computing platform (see documentation here: https://leap-stc.github.io/intro.html)

Brief summary on each notebook:
- residual_utils: Files with supporting functions
- 00_regridding_members : CMIP6 member selection and re-gridding 
- 01_create_ML_inputs : Creates a dataframe with all variables needed to run notebook 02
- 02_xgboost : Reconstructs pCO2-Residual using XGBoost
- 03_add_back_pco2-T : Adds pCO2-T to pCO2-Residual to get pCO2

If you want to follow our process from beginning to end, start with notebook 00 and work through notebook 03. Otherwise, if you want to use our already-made testbed, work through notebook 02 and 03. 

Processed testbed files (produced after running notebook 00 and 01) are available in the LEAP Pangeo cloud storage. The testbed files are pickle files containing the target and all driver variables; there is one separate pickle file for each of the 45 members of the testbed. 

If you want to use the processed testbed files, but you do not wish to use LEAP Pangeo, the testbed is publicly accessible on the Open Storage Network (OSN) pod. In order to access the testbed on the OSN pod, switch the "MLinputs_path" (in notebook 02) to the following path: https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs. 

The testbed is structured as follows: within the "post01_xgb_inputs" folder, there are 9 sub-folders, one for each of the Earth System Model (ESM) in the testbed. Within each of the 9 folders, there are sub-folders representing members and the number of members vary between each ESM. Within each member folder, there is one pickle file including the target and all driver variables.

https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/ACCESS-ESM1-5
https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/CESM2
https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/CESM2-WACCM
https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/CMCC-ESM2
https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/CanESM5
https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/CanESM5-CanOE
https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/GFDL-ESM4
https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/MPI-ESM1-2-LR
https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/UKESM1-0-LL

The folder named "Figures_Heimdal_et_al_2025" includes notebooks for figures and calculations presented in Heimdal et al. (2025): https://eartharxiv.org/repository/view/8958/
