# The CMIP6 testbed using the pCO2Residual method

!! Code and documentation in progress !!

Code to implement the pCO2-Residual method (Bennington et al. 2022; https://doi.org/10.1029/2021ms002960) using the CMIP6 testbed (Heimdal et al., 2025; https://eartharxiv.org/repository/view/8958/).  

This code is set up to be run on the LEAP Pangeo computing platform (see documentation here: https://leap-stc.github.io/intro.html)

Brief summary on each notebook:
- residual_utils: Files with supporting functions
- 00_regridding_members : CMIP6 member selection and re-gridding 
- 01_create_ML_inputs : Creates a dataframe with all variables needed to run notebook 02
- 02_xgboost : Reconstructs pCO2-Residual using XGBoost
- 03_add_back_pco2-T : Adds pCO2-T to pCO2-Residual to get pCO2

If you want to follow our process from beginning to end, start with notebook 00 and work through notebook 03. Otherwise, if you want to use our 'already-made' testbed, work through notebook 02 and 03. Processed testbed files (produced after running notebook 00 and 01) are available in the LEAP Pangeo cloud storage. The testbed files are pickle files containing the target and all driver variables; there is one individual pickle file for each of the 45 members of the testbed. 

If you want to use the processed testbed pickle files, but you do not wish to use LEAP Pangeo, the testbed is publicly accessible on the Open Storage Network (OSN) pod. In order to access the testbed on the OSN pod, switch the "MLinputs_path" (in notebook 02) to the following path: https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs. 

The testbed is structured as follows: within the "post01_xgb_inputs" folder, there are 9 sub-folders, one for each of the Earth System Models (ESM) in the testbed (see overview below; see also Heimdal et al., 2025). Within each of the 9 folders, there are sub-folders representing each member per ESM. Note that the number of members vary between each ESM. Within each member folder, there is one pickle file including the target and all driver variables. There is a total of 45 individual pickle files (one per member). 

ACCESS-ESM1-5:
  member_r4i1p1f1,
  member_r5i1p1f1

CESM2:
  member_r10i1p1f1,
  member_r11i1p1f1,
  member_r4i1p1f1,

CESM2-WACCM:
  member_r1i1p1f1
  member_r2i1p1f1
  member_r3i1p1f1

CMCC-ESM2:
  member_r1i1p1f1

CanESM5:
  member_r10i1p2f1
  member_r1i1p1f1
  member_r1i1p2f1
  member_r2i1p1f1
  member_r2i1p2f1
  member_r3i1p1f1
  member_r3i1p2f1
  member_r4i1p1f1
  member_r4i1p2f1
  member_r5i1p1f1
  member_r5i1p2f1
  member_r6i1p1f1
  member_r6i1p2f1
  member_r7i1p1f1
  member_r7i1p2f1
  member_r8i1p1f1
  member_r8i1p2f1
  member_r9i1p2f1

GFDL-ESM4:
  member_r1i1p1f1

MPI-ESM1-2-LR:
  member_r11i1p1f1
  member_r12i1p1f1
  member_r14i1p1f1
  member_r15i1p1f1
  member_r16i1p1f1
  member_r22i1p1f1
  member_r23i1p1f1
  member_r26i1p1f1
  member_r27i1p1f1

UKESM1-0-LL:
  member_r1i1p1f2
  member_r2i1p1f2
  member_r3i1p1f2
  member_r4i1p1f2
  member_r8i1p1f2

Here is an example of how to open one of the pickle files:

df = pd.read_pickle('https://nyu1.osn.mghpcc.org/leap-pangeo-manual/pco2_all_members_1982-2022/post01_xgb_inputs/ACCESS-ESM1-5/member_r5i1p1f1/MLinput_ACCESS-ESM1-5_r5i1p1f1_mon_1x1_198202_202312.pkl')

To access any of the other members, swap out the member and ESM names.

It is also possible to download the pickle files by clicking the invidual links. Each pickle file occupy about 5Gb of storage.

The folder named "Figures_Heimdal_et_al_2025" includes notebooks for figures and calculations presented in Heimdal et al. (2025).
