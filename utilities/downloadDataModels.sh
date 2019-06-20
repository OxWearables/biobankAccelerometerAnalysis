#!/bin/bash
# ------------------------------------------------------------------
# [Aiden Doherty] Download sample data and activity models
# ------------------------------------------------------------------

downloadDir="http://gas.ndph.ox.ac.uk/aidend/accModels/"
# download sample data file
wget ${downloadDir}sample.cwa.gz -P data/

# download activity model files
wget ${downloadDir}doherty2018.tar -P activityModels/
wget ${downloadDir}willetts2018.tar -P activityModels/

# download sample training file
wget ${downloadDir}labelled-acc-epochs.csv -P activityModels/
