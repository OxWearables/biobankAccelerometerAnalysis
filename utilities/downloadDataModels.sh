#!/bin/bash
# ------------------------------------------------------------------
# [Aiden Doherty] Download sample data and activity models
# ------------------------------------------------------------------

downloadDir="http://gas.ndph.ox.ac.uk/aidend/accModels/"
# download sample data file
if ! [ -f "data/sample.cwa.gz" ]
then
    wget ${downloadDir}sample.cwa.gz -P data/
fi

# delete and newly download activity model files
rm activityModels/doherty-may20.tar
rm activityModels/willetts-may20.tar
wget ${downloadDir}doherty-may20.tar -P activityModels/
wget ${downloadDir}willetts-may20.tar -P activityModels/

# download sample training file
if ! [ -f "activityModels/labelled-acc-epochs.csv" ]
then
    wget ${downloadDir}labelled-acc-epochs.csv -P activityModels/
fi
