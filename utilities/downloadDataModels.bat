rem ------------------------------------------------------------------
rem [Aiden Doherty] Download sample data and activity models
rem ------------------------------------------------------------------

set downloadDir=https://gas.ndph.ox.ac.uk/aidend/accModels/
rem download sample data file
if not exist "data\sample.cwa.gz" (
    curl -o data\sample.cwa.gz %downloadDir%sample.cwa.gz
)

rem delete and newly download activity model files
rm activityModels\doherty-may20.tar
rm activityModels\willetts-may20.tar
rm activityModels\walmsley-nov20.tar
rm activityModels\doherty-jan21.tar
rm activityModels\willetts-jan21.tar
rm activityModels\walmsley-jan21.tar
curl -o activityModels\doherty-may20.tar %downloadDir%doherty-may20.tar
curl -o activityModels\willetts-may20.tar %downloadDir%willetts-may20.tar
curl -o activityModels\walmsley-nov20.tar %downloadDir%walmsley-nov20.tar
curl -o activityModels\doherty-jan21.tar %downloadDir%doherty-jan21.tar
curl -o activityModels\willetts-jan21.tar %downloadDir%willetts-jan21.tar
curl -o activityModels\walmsley-jan21.tar %downloadDir%walmsley-jan21.tar

rem download sample training file
if not exist "activityModels\labelled-acc-epochs.csv" (
    curl -o activityModels\labelled-acc-epochs.csv %downloadDir%labelled-acc-epochs.csv
)
