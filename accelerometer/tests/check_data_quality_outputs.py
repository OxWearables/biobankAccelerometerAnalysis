import json

with open('data/sample-summary.json') as f:
      s = json.load(f)

if s['totalReads'] != 51048852:
    raise Exception("totalReads wrong for sample") 

if s['clipsBeforeCalibration'] != 166: 
    raise Exception("clipsBeforeCalibration wrong for sample")

if s['clipsAfterCalibration'] != 60:
    raise Exception("clipsAfterCalibration wrong for sample")
