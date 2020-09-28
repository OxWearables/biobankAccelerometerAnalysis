#!/bin/bash

if [[ $(diff java/Tests/Resources/sampleV2Epoch.csv java/Tests/Resources/sampleV2EpochRef.csv) ]]; then
  exit 1
else
  exit 0
fi
