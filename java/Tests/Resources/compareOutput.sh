#!/bin/bash

if [[ $(diff java/Tests/Resources/sampleV2Epoch.csv java/Tests/Resources/sampleV2.gt3x) ]]; then
  exit 0
else
  exit 1
fi
