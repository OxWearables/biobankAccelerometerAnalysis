#!/bin/bash

if [[ $(diff java/Tests/Resources/sampleV2Epoch.csv java/Tests/Resources/sampleV2.gt3x) ]]; then
  exit 1
else
  exit 0
fi
