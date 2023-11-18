#!/bin/bash

# Define the variable
fold="1.0"
numbers="40 95 155 215 270"

# Use the variable in the file paths
for number in $numbers; do
    mrs_alm_inpainting -v ./inputcmb/smoothcmb/${number}.fits ../../FG5/strongps/psmaskfits/1.5/215.fits ./outputcmb/smoothcmb/${number}.fits
done

