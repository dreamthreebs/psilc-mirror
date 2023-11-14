#!/bin/bash

# Define the variable
fold="0.9"
numbers="40 95 155 215 270"

# Use the variable in the file paths
for number in $numbers; do
    mrs_alm_inpainting -v ./inputcmb/1.8/${number}.fits ../../FG5/strongps/psmaskfits/${fold}/${number}.fits ./outputcmb/${fold}/${number}.fits
done

