#!/bin/bash

# Define the variable
fold="1.5"
# numbers="40 95 155 215 270"
numbers="40"

# Use the variable in the file paths

for number in $numbers; do
    mrs_alm_inpainting -v ./${number}.fits ../../FG5/strongps/psmaskfits2048/${fold}/95.fits ./INPAINT/${fold}/${number}.fits
done

