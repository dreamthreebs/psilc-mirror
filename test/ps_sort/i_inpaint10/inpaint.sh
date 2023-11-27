#! /bin/bash
# Define the variable
fold="1.0"
# numbers="40 95 155 215 270"
numbers="40"

# Use the variable in the file paths
mkdir ./${fold}

for number in $numbers; do
    mrs_alm_inpainting -v ../../../inpaintingdata/CMBPS5/${number}.fits ../i_mask10/${fold}/${number}.fits    ./${fold}/${number}.fits
done

date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)


