#!/bin/bash

# Path to the original file you want to modify
base_file="./submit.sh"

for rlz_idx in {0..99}; do
    # Use sed to replace "number = \"0\"" with "number = \"${rlz_idx}\"" in the base file
    # and save the output to a new file named "submit_${rlz_idx}"
    sed "43s/number=\"0\"/number=\"${rlz_idx}\"/" "$base_file" > "submit_${rlz_idx}.sh"
done

