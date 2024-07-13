#!/bin/bash

# Original file name
original_file="./near_by.py"

# Start and end values
start=5.5
end=7.5

# Number of files (points in linspace)
copies=5

# Calculate delta
delta=$(echo "($end - $start) / ($copies - 1)" | bc -l)

for i in $(seq 0 $((copies - 1))); do
    # Current value to be inserted
    current_value=$(echo "$start + $delta * $i" | bc -l)
    echo $i

    # Format current value to a fixed number of decimal places if needed
    formatted_value=$(printf "%.2f" $current_value)

    # Create new file name
    new_file="run_$i.py"

    # Copy the original file to the new file
    cp $original_file $new_file

    # Replace text on the 5th line, adapting for the current value
    # Ensure the sed command handles potential floating point values properly
    sed -i "24s/distance_factor = [0-9]*\.[0-9]*/distance_factor = $formatted_value/" $new_file
done

