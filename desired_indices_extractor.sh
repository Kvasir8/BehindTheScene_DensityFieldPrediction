#!/bin/bash

# Desired indices
indices1=(81 265 266 430 431 446 506 507 512 514 531 537 567 568 574 599)
indices2=(29 30 39 51 57 70 102 121 145 196 203 214 250 294 306 361 397 418 475)

# Base directory
base_dir="/storage/group/dataset_mirrors/01_incoming/kitti_360/KITTI-360/data_2d_raw/"  # Adjust to your root directory if different

# Destination directory (you'll need to specify this)
destination_dir1="/usr/stud/hank/storage/user/BehindTheScenes/BTS_imgs/kitti360"  # Adjust to where you want to copy indices1 files
destination_dir2="/usr/stud/hank/storage/user/BehindTheScenes/BTS_imgs/topdown"  # Adjust to where you want to copy indices2 files

# Create destination directories if they don't exist
mkdir -p "$destination_dir1"
mkdir -p "$destination_dir2"

# Extracting files based on indices1
for index in "${indices1[@]}"; do
    filename=$(printf "000000%04d.png" $index)
    for drive in "${base_dir}"2013_05_28_drive_*_sync; do
        filepath="${drive}/image_00/data_192x640/${filename}"
        if [[ -f "$filepath" ]]; then
            cp "$filepath" "$destination_dir1"
        fi
    done
done

# Extracting files based on indices2
for index in "${indices2[@]}"; do
    filename=$(printf "000000%04d.png" $index)
    for drive in "${base_dir}"2013_05_28_drive_*_sync; do
        filepath="${drive}/image_00/data_192x640/${filename}"
        if [[ -f "$filepath" ]]; then
            cp "$filepath" "$destination_dir2"
        fi
    done
done
