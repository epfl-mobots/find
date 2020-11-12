#!/bin/sh

export exp_d=$1;
export exp_ts=$2

./plots/plot_res_vel_distribution.py -p $exp_d -t $exp_ts
./plots/plot_angular_vel.py -p $exp_d -t $exp_ts
./plots/plot_res_acc_distribution.py -p $exp_d -t $exp_ts 
./plots/plot_distance_to_wall.py -p $exp_d -t $exp_ts

./plots/plot_grid_occupancy.py -p "`echo $exp_d`/*processed_positions.dat" --regex -o occupancy_real --open; 
./plots/plot_grid_occupancy.py -p "`echo $exp_d`/*generated_positions.dat" --regex -o occupancy_hybrid --open; 
./plots/plot_grid_occupancy.py -p "`echo $exp_d`/*generated_virtu_positions.dat" --regex -o occupancy_virtu --open; 