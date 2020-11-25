#!/bin/sh

export exp_d=$1;
export exp_ts=$2
export exp_r=$3

./find/plots/plot_res_vel_distribution.py -p $exp_d -t $exp_ts -r $exp_r; 
./find/plots/plot_angular_vel.py -p $exp_d -t $exp_ts -r $exp_r; 
./find/plots/plot_res_acc_distribution.py -p $exp_d -t $exp_ts -r $exp_r; 
./find/plots/plot_distance_to_wall.py -p $exp_d -t $exp_ts -r $exp_r; 

./find/plots/plot_relative_orientation.py -p $exp_d -t $exp_ts -r $exp_r
./find/plots/plot_interindividual_distance.py -p $exp_d -r $exp_r; 
./find/plots/plot_geometrical_leader_info.py -p "`echo $exp_d`/*generated_velocities.dat" --regex; 

./find/plots/plot_grid_occupancy.py -p "`echo $exp_d`/raw/*processed_positions.dat" --regex -o occupancy_real --open; 
./find/plots/plot_grid_occupancy.py -p "`echo $exp_d`/generated/*generated_positions.dat" --regex -o occupancy_hybrid --open; 
./find/plots/plot_grid_occupancy.py -p "`echo $exp_d`/generated/*generated_virtu_positions.dat" --regex -o occupancy_virtu --open; 




