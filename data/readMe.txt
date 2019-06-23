Data associated with a model are listed in different directories indicates by the respective model name. The directories are named as follows: 

    - fish_only                    -> fish-only experiments
    - Follower_model               -> Follower model (FM) experiments
    - Despotic_model               -> Despotic model (DM) experiments
    - Feedback-Initiative_model    -> Feedback-Initiative model (FIM) experiments

In each directory we include:

    - The trajectories extracted by idTracker (Perez et al. 2014) that were used for the analysis.
    - The robot index in the matrix of trajectories for mixed group experiments (robot_index.txt).
   
Concerning idTracker data
-------------------------

> idTracker outputs two different files: (1) trajectories.txt and (2) trajectories_nogaps.txt. The former includes the raw trajectories as reported by the tracking algorithm with NaN values for timesteps where an individual was not found, while the latter will attempt to fill in those missing values (there are still cases where NaN values appear). The equivalent .mat files are also included.

> For both of these files the format is a matrix of "Xi Yi ProbIdi" columns, where i the id of the individual for a specific replicate. In the case of (1) the ProbIdi is always NaN. In the case of (2) ProbIdi can take the values:

    - NaN 
    - 0-1 
    - -1
    - -2

An extract from the text in https://github.com/idTracker/idTracker/blob/e20059b8a6458828624d3fea556bc495a73a5e5c/src/despedida.m#L63 explains the values in more detail:

"... 
Difference between trajectories and trajectories_nogaps. The files called "trajectories" contain only the position of each individual when it is not occluded. The files called "trajectories_nogaps" contain the position of each individual also when occluded. The probability of correct identity contains a negative number when the position comes from an estimation. -1 means that the animal was occluded, but a centroid was found after resegmentation of the image. -2 means that the image could not be resegmented, so the position of the centroid is not very accurate. See the paper [1] for more information. There may be small differences between "trajectories" and "trajectories_nogaps" even in the non-occluded frames, due to a correction algorithm during the estimation of occluded centroids.

[1] Pérez-Escudero, Vicente-Page, Hinz, Arganda, de Polavieja. idTracker: Tracking individuals in a group by automatic identification of unmarked animals. Nature Methods 11(7):743-748 (2014)'),'idTracker - About the output files') 
..."

> In all experiments we use the "trajectories_nogaps" files.

