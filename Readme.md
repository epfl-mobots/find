# Fish INteraction moDeling framework (~> find)

This repo contains contains an attempt to unify multiple methodologies and packages to allow for easily pre-processing and modeling social interactions and behavioural responses of fish. While this could as well be adapted to any kind of social or behavioural interactions that can be observed from spatial data, this has been thorougly developed and tested for fish interactions.

To this point the framework contains the following discrete packages that help in the study and understanding of interactions:

1. Pre-processing code to filter out tracking system inaccuracies, smooth data, segment experimental files when tracking confidence is low and storage.
2. Modeling code that allows the user to choose a model if her/his choice (or build one) and quickly train it.
3. Simulation code that allows the user to thoroughly test the generated model against the real data either in a 'Real VS Model' comparison or 'Model VS Model' ability to produce the global dynamics and demonstrate the emergence of more complex patterns observed in the real data.
4. Plots to easily compare the results of steps 1, 2 and 3. 
5. **[Under development]** Behavioural tools that not only capture social interactions but behaviour as a whole along with intelligent integration in simulation. 


## Installation

- Open your favourite terminal and clone this repository:

    ```shell
    git clone https://github.com/bpapaspyros/find.git && cd find
    ```

    To use some of our data-sets you can clone one of the following:

    ```shell
    git clone git@github.com:epfl-mobots/plos_one_experiments.git data/ring
    ```

    ```shell
    git clone git@github.com:epfl-mobots/ncs_deep_learning_2022.git data/open_50cm
    ```

- **[Optional but suggested]** Create a virtual python environment:
    
    ```shell
    virtualenv -p /usr/bin/python3 venv
    ```

    Notice that *find* has been tested with **Python 3.9.7**. 



    Once you have created the environment go ahead and enable it:

    ```shell
    source venv/bin/activate
    ```
- Go ahead and install dependencies as follows: 



    For the core functionality only **[TF2 should detect a GPU if available and CUDA is installed]** : 
    
    ```shell
    pip install -e .
    ```

    Additional packages to install dependencies related to linting and plotting: 
    
    ```shell
    pip install -e '.[test, plot]'
    ```


## Pre-processing your data

You can use the available fish data to test this part or take a deeper look in the code and adapt your data and/or the code to go through this section. Before you process anything, take a look at the available options that the pre-processing script offers, as follows:

```shell
python -m find.utils.preprocess -h
```

For example, you can go ahead and pre-process the Hemigrammus rhodostomus data provided by our partners at the Université Toulouse III - Paul Sabatier in France:

```shell
python -m find.utils.preprocess -p data/open_50cm/rummy/pair/ -f 'raw_positions*2G*.dat' --fps 25 -c 3 --toulouse --radius 0.25
```

This should create a new folder at the current directory with the format `$hostname_$hour_$minute_$second`. Inside this folder you will find processed versions of your original data files along with a correspondance file letting you know which processed file corresponds to which raw file.


## Train a model to reproduce the interaction dynamics observed in the processed data

Before you start training models, you can take a look at the available models and training options by invoking the help function:

```shell
python -m find.models.trainer -h
```

For example, you can use a simple probabilistic LSTM structure as follows:

```shell
python -m find.models.trainer -p experiment_folder -t 0.12 -e 81 -d 1 -b 512 --model PLSTM
```

Notice that despite the `81` epoch limit, there are additional stopping criteria that you can edit by taking a look in the `trainer.py`.


## Simulations

Once you have a version of the model you can run simulations and already start plotting some results. The simulation module contains multiple option that you can see by invoking the following command:

```shell
python -m find.simulation.simulation -h
```

For example, assuming you provided data with multiple individuals, let's say 2, then you can run a hybrid simulation (i.e., one replayed trajectory plus the model interacting) as follows:

```shell
python -m find.simulation.simulation -p <path to the experiment> -r <path to a reference file> -t <timestep> --exclude_index <id of the individual to be replaced by the model>
```

or run a complety virtual (i.e., model VS model):

```shell
python -m find.simulation.simulation -p <path to the experiment> -r <path to a reference file> -t <timestep> --exclude_index -1 -i <number of timesteps to simulate>
```

Finally there is an option that allows you to create simulations with more individuals than the original dataset to study the scalability of the system. For example, you can do the following:

```shell
python -m find.simulation.simulation -p <path to the experiment> -r <path to a reference file> -t <timestep> --exclude_index -1 -i <number of timesteps to simulate> --num_extra_virtu 4
```

that would correspond to a simulation of 4 + the original number of individuals.
