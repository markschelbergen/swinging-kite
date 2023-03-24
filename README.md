# Swinging Motion of a Kite with Suspended Control Unit Flying Turning Manoeuvres

This repository contains the Python code used for compiling the paper "Swinging Motion of a Kite with Suspended Control Unit Flying Turning Manoeuvres" that has been submitted to a special issue of Energies on airborne wind energy systems. The code uses experimental flight data of Kitepower B.V. to impose the measured flight path of the wing.

The paper performs a detailed analysis of a specific figure-of-eight cross-wind manoeuvre of the 65th pumping cycle for which the flight data is provided in [20191008_0065_fig8.csv](20191008_0065_fig8.csv). Moreover, ten pumping cycles are studied in the paper and the corresponding data files are included in the [cycles](cycles) directory. The full flight data can be accessed at DOI: [10.4121/19376174](https://doi.org/10.4121/19376174).

Two models are used in this analysis: a [quasi-static](time_invariant_model.py) and [dynamic](dynamic_model.py) model, based on the papers of Williams [[1](#Williams)] and Zanon et al. [[2](#Zanon)], respectively. The dynamic model is implemented using [CasADi](https://github.com/casadi/casadi) to efficiently solve the motion of the kite and tether. Moreover, [CasADi](https://github.com/casadi/casadi) is used to pre-process the recorded kinematics of the wing using an optimal control problem. 

 <!-- This repository is archived at time of the publication of the paper and can be accessed at DOI:[](http://doi.org/)-->
## Preparing the Python environment

The code is tested in Python 3.11. It is recommended to use [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) for setting up the environment. The following instructions are for Linux.

### Installation using Anaconda

We assume that a version of Anaconda is installed on your machine. Set the name of your environment in [environment.yml](environment.yml) by a name of your choice and create the virtual environment using the following command:

 ```commandline
conda env create -f environment.yml
```

Activate the new environment to use it:
```commandline
conda activate [env_name]
```
in which [env_name] should be replaced by the chosen name (previously ```source activate [env_name]```).

All the required Python packages (listed in requirements.yml) are installed when creating the environment. Make sure that the new environment is active every time you run any of the Python scripts.

## Running the simulations and plotting the results
The [time_invariant_model.py](time_invariant_model.py) and [dynamic_simulation.py](dynamic_simulation.py) scripts output files in the [results](results) directory that are input to [compare_results.py](compare_results.py). Therefore, these scripts should be executed prior to generating the plots that compare the simulation results of the two models. The table below lists the scripts that need to be executed to generate the figures of the paper.

| Script                           | Output figure(s)   |
|:---------------------------------|:-------------------|
| turning_center.py                | 4, 6               |
| time_invariant_model.py          | 7, 12              |
| dynamic_simulation.py            | 10                 |
| compare_results.py               | 8, 9               |
| plot_steering_input_relations.py | 11                 |

To execute one of these scripts, open the directory of the project and use the following command:

```commandline
python [file_name]
```
in which [file_name] should be replaced by the file name of the script.

## References

<a name="Williams">[1]</a> Paul Williams (2017). Cable Modeling Approximations for Rapid Simulation. Journal of Guidance Control and Dynamics, 40:7, 1779-1788. [10.2514/1.G002354](https://doi.org/10.2514/1.G002354)

<a name="Zanon">[2]</a> Mario Zanon, Sébastien Gros, Joel Andersson and Moritz Diehl (2013). Airborne Wind Energy Based on Dual Airfoils. IEEE Transactions on Control Systems Technology, 21:4, 1215-1222. [10.1109/TCST.2013.2257781](https://doi.org/10.1109/TCST.2013.2257781)


## Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the grant agreement No. 691173 (REACH) and the Marie Sklodowska-Curie grant agreement No 642682 (AWESCO).
