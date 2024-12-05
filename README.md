<!-- =================================================
# Copyright (c) Heriot-Watt University and the University of Edinburgh
Authors  :: Hao Yu (yu1997hao@gmail.com), Zebin Huang
================================================= -->

## Introduction
This work expanded on our prior study [WCCI2024Impedance](https://arxiv.org/abs/2402.02904), in which we built a simulation tool for physical interaction between an elbow and an exoskeleton in [MyoSuite](https://sites.google.com/view/myosuite) to study the human similarity of a reinforcement learning (RL) agent in controlling an elbow musculoskeletal model. We newly added a stretch reflex controller unit into the simulation tool to enable the simulation of spasticity models on the virtual elbow. Afterwards, we simulated the robot-assisted constant-velocity stretch experiment for four types of spasticity models to explore the optimal modelling method of elbow spasticity.

## Installation
```
python3 -m venv .venv
# active your venv
git clone --recursive https://github.com/myo-manipulation/SpasticMyoElbow.git
pip install -r requirements
pip install -e .
```

## Code
The codes for this project is in 'dev' folder: 
1) use check_mode.ipynb to observe the model utilised in this project
2) use test_environment.py to test the configuration and installation of your environment 
3) new_model_validation.py is a further step of test_spasticity_model.py, which allows inputing a series of reference trajectories of elbow extension and recording the time, angle, and torque. It reads ref_data_0_1.pkl in 'data' folder. New_model_validation.py can generate the dataset of spasticity model validation saved in 'data/sim_data2'. You can observe the refernece dataset and simulation results through '/figures/ICORR_simdata2.ipynb'.
4) We then further improved new_model_validation.py to new_model_fitting.py to find the best parameters for the hybird model of spasticity.

## Dataset
The dataset for this project is in 'data' folder: 
1) data/passive_force: data for passive resistance calibration
2) data/sim_data1: validation dataset for the simple spasticity model 
3) data/sim_data2: validation dataset for the complete spasticity model
4) ref_data_0_1.pkl: the reference torque and angle data from a healthy people
5) ref_data_1.csv: the reference data extracted from 'Biomechanical parameters of the elbow stretch reflex in chronic hemiparetic stroke'.

## License
SpasticMyoElbow is licensed under the [Apache License](LICENSE).

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and cite our [arXiv paper](https://arxiv.org/abs/2205.13600)  by using the following BibTeX entrys.

```BibTeX
@Misc{SpasticMyoElbow2024,
}
```
