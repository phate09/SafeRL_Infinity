# SafeRL_Infinity #
Repository for the paper "Verifying Reinforcement Learning up to Inifinity"

```
conda create -n SafeRL_Infinity python=3.7

conda activate SafeRL_Infinity

pip install -r requirements.txt

conda config --add channels http://conda.anaconda.org/gurobi

conda install gurobi
```

##Paper results
download and unzip [experiment_collection_final.zip](https://mega.nz/file/11xigbgS#ld3MSRAHXUBl1lY-6ltgnmwtm53ESBfL_1WAqAJUSxc) in the 'save' directory

run `tensorboard --logdir=./save/experiment_collection_final`

##Train neural networks from scratch
run either:
* `training/tune_train_PPO_bouncing_ball.py`
* `training/tune_train_PPO_car.py`
* `training/tune_train_PPO_cartpole.py`

##Check safety of pretrained agents
download and unzip [pretrained_agents.zip](https://mega.nz/file/014wHRbB#3OHzROCTyPcq_1lVKVryGgkuRaPBfhme4j7n6GGrWoc) in the 'save' directory

run verification/run_tune_experiments.py

(to monitor the progress of the algorithm run `tensorboard --logdir=./save/experiment_collection_final`)