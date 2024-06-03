# Communication-Dependent Computing Resource Management for Concurrent Task Orchestration in IoT Systems

This repository contains the codes for our work "Communication-Dependent Computing Resource Management for Concurrent Task Orchestration in IoT Systems".


 
# Requirements
The following versions have been tested: Python 3.7.16 + Pytorch 1.13.1+cu116. But newer versions should also be fine.



## The introduction of each file

### Configurations
`options.py`: The whole configuration of this project, and the parameters can be changed for better performance.

### Environment, server, and client settings

`Environment.py`: the multicell cellular wireless environment simulator.

`main_1.py`: run this main file to simulate the methods for CDC resource management and concurrent task orchestration.

`TG_server.py`: the TG server simulator.

`TC_client.py`: the TC client simulator.

### Data Processing
`datasets/cifar_mnist.py`: get the mnist and cifar10 datasets, then some split operations on them to distribute on TC clients and TG servers.


### Algorithms

`TD3.py`: Twin Delayed DDPG (TD3).

`DDPG.py`: Deep Deterministic Plocy gradient (DDPG).

More state-of-the-art MARL and baseline algorithms will be added.
