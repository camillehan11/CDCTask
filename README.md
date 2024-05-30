# CDC Resource Management and Concurrent Task Orchestration 
Communication-Dependent Computing Resource Management for Concurrent Task Orchestration in IoT Systems

This repository contains the codes for our work "Communication-Dependent Computing Resource Management for Concurrent Task Orchestration in IoT Systems".

 


# Requirements
The following versions have been tested: Python 3.7.16 + Pytorch 1.13.1+cu116 But newer versions should also be fine.



## The introduction of each file


Environment, server, and client settings:

`Environment.py`: the multicell cellular wireless environment simulator.

`main_1.py`: run this main file to simulate the methods for CDC resource management and concurrent task orchestration.

`TG_server.py`: the TG server simulator.

`TC_client.py`: the TC client simulator.

The wireless environment simulator and dataset were taken from the repositories:
https://github.com/PeymanTehrani/FDRL-PC-Dyspan

https://github.com/LuminLiu/HierFL


MARL algorithms

`TD3.py`: Twin Delayed DDPG (TD3).

`DDPG.py`: Deep Deterministic Plocy gradient (DDPG).

More state-of-the-art MARL and baseline algorithms will be added.
