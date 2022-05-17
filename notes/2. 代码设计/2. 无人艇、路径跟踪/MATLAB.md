---
aliases: 
tags: 
date created: 2022-05-10 19:51:54
date modified: 2022-05-14 20:57:51
---

# MATLAB

文件结构:

- `main.mlx`: 入口文件, 用于**设定各项参数**, 调用simulink文件进行仿真, 并处理仿真结果
- `ADRC_Control.slx`: ADRC 控制 USV 进行路径跟踪
    - `USV_Model.slx` 无人艇数学模型
    - `ADRC.slx`: 非线性ADRC实现
        - 实现方式1: 3个模块TD, ESO, NLSEF由S函数编写. 方便观察控制器内部信号, 可**用于调试**, 仿真速度很慢. (**模块输出占大部分时间**)
            - `ADRC_ESO.m`
            - `ADRC_NLSEF.m`
            - `ADRC_TD.m`
        - 实现方式2: 控制器整体由S函数编写, 仿真速度快
            - `ADRC_AIO.m` "All in One"
- 其他文件
    - `run_data`文件夹提供了一个测试路径`path.csv`以及之前获得的结果. `matlab_workspace.mat`为程序执行后的工作区变量.
    - `LADRC_Control.slx` LADRC 控制 USV 进行路径跟踪 (测试用, 这个没怎么调, 效果一般)
    - `test.mlx`

说明:
线性 ADRC (LADRC) 是简化版本, 参数整定比较容易, 并且方便改进, 也可以用于航行器控制[^ladrc].
[^ladrc]:[Path following control of fully-actuated autonomous underwater vehicle in presence of fast-varying disturbances - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0141118718307752?via%3Dihub)
> It was observed that the LADRC technique does not give an effective transient and disturbance rejection response, when the system to be controlled has complex dynamics such as right half plane poles (unstable system), right half plane zeros (non-minimum phase systems) and time delay[^shortcoming].
[^shortcoming]: [Generalized Active Disturbance Rejection Control: Review, Applications and Challenges | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9465246)
