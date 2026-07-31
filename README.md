# Recommendations for the Adoption of GPU Computing by Small Research Teams in the Scientific Community

Repository of computer codes.


Trial codes:

- basic operations (add, multiply, sqrt, power, log, exp)
- sum vector
- dgemm
- stopping power

Evaluation codes:

- 2D heat equation
- 3D toy molecular dynamics 


## Trial 001 - Basic Operations

AMD Ryzen 5, 6 core CPU with NVIDIA RTX3050 GPU

![Description](readme_images/trial_001_workstation/Basic_Operations_add_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_workstation/Basic_Operations_divide_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_workstation/Basic_Operations_exp_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_workstation/Basic_Operations_log_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_workstation/Basic_Operations_multiply_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_workstation/Basic_Operations_power_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_workstation/Basic_Operations_sqrt_performance_max_iterations_per_second.png)

AMD EPYC, 32 core CPU with AMD Instinct MI200 GPU

![Description](readme_images/trial_001_hpc/Basic_Operations_add_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_hpc/Basic_Operations_divide_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_hpc/Basic_Operations_exp_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_hpc/Basic_Operations_log_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_hpc/Basic_Operations_multiply_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_hpc/Basic_Operations_power_performance_max_iterations_per_second.png)
![Description](readme_images/trial_001_hpc/Basic_Operations_sqrt_performance_max_iterations_per_second.png)


## Trial 002 - Reduction

AMD Ryzen 5, 6 core CPU with NVIDIA RTX3050 GPU

![Description](readme_images/trial_002/Sum_Vector_sum_performance_max_iterations_per_second.png)


## Trial 003 - Matrix Multiplication

AMD Ryzen 5, 6 core CPU with NVIDIA RTX3050 GPU

![Description](readme_images/trial_003/GEMM_128x128_by_128x128_performance_max_iterations_per_second.png)
![Description](readme_images/trial_003/GEMM_1000x800_by_800x1200_performance_max_mem.png)
![Description](readme_images/trial_003/GEMM_1024x1024_by_1024x1024_performance_max_mem.png)
![Description](readme_images/trial_003/GEMM_4096x4096_by_4096x4096_performance_max_mem.png)


## Trial 004 - Bethe-Bloch Stopping Power

AMD Ryzen 5, 6 core CPU with NVIDIA RTX3050 GPU

![Description](readme_images/trial_004_workstation/Bethe-Bloch_Stopping_Power_Bethe-Bloch_Stopping_Power_performance_max_iterations_per_second.png)
![Description](readme_images/trial_004_workstation/complexity_characters_per_file.png)


## Trial 004 - Bethe-Bloch Stopping Power

AMD EPYC, 32 core CPU with AMD Instinct MI200 GPU

![Description](readme_images/trial_004_hpc/Bethe-Bloch_Stopping_Power_Bethe-Bloch_Stopping_Power_performance_max_iterations_per_second.png)
![Description](readme_images/trial_004_hpc/complexity_characters_per_file.png)


## Heat2D

![Description](readme_images/heat2d/heatmap__difference_20.png)
![Description](readme_images/heat2d/heat2d_min_runtime.png)
![Description](readme_images/heat2d/heat2d_max_memory.png)
![Description](readme_images/heat2d/complexity_characters_per_file.png)



## SimpleMD

[![Watch SimpleMD](readme_images/simplemd/simplemd.gif)](readme_images/simplemd/simplemd.mp4)



