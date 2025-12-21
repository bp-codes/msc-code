
import os
import subprocess

os.environ["OMP_NUM_THREADS"] = "10" 

n_times = 5

# Run the executable and capture output


test_dirs = [   
                "test_004_serial",
                "test_004_nvcc",
                "test_004_opencl",
                "test_004_sycl",
                "test_004_sycl32",
                "test_004_sycl16",
                "test_004_openmp",
                "test_004_thread",
                "test_004_openmp_gpu"
            ]

fh = open("output.csv", "w")

for dir in test_dirs:

    os.chdir(dir)

    setup_time = 0.0
    calc_time = 0.0
    for i in range(n_times):
        result = subprocess.run("./main.x", capture_output=True, text=True)
        print(result)
        d = result.stdout.strip().split(",")
        print("main.x", ",", d[0], ",", d[1], ",", d[2])
    
        setup_time = setup_time + float(d[0])
        calc_time = calc_time + float(d[1])

    setup_time = setup_time / n_times
    calc_time = calc_time / n_times

    fh.write(dir + "," + str(setup_time) + "," + str(calc_time) + "\n")

    os.chdir("../")

fh.close()











