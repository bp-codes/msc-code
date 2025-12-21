

import subprocess



n_times = 5

# Run the executable and capture output



test_exes = [   "test_002_serial/main.x",
                "test_002_nvcc/main.x", 
                "test_002_cublas/main.x", 
                "test_002_opencl/main.x",
                "test_002_sycl/main.x",
                "test_002_openmp/main.x",
                "test_002_openmp_gpu/main.x"
            ]

fh = open("output.csv", "w")

for exe in test_exes:
    setup_time = 0.0
    calc_time = 0.0
    for i in range(n_times):
        result = subprocess.run([exe], capture_output=True, text=True)
        d = result.stdout.strip().split(",")
        print(exe, ",", d[0], ",", d[1], ",", d[2])
    
        setup_time = setup_time + float(d[0])
        calc_time = calc_time + float(d[1])

    setup_time = setup_time / n_times
    calc_time = calc_time / n_times

    fh.write(exe + "," + str(setup_time) + "," + str(calc_time) + "\n")
fh.close()











