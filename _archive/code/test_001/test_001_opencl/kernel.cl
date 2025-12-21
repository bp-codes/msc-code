__kernel void exp_kernel(__global double* results, int count) {
    int i = get_global_id(0);
    if (i < count) {
        int idx = i % 100;
        results[i] = exp((double)idx / 10.0);
    }
}
 
