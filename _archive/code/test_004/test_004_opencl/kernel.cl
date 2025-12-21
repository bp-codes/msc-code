// kernel.cl

typedef struct {
    double x;
    double y;
    double z;
} ClVec3;

double distance(ClVec3 a, ClVec3 b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__kernel void compute(
    __global const ClVec3* points,
    __global const ClVec3* refs,
    const int num_refs,
    __global double* results
) {
    int idx = get_global_id(0);
    ClVec3 point = points[idx];
    double result_local = 0.0;

    for (int j = 0; j < num_refs; ++j) {
        ClVec3 ref = refs[j];
        double r = distance(point, ref);

        double term_1 = 0.5 * exp(-0.1 * r);
        double term_2 = pow(r, 0.33);
        double term_3 = 0.25 * r * r * r - 2.5 * r * r - 0.3 * r + 3.2;
        double term_4 = 0.22 * pow(r, -1.5);
        double term_5 = 0.1 * pow(r, -2.5);
        double term_6 = -0.2 * pow(r, -3.5);
        double term_7 = -0.8 * pow(r, 0.5);

        result_local = result_local + 1.0e-4 * (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7);
    }

    results[idx] = result_local;
}

