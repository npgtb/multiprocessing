__kernel void add_matrix(__global float* m1, __global float* m2, __global float* output, const int size){
    int i = get_global_id(0);
    if(i < size){
        output[i] = m1[i] + m2[i];
    }
}