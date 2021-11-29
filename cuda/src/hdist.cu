#include <stdio.h>
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>
#define DEBUGx
/* 2 algo to compute */

/*
* Jacobi: t[i,j]' = 0.25 * (t[i-1,j]+t[i,j-1]+t[i+1,j]+t[i,j+1])
* Sor: t[i,j]'= t[i,j] + (t[i-1,j]+t[i,j-1]+t[i+1,j]+t[i,j+1] - 4*t[i,j])/w
* w = 2: converge faster
* w < 2: diverge
*/

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}


// global buffer indicator
__device__ int current_buffer = 0;
int current_buffer_h = 0;


// grid[i, j]: return by value
__device__ double get_grid_at_2index(int i, int j, double *data0, double *data1, int room_size) {
    if (current_buffer == 0) return data0[i * room_size + j];
    else if (current_buffer == 1) return data1[i * room_size + j];
}

// __device__ &double get_grid_at_3index(int i, int j, double *data0, double *data1, int room_size) {
//     if (current_buffer == 0) return data1[i * room_size + j];
//     else if (current_buffer == 1) return data0[i * room_size + j];
// }


// update a single point (called by device)
__device__ double update_single_d(size_t i, size_t j, double *data0, double *data1, 
    int room_size, int source_x, int source_y, float source_temp, float border_temp, float sor_constant, int algo) {
    double temp;
    if (i == 0 || j == 0 || i == room_size - 1 || j == room_size - 1) {
        temp = border_temp;
    } else if (i == source_x && j == source_y) {
        temp = source_temp;
    } else {
        auto sum = get_grid_at_2index(i+1,j,data0,data1,room_size) 
                 + get_grid_at_2index(i-1,j,data0,data1,room_size) 
                 + get_grid_at_2index(i,j+1,data0,data1,room_size)
                 + get_grid_at_2index(i,j-1,data0,data1,room_size);
        switch (algo) {
            case 0:
                temp = 0.25 * sum;
                break;
            case 1:
                temp = get_grid_at_2index(i,j,data0,data1,room_size) + (1.0 / sor_constant) * (sum - 4.0 * get_grid_at_2index(i,j,data0,data1,room_size));
                break;
        }
    }
    return temp;
}


// kernel launched function by host
// Jacobi = 0, Sor = 1 (algor)
__global__ void calculation_kernel(int room_size, float block_size, int source_x, int source_y, float source_temp, float border_temp, float tolerance, float sor_constant, int algo, 
    double *data0, double *data1, int k) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid(); 
    switch (algo) {
        case 0:
            for (size_t i = index; i < room_size; i+=stride) {
                for (size_t j = 0; j < room_size; ++j) {
                    //auto temp = update_single_d(i, j, grid, state);
                    auto temp = update_single_d(i, j, data0, data1, room_size, source_x, source_y, source_temp, border_temp, sor_constant, algo);
                    // grid[{alt, i, j}] = temp;
                    if (current_buffer == 0) data1[i * room_size + j] = temp;
                    else if (current_buffer == 1) data0[i * room_size + j] = temp;
                }
            }
            //g.sync();
            //if (index == 0)
                //current_buffer = !current_buffer;
            break;
        case 1:
            // odd-even turn
            //for (int k = 0; k <= 1; k++) {
            for (size_t i = index; i < room_size; i+=stride) {
                for (size_t j = 0; j < room_size; j++) {
                    if (k == ((i + j) & 1)) {
                        auto temp = update_single_d(i, j, data0, data1, room_size, source_x, source_y, source_temp, border_temp, sor_constant, algo);
                        if (current_buffer == 1) data0[i * room_size + j] = temp;
                        else if (current_buffer == 0) data1[i * room_size + j] = temp;
                    } else {
                        if (current_buffer == 1) data0[i * room_size + j] = get_grid_at_2index(i,j,data0,data1,room_size);
                        else if (current_buffer == 0) data1[i * room_size + j] = get_grid_at_2index(i,j,data0,data1,room_size);
                    }
                }
            }
                //g.sync();
                // if (index == 0)
                //     current_buffer = !current_buffer;
            //}
    }
}


// host: memory allocation in device (data0_d & data1_d)
__host__ void host_call_memory_copy_to_device(int room_size, double *data0, double *data1, double **data0_d, double **data1_d){
    // malloc new memory in device
    CHECK(cudaMalloc((double**)data0_d, sizeof(double) * room_size * room_size));
    CHECK(cudaMalloc((double**)data1_d, sizeof(double) * room_size * room_size));

    // copy HOST memory to DEVICE memory
    CHECK(cudaMemcpy(*data1_d, data1, sizeof(double)*room_size*room_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(*data0_d, data0, sizeof(double)*room_size*room_size, cudaMemcpyHostToDevice));
}

__host__ void host_call_memory_copy_from_device(int room_size, double *data0, double *data1, double *data0_d, double *data1_d){
    #ifdef DEBUG
    // printf("before copy to host\n");
    // for (int i = 0; i < room_size; ++i) {
    //     for (int j = 0; j < room_size; ++j) {
    //         printf("index [%d, %d] is %f %f\n", i, j, data0_d[i*room_size+j], data1_d[i*room_size+j]);
    //     }
    // }
    #endif
    CHECK(cudaMemcpy(data0, data0_d, sizeof(double)*room_size*room_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(data1, data1_d, sizeof(double)*room_size*room_size, cudaMemcpyDeviceToHost));
}


//host: memory clean and cuda reset
__host__ void host_call_CUDA_clean(double *data0_d, double *data1_d) {
    CHECK(cudaFree(data0_d));
    CHECK(cudaFree(data1_d));
    CHECK(cudaDeviceReset());
}



// host: calculation entry
__host__ void host_call_calculate_entry(int room_size, float block_size, int source_x, int source_y, 
    float source_temp, float border_temp, float tolerance, float sor_constant, int algo, 
    double *data0_d, double *data1_d, int nElem) {
    int blocksize = nElem;
    int gridsize = room_size / nElem;
    int k = 0;
    if (algo == 0) {
        calculation_kernel<<<gridsize, blocksize>>>(room_size, block_size, source_x, source_y, source_temp, border_temp, tolerance, sor_constant, algo, data0_d, data1_d, k);
        CHECK(cudaDeviceSynchronize());
        current_buffer_h = !current_buffer_h;
        CHECK(cudaMemcpyToSymbol(current_buffer, &current_buffer_h, sizeof(int)));
    }
    else if (algo == 1) {
        calculation_kernel<<<gridsize, blocksize>>>(room_size, block_size, source_x, source_y, source_temp, border_temp, tolerance, sor_constant, algo, data0_d, data1_d, k);
        CHECK(cudaDeviceSynchronize());
        current_buffer_h = !current_buffer_h;
        CHECK(cudaMemcpyToSymbol(current_buffer, &current_buffer_h, sizeof(int)));
        k = !k;
        
        calculation_kernel<<<gridsize, blocksize>>>(room_size, block_size, source_x, source_y, source_temp, border_temp, tolerance, sor_constant, algo, data0_d, data1_d, k);
        CHECK(cudaDeviceSynchronize());
        current_buffer_h = !current_buffer_h;
        CHECK(cudaMemcpyToSymbol(current_buffer, &current_buffer_h, sizeof(int)));
        k = !k;
    }
}

