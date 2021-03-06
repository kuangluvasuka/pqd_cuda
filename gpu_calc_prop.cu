#include "pqd1.h"
#include <cuda.h>
#include <cuda_runtime.h>


/* GPU kernel prototypes */
__global__ void gpu_pot_prop(double* psi, double* u);
__global__ void gpu_kin_prop(double* psi, double* wrk, double* al, double* blx, double* bux, int t);

extern "C" void gpu_init(int myid) {
    int dev_num;
	cudaSetDevice(myid % 2);
    cudaGetDevice(&dev_num);
    printf("myid is %d, GPU id is %d\n", myid, dev_num);
    
    cudaMalloc((void**) &dev_psi, sizeof(double) * 2 * (NX+2));
    cudaMalloc((void**) &dev_wrk, sizeof(double) * 2 * (NX+2));
    cudaMalloc((void**) &dev_u, sizeof(double) * 2 * (NX+2));
    cudaMalloc((void**) &dev_al, sizeof(double) * 2 * 2);
    cudaMalloc((void**) &dev_blx0, sizeof(double) * 2 * (NX+2));
    cudaMalloc((void**) &dev_blx1, sizeof(double) * 2 * (NX+2));
    cudaMalloc((void**) &dev_bux0, sizeof(double) * 2 * (NX+2));
    cudaMalloc((void**) &dev_bux1, sizeof(double) * 2 * (NX+2));
    
    cudaMemcpy2D(dev_u, 2*sizeof(double), u, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_al, 2*sizeof(double), al, 2*sizeof(double), 2*sizeof(double), 2, cudaMemcpyHostToDevice);

    cudaMemcpy2D(dev_blx0, 2*sizeof(double), blx[0], 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_blx1, 2*sizeof(double), blx[1], 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_bux0, 2*sizeof(double), bux[0], 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_bux1, 2*sizeof(double), bux[1], 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);

}



extern "C" void gpu_lanch_pot_prop() {
    cudaMemcpy2D(dev_psi, 2*sizeof(double), psi, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);

    int thread_per_block = 64;
    int num_block = NX / thread_per_block;
    
    gpu_pot_prop<<<num_block, thread_per_block>>>(dev_psi, dev_u);
    
    cudaMemcpy2D(psi, 2*sizeof(double), dev_psi, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyDeviceToHost);
}
    
extern "C" void gpu_lanch_kin_prop(int t) {
    cudaMemcpy2D(dev_psi, 2*sizeof(double), psi, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);

    int thread_per_block = 64;
    int num_block = NX / thread_per_block;
    
    if (t == 0) {
      gpu_kin_prop<<<num_block, thread_per_block>>>(dev_psi, dev_wrk, dev_al, dev_blx0, dev_bux0, t);

    } else {
      gpu_kin_prop<<<num_block, thread_per_block>>>(dev_psi, dev_wrk, dev_al, dev_blx1, dev_bux1, t);
    }

    cudaMemcpy2D(psi, 2*sizeof(double), dev_psi, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyDeviceToHost);

}

__global__ void gpu_pot_prop(double* psi, double* u)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int sx = tid + 1;
    double wr, wi;
    if (sx <= NX) {
        double* row_psi = (double*)((char*)psi + sx * 2 * sizeof(double));
        double* row_u = (double*)((char*)u + sx * 2 * sizeof(double));

        wr = row_u[0]*row_psi[0]-row_u[1]*row_psi[1];
		wi = row_u[0]*row_psi[1]+row_u[1]*row_psi[0];
		row_psi[0] = wr;
		row_psi[1] = wi;

    }

}

__global__ void gpu_kin_prop(double* psi, double* wrk, double* al, double* blx, double* bux, int t)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int sx = tid + 1;
	double wr,wi;
    if (sx <= NX) {
        double* row_blx = (double*)((char*)blx + sx * 2 * sizeof(double));
        double* row_bux = (double*)((char*)bux + sx * 2 * sizeof(double));
        double* row_psi = (double*)((char*)psi + sx * 2 * sizeof(double));
        double* row_psi_r = (double*)((char*)psi + (sx+1) * 2 * sizeof(double));
        double* row_psi_l = (double*)((char*)psi + (sx-1) * 2 * sizeof(double));
        double* row_al = (double*)((char*)al + t * 2 * sizeof(double));
        double* row_wrk = (double*)((char*)wrk + sx * 2 * sizeof(double)); 
        
        wr = row_al[0]*row_psi[0]-row_al[1]*row_psi[1];
        wi = row_al[0]*row_psi[1]+row_al[1]*row_psi[0];
        wr += (row_blx[0]*row_psi_l[0]-row_blx[1]*row_psi_l[1]);
        wi += (row_blx[0]*row_psi_l[1]+row_blx[1]*row_psi_l[0]);
        wr += (row_bux[0]*row_psi_r[0]-row_bux[1]*row_psi_r[1]);
        wi += (row_bux[0]*row_psi_r[1]+row_bux[1]*row_psi_r[0]);
        row_wrk[0] = wr;
        row_wrk[1] = wi;
	
	row_psi[0] = row_wrk[0];
	row_psi[1] = row_wrk[1];
    }

}





