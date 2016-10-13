#include "pqd1.h"



__global__ void gpu_pot_prop(double* psi, double* u)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int sx = tid + 1;
    double wr, wx;
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
        double* row_blx = (double*)((char*)blx + t * NX * 2 * sizeof(double) + sx * 2 * sizeof(double));
        double* row_bux = (double*)((char*)bux + t * NX * 2 * sizeof(double) + sx * 2 * sizeof(double));
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
    }

}





