/*******************************************************************************
Quantum dynamics (QD) simulation of an electron in one dimension.

USAGE

%mpicc -o pqd1 pqd1.c -lm
%mpirun -np ### pqd1 (see pqd1.h for the format of the input file, pqd1.in)
*******************************************************************************/
#include "pqd1.h"

#define USE_GPU_PROP 1

int main(int argc, char **argv) {
	int step; /* Simulation loop iteration index */
	double cpu1;

	MPI_Init(&argc,&argv); /* Initialize the MPI environment */
	MPI_Comm_rank(MPI_COMM_WORLD,&myid); /* My process ID */
	MPI_Comm_size(MPI_COMM_WORLD,&nproc); /* Number of processors */

	init_param();  /* Read input parameters */
	init_prop();   /* Initialize the kinetic & potential propagators */
	init_wavefn(); /* Initialize the electron wave function */

	if (USE_GPU_PROP == 1) {
		int dev_num;
		cudaSetDevice(myid % 2);
		cudaGetDevice(&dev_num);
		printf("myid is %d, GPU id is %d", myid, dev_num);
		
		cudaMalloc((void**) &dev_psi, sizeof(double) * 2 * (NX+2));
		cudaMalloc((void**) &dev_wrk, sizeof(double) * 2 * (NX+2));
		cudaMalloc((void**) &dev_u, sizeof(double) * 2 * (NX+2));
		cudaMalloc((void**) &dev_al, sizeof(double) * 2 * 2);
		cudaMalloc((void**) &dev_blx, sizeof(double) * 2 * (NX+2) * 2);
		cudaMalloc((void**) &dev_bux, sizeof(double) * 2 * (NX+2) * 2);
		
		cudaMemcpy2D(dev_u, 2*sizeof(double), u, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);
		cudaMemcpy2D(dev_al, 2*sizeof(double), u, 2*sizeof(double), 2*sizeof(double), 2, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_blx, blx, 2*(NX+2)*2*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_bux, bux, 2*(NX+2)*2*sizeof(double), cudaMemcpyHostToDevice);
	}

	cpu1 = MPI_Wtime();
	for (step=1; step<=NSTEP; step++) {
		single_step(); /* Time propagation for one step, DT */
		if (step%NECAL==0) {
			calc_energy();
			if (myid==0) printf("%le %le %le %le\n",DT*step,ekin,epot,etot);
		}
	}
	if (myid == 0) printf("CPU = %le\n",MPI_Wtime() - cpu1);

	MPI_Finalize(); /* Clean up the MPI environment */

	return 0;
}

/*----------------------------------------------------------------------------*/
void init_param() {
/*------------------------------------------------------------------------------
	Initializes parameters by reading them from standard input.
------------------------------------------------------------------------------*/
	FILE *fp;

	/* Read control parameters */
	fp = fopen("pqd1.in","r");
	fscanf(fp,"%le",&LX);
	fscanf(fp,"%le",&DT);
	fscanf(fp,"%d",&NSTEP);
	fscanf(fp,"%d",&NECAL);
	fscanf(fp,"%le%le%le",&X0,&S0,&E0);
	fscanf(fp,"%le%le",&BH,&BW);
	fscanf(fp,"%le",&EH);
	fclose(fp);

	/* Calculate the mesh size */
	dx = LX/NX;
}

/*----------------------------------------------------------------------------*/
void init_prop() {
/*------------------------------------------------------------------------------
	Initializes the kinetic & potential propagators.
------------------------------------------------------------------------------*/
	int stp,s,i,up,lw;
	double a,exp_p[2],ep[2],em[2];
	double x;

	/* Set up kinetic propagators */
	a = 0.5/(dx*dx);

	for (stp=0; stp<2; stp++) { /* Loop over half & full steps */
		exp_p[0] = cos(-(stp+1)*DT*a);
		exp_p[1] = sin(-(stp+1)*DT*a);
		ep[0] = 0.5*(1.0+exp_p[0]);
		ep[1] = 0.5*exp_p[1];
		em[0] = 0.5*(1.0-exp_p[0]);
		em[1] = -0.5*exp_p[1];

		/* Diagonal propagator */
		for (s=0; s<2; s++) al[stp][s] = ep[s];

		/* Upper & lower subdiagonal propagators */
		for (i=1; i<=NX; i++) { /* Loop over mesh points */
			if (stp==0) { /* Half-step */
				up = i%2;     /* Odd mesh point has upper off-diagonal */
				lw = (i+1)%2; /* Even               lower              */
			}
			else { /* Full step */
				up = (i+1)%2; /* Even mesh point has upper off-diagonal */
				lw = i%2;     /* Odd                 lower              */
			}
			for (s=0; s<2; s++) {
				bux[stp][i][s] = up*em[s];
				blx[stp][i][s] = lw*em[s];
			}
		} /* Endfor mesh points, i */
	} /* Endfor half & full steps, stp */

	/* Set up potential propagator */
	for (i=1; i<=NX; i++) {
		x = dx*i + LX*myid; /* Add processor origin to get global position */
		/* Construct the edge potential */
		/* Only at the two ends of the total simulation box */
		if ((myid==0 && i==1) || (myid==nproc-1 && i==NX))
			v[i] = EH;
		/* Construct the barrier potential */
		/* Total simulation-box length = LX*nproc */
		else if (0.5*(LX*nproc-BW)<x && x<0.5*(LX*nproc+BW))
			v[i] = BH;
		else
			v[i] = 0.0;
		/* Half-step potential propagator */
		u[i][0] = cos(-0.5*DT*v[i]);
		u[i][1] = sin(-0.5*DT*v[i]);
	}
}

/*----------------------------------------------------------------------------*/
void init_wavefn() {
/*------------------------------------------------------------------------------
	Initializes the wave function as a traveling Gaussian wave packet.
------------------------------------------------------------------------------*/
	int sx,s;
	double x,gauss,lpsisq,psisq,norm_fac;

	/* Calculate the the wave function value mesh point-by-point */
	for (sx=1; sx<=NX; sx++) {
		x = LX*myid+dx*sx-X0; /* Add processor origin to get global position */
		gauss = exp(-0.25*x*x/(S0*S0));
		psi[sx][0] = gauss*cos(sqrt(2.0*E0)*x);
		psi[sx][1] = gauss*sin(sqrt(2.0*E0)*x);
	}

	/* Normalize the wave function */
	lpsisq=0.0;
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<2; s++)
			lpsisq += psi[sx][s]*psi[sx][s];
	lpsisq *= dx;
	/* Global sum */
	MPI_Allreduce(&lpsisq,&psisq,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	norm_fac = 1.0/sqrt(psisq);
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<2; s++)
			psi[sx][s] *= norm_fac;
}

/*----------------------------------------------------------------------------*/
void single_step() {
/*------------------------------------------------------------------------------
	Propagates the electron wave function for a unit time step, DT.
------------------------------------------------------------------------------*/
	pot_prop();  /* half step potential propagation */

	kin_prop(0); /* half step kinetic propagation   */
	kin_prop(1); /* full                            */
	kin_prop(0); /* half                            */

	pot_prop();  /* half step potential propagation */
}

/*----------------------------------------------------------------------------*/
void pot_prop() {
/*------------------------------------------------------------------------------
	Potential propagator for a half time step, DT/2.
------------------------------------------------------------------------------*/
	if (USE_GPU_PROP == 0) {
		int sx;
		double wr,wi;

		for (sx=1; sx<=NX; sx++) {
			wr=u[sx][0]*psi[sx][0]-u[sx][1]*psi[sx][1];
			wi=u[sx][0]*psi[sx][1]+u[sx][1]*psi[sx][0];
			psi[sx][0]=wr;
			psi[sx][1]=wi;
		}
	} else {
		cudaMemcpy2D(dev_psi, 2*sizeof(double), psi, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);
		
		gpu_pot_prop<<<1, NX>>>(dev_psi, dev_u);
		
		cudaMemcpy2D(psi, 2*sizeof(double), dev_psi, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyDeviceToHost);
		
	}
}

/*----------------------------------------------------------------------------*/
void kin_prop(int t) {
/*------------------------------------------------------------------------------
	Kinetic propagation for t (=0 for DT/2--half; 1 for DT--full) step.
-------------------------------------------------------------------------------*/
	
	/* Apply the periodic boundary condition */
	periodic_bc();

	if (USE_GPU_PROP == 0) {
		int sx,s;
		double wr,wi;
		/* WRK|PSI holds the new|old wave function */
		for (sx=1; sx<=NX; sx++) {
			wr = al[t][0]*psi[sx][0]-al[t][1]*psi[sx][1];
			wi = al[t][0]*psi[sx][1]+al[t][1]*psi[sx][0];
			wr += (blx[t][sx][0]*psi[sx-1][0]-blx[t][sx][1]*psi[sx-1][1]);
			wi += (blx[t][sx][0]*psi[sx-1][1]+blx[t][sx][1]*psi[sx-1][0]);
			wr += (bux[t][sx][0]*psi[sx+1][0]-bux[t][sx][1]*psi[sx+1][1]);
			wi += (bux[t][sx][0]*psi[sx+1][1]+bux[t][sx][1]*psi[sx+1][0]);
			wrk[sx][0] = wr;
			wrk[sx][1] = wi;
		}
	} else {
		cudaMemcpy2D(dev_psi, 2*sizeof(double), psi, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);
		cudaMemcpy2D(dev_wrk, 2*sizeof(double), wrk, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyHostToDevice);
		
		gpu_kin_prop<<<1, NX>>>(dev_psi, dev_wrk, dev_al, dev_blx, dev_bux, t);

		cudaMemcpy2D(wrk, 2*sizeof(double), dev_wrk, 2*sizeof(double), 2*sizeof(double), NX+2, cudaMemcpyDeviceToHost);
	}



	/* Copy the new wave function back to PSI */
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<=1; s++)
			psi[sx][s] = wrk[sx][s];
}

/*----------------------------------------------------------------------------*/
void periodic_bc() {
/*------------------------------------------------------------------------------
	Applies the periodic boundary condition to wave function PSI, by copying
	the boundary values to the auxiliary array positions at the other ends.
------------------------------------------------------------------------------*/
	int plw,pup,s;
	double dbuf[2],dbufr[2];
	MPI_Status status;

	plw = (myid-1+nproc)%nproc; /* Lower partner process */
	pup = (myid+1      )%nproc; /* Upper partner process */

	/* Cache boundary wave function value at the lower end*/
	for (s=0; s<2; s++) /* Message composition */
		dbuf[s] = psi[NX][s];
	MPI_Send(dbuf, 2,MPI_DOUBLE,pup,10,MPI_COMM_WORLD);
	MPI_Recv(dbufr,2,MPI_DOUBLE,plw,10,MPI_COMM_WORLD,&status);
	for (s=0; s<2; s++) /* Message storing */
		psi[0][s] = dbufr[s];

	/* Cache boundary wave function value at the upper end*/
	for (s=0; s<2; s++) /* Message composition */
		dbuf[s] = psi[1][s];
	MPI_Send(dbuf, 2,MPI_DOUBLE,plw,20,MPI_COMM_WORLD);
	MPI_Recv(dbufr,2,MPI_DOUBLE,pup,20,MPI_COMM_WORLD,&status);
	for (s=0; s<2; s++) /* Message storing */
		psi[NX+1][s] = dbufr[s];
}

/*----------------------------------------------------------------------------*/
void calc_energy() {
/*------------------------------------------------------------------------------
	Calculates the kinetic, potential & total energies, EKIN, EPOT & ETOT.
------------------------------------------------------------------------------*/
	int sx,s;
	double a,bx;
	double lekin,lepot; /* Local energy values */

	/* Apply the periodic boundary condition */
	periodic_bc();

	/* Tridiagonal kinetic-energy operators */
	a =   1.0/(dx*dx);
	bx = -0.5/(dx*dx);

	/* |WRK> = (-1/2)Laplacian|PSI> */
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<=1; s++)
			wrk[sx][s] = a*psi[sx][s]+bx*(psi[sx-1][s]+psi[sx+1][s]);

	/* Kinetic energy = <PSI|(-1/2)Laplacian|PSI> = <PSI|WRK> */
	lekin = 0.0;
	for (sx=1; sx<=NX; sx++)
		lekin += (psi[sx][0]*wrk[sx][0]+psi[sx][1]*wrk[sx][1]);
	lekin *= dx;
	MPI_Allreduce(&lekin,&ekin,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	/* Potential energy */
	lepot = 0.0;
	for (sx=1; sx<=NX; sx++)
		lepot += v[sx]*(psi[sx][0]*psi[sx][0]+psi[sx][1]*psi[sx][1]);
	lepot *= dx;
	MPI_Allreduce(&lepot,&epot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	/* Total energy */
	etot = ekin+epot;
}
