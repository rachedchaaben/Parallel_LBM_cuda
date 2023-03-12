#ifndef LBM_FLOW_UTILS_KERNELS_H
#define LBM_FLOW_UTILS_KERNELS_H

// ================================================================
// ================================================================
__global__ 
void macroscopic_kernel(const LBMParams params,
                        const velocity_array_t v,
                        const real_t* fin_d,
                        real_t* rho_d,
                        real_t* ux_d,
                        real_t* uy_d)
{

	const int nx = params.nx;
	const int ny = params.ny;
	const int npop = LBMParams::npop;

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i<nx && j<ny) {

		int base_index = i + nx*j;

		real_t rho_tmp = 0;
		real_t ux_tmp = 0;
		real_t uy_tmp = 0;
		for (int ipop = 0; ipop < npop; ++ipop) {

			int index = base_index + ipop*nx*ny;

			// Oth order moment
			rho_tmp += fin_d[index];

			// 1st order moment
			ux_tmp += v(ipop, 0) * fin_d[index];
			uy_tmp += v(ipop, 1) * fin_d[index];

		} // end for ipop

		rho_d[base_index] = rho_tmp;
		ux_d[base_index] = ux_tmp / rho_tmp;
		uy_d[base_index] = uy_tmp / rho_tmp;

	}
} // macroscopic_kernel

// ================================================================
// ================================================================
__global__ void equilibrium_kernel(const LBMParams params,
                                   const velocity_array_t v,
                                   const weights_t t,
                                   const real_t *rho_d, 
                                   const real_t *ux_d,
                                   const real_t *uy_d, 
                                   real_t *feq_d)
{

	const int nx = params.nx;
	const int ny = params.ny;
	const int npop = LBMParams::npop;

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i<nx && j<ny) {

		int index = i + nx * j;

		real_t usqr = 3.0 / 2 * (ux_d[index] * ux_d[index] +
			uy_d[index] * uy_d[index]);

		for (int ipop = 0; ipop < npop; ++ipop) {
			real_t cu = 3 * (v(ipop, 0) * ux_d[index] +
				v(ipop, 1) * uy_d[index]);

			int index_f = index + ipop * nx * ny;
			feq_d[index_f] = rho_d[index] * t(ipop) * (1 + cu + 0.5*cu*cu - usqr);
		}

	} 
} // equilibrium_kernel

// ================================================================
// ================================================================
__global__ void border_outflow_kernel(const LBMParams params, 
                                      real_t *fin_d)
{

	const int nx = params.nx;
	const int ny = params.ny;

	const int nxny = nx*ny;

	const int i1 = nx - 1;
	const int i2 = nx - 2;

	int j = threadIdx.x + blockIdx.x*blockDim.x;

	if (j<ny) {

		int index1 = i1 + nx * j;
		int index2 = i2 + nx * j;

		fin_d[index1 + 6 * nxny] = fin_d[index2 + 6 * nxny];
		fin_d[index1 + 7 * nxny] = fin_d[index2 + 7 * nxny];
		fin_d[index1 + 8 * nxny] = fin_d[index2 + 8 * nxny];

	} // end for j
} // border_outflow_kernel

// ================================================================
// ================================================================
__global__ void border_inflow_kernel(const LBMParams params, 
                                     const real_t *fin_d,
                                     real_t *rho_d,
                                     real_t *ux_d,
                                     real_t *uy_d)
{

	const int nx = params.nx;
	const int ny = params.ny;

	const int nxny = nx*ny;

	const int i = 0;

	int j = threadIdx.x + blockIdx.x*blockDim.x;

	if (j<ny) {

		int index = i + nx * j;

		ux_d[index] = compute_vel(0, i, j, params.uLB, params.ly);
		uy_d[index] = compute_vel(1, i, j, params.uLB, params.ly);
		rho_d[index] = 1 / (1 - ux_d[index]) *
			(fin_d[index + 3 * nxny] + fin_d[index + 4 * nxny] + fin_d[index + 5 * nxny] +
				2 * (fin_d[index + 6 * nxny] + fin_d[index + 7 * nxny] + fin_d[index + 8 * nxny]));

	} 

} // border_inflow_kernel

// ================================================================
// ================================================================
__global__ void update_fin_inflow_kernel(const LBMParams params, 
                                         const real_t *feq_d,
                                         real_t *fin_d)
{

	const int nx = params.nx;
	const int ny = params.ny;

	const int nxny = nx*ny;

	const int i = 0;

	int j = threadIdx.x + blockIdx.x*blockDim.x;

	if (j<ny) {

		int index = i + nx * j;

		//fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6],0,:] - feq[[8,7,6],0,:]

		fin_d[index + 0 * nxny] = feq_d[index + 0 * nxny] + fin_d[index + 8 * nxny] - feq_d[index + 8 * nxny];
		fin_d[index + 1 * nxny] = feq_d[index + 1 * nxny] + fin_d[index + 7 * nxny] - feq_d[index + 7 * nxny];
		fin_d[index + 2 * nxny] = feq_d[index + 2 * nxny] + fin_d[index + 6 * nxny] - feq_d[index + 6 * nxny];

	} 
} // update_fin_inflow_kernel

// ================================================================
// ================================================================
__global__ void compute_collision_kernel(const LBMParams params,
                                         const real_t *fin_d, 
                                         const real_t *feq_d, 
                                         real_t *fout_d)
{
	const int nx = params.nx;
	const int ny = params.ny;

	const int nxny = nx*ny;

	const int npop = LBMParams::npop;
	const real_t omega = params.omega;

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i<nx && j<ny) {

		int index = i + nx * j;

		for (int ipop = 0; ipop<npop; ++ipop) {

			int index_f = index + ipop*nxny;

			fout_d[index_f] = fin_d[index_f] - omega * (fin_d[index_f] - feq_d[index_f]);

		} // end for ipop
	} 
} // compute_collision_kernel

// ================================================================
// ================================================================
__global__ void update_obstacle_kernel(const LBMParams params,
                                       const real_t *fin_d, 
                                       const int *obstacle_d, 
                                       real_t *fout_d)
{
	const int nx = params.nx;
	const int ny = params.ny;
	const int nxny = nx*ny;
	const int npop = LBMParams::npop;

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i<nx && j<ny) {

		int index = i + nx * j;

		if (obstacle_d[index] == 1) {

			for (int ipop = 0; ipop < npop; ++ipop) {

				int index_out = index + ipop  * nxny;
				int index_in = index + (8 - ipop) * nxny;

				fout_d[index_out] = fin_d[index_in];

			} // end for ipop

		} // end inside obstacle

	} 
} // update_obstacle_kernel

// ================================================================
// ================================================================
__global__ void streaming_kernel(const LBMParams params,
                                 const velocity_array_t v,
                                 const real_t *fout_d,
                                 real_t *fin_d)
{

	const int nx = params.nx;
	const int ny = params.ny;
	const int nxny = nx*ny;
	const int npop = LBMParams::npop;

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;

	if (i<nx && j<ny) {

		int index = i + nx * j;

		for (int ipop = 0; ipop < npop; ++ipop) {

			int index_in = index + ipop * nxny;

			int i_out = i - v(ipop, 0);
			if (i_out<0)
				i_out += nx;
			if (i_out>nx - 1)
				i_out -= nx;

			int j_out = j - v(ipop, 1);
			if (j_out<0)
				j_out += ny;
			if (j_out>ny - 1)
				j_out -= ny;

			int index_out = i_out + nx*j_out + ipop*nxny;

			fin_d[index_in] = fout_d[index_out];

		} // end for ipop

	} 
} // streaming_kernel

#endif // LBM_FLOW_UTILS_KERNELS_H
