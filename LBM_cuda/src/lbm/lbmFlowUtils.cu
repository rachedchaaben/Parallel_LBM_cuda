#include <math.h> // for M_PI = 3.1415....

#include "lbmFlowUtils.h"

#include "lbmFlowUtils_kernels.h"
#include "cuda_error.h"

const int nbThreads = 32;

// ======================================================
// ======================================================
void macroscopic(const LBMParams& params, 
                 const velocity_array_t v,
                 const real_t* fin_d,
                 real_t* rho_d,
                 real_t* ux_d,
                 real_t* uy_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // call kernel
  dim3 grid_size((nx+nbThreads-1)/nbThreads,(ny+nbThreads-1)/nbThreads);
  dim3 block_dim(nbThreads, nbThreads);
  
  macroscopic_kernel<<<grid_size,block_dim>>>(params,v,fin_d,rho_d,ux_d,uy_d);

} // macroscopic

// ======================================================
// ======================================================
void equilibrium(const LBMParams& params, 
                 const velocity_array_t v,
                 const weights_t t,
                 const real_t* rho_d,
                 const real_t* ux_d,
                 const real_t* uy_d,
                 real_t* feq_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // call kernel 
  dim3 grid_size((nx+nbThreads-1)/nbThreads,(ny+nbThreads-1)/nbThreads);
  dim3 block_dim(nbThreads, nbThreads);
  
  equilibrium_kernel<<<grid_size,block_dim>>>(params,v,t,rho_d,ux_d,uy_d,feq_d);
  
} // equilibrium

// ======================================================
// ======================================================
void init_obstacle_mask(const LBMParams& params, 
                        int* obstacle, 
                        int* obstacle_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  const real_t cx = params.cx;
  const real_t cy = params.cy;

  const real_t r = params.r;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      real_t x = 1.0*i;
      real_t y = 1.0*j;

      obstacle[index] = (x-cx)*(x-cx) + (y-cy)*(y-cy) < r*r ? 1 : 0;

    } // end for i
  } // end for j

  // copy host to device
  cudaMemcpy(obstacle_d,obstacle,nx*ny*sizeof(int),cudaMemcpyHostToDevice);

} // init_obstacle_mask

// ======================================================
// ======================================================
__host__ __device__
real_t compute_vel(int dir, int i, int j, real_t uLB, real_t ly)
{

  // flow is along X axis
  // X component is non-zero
  // Y component is always zero

  return (1-dir) * uLB * (1 + 1e-4 * sin(j/ly*2*M_PI));

} // compute_vel

// ======================================================
// ======================================================
void initialize_macroscopic_variables(const LBMParams& params, 
                                      real_t* rho, real_t* rho_d,
                                      real_t* ux, real_t* ux_d,
                                      real_t* uy, real_t* uy_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      rho[index] = 1.0;
      ux[index]  = compute_vel(0, i, j, params.uLB, params.ly);
      uy[index]  = compute_vel(1, i, j, params.uLB, params.ly);

    } // end for i
  } // end for j

  // copy host to device
  cudaMemcpy(rho_d,rho,nx*ny*sizeof(real_t),cudaMemcpyHostToDevice);
  cudaMemcpy(ux_d,ux,nx*ny*sizeof(real_t),cudaMemcpyHostToDevice);
  cudaMemcpy(uy_d,uy,nx*ny*sizeof(real_t),cudaMemcpyHostToDevice);

} // initialize_macroscopic_variables

// ======================================================
// ======================================================
void border_outflow(const LBMParams& params, real_t* fin_d)
{
  const int ny = params.ny;
  
  // call kernel
  dim3 grid_size((ny+nbThreads-1)/nbThreads);
  dim3 block_dim(nbThreads);
  border_outflow_kernel<<<grid_size,block_dim>>>(params,fin_d);
  

} // border_outflow

// ======================================================
// ======================================================
void border_inflow(const LBMParams& params, const real_t* fin_d, 
                   real_t* rho_d, real_t* ux_d, real_t* uy_d)
{
  const int ny = params.ny;
  
  // call kernel
  dim3 grid_size((ny+nbThreads-1)/nbThreads);
  dim3 block_dim(nbThreads);
  
  border_inflow_kernel<<<grid_size,block_dim>>>(params,fin_d,rho_d,ux_d,uy_d);
  
} // border_inflow

// ======================================================
// ======================================================
void update_fin_inflow(const LBMParams& params, const real_t* feq_d, 
                       real_t* fin_d)
{

  const int ny = params.ny;
  
  // call kernel
  dim3 grid_size((ny+nbThreads-1)/nbThreads);
  dim3 block_dim(nbThreads);
  
  update_fin_inflow_kernel<<<grid_size,block_dim>>>(params,feq_d,fin_d);
  
} // update_fin_inflow
  
// ======================================================
// ======================================================
void compute_collision(const LBMParams& params, 
                       const real_t* fin_d,
                       const real_t* feq_d,
                       real_t* fout_d)
{

  const int nx = params.nx;
  const int ny = params.ny;
  
  // call kernel
  dim3 grid_size((nx+nbThreads-1)/nbThreads,(ny+nbThreads-1)/nbThreads);
  dim3 block_dim(nbThreads,nbThreads);
  
  compute_collision_kernel<<<grid_size,block_dim>>>(params,fin_d,feq_d,fout_d);
  
} // compute_collision

// ======================================================
// ======================================================
void update_obstacle(const LBMParams &params, 
                     const real_t* fin_d,
                     const int* obstacle_d, 
                     real_t* fout_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // call kernel
  dim3 grid_size((nx+nbThreads-1)/nbThreads,(ny+nbThreads-1)/nbThreads);
  dim3 block_dim(nbThreads,nbThreads);
  
  update_obstacle_kernel<<<grid_size,block_dim>>>(params,fin_d,obstacle_d,fout_d);
} // update_obstacle

// ======================================================
// ======================================================
void streaming(const LBMParams& params,
               const velocity_array_t v,
               const real_t* fout_d,
               real_t* fin_d)
{

  const int nx = params.nx;
  const int ny = params.ny;

  // call kernel  
  dim3 grid_size((nx+nbThreads-1)/nbThreads,(ny+nbThreads-1)/nbThreads);
  dim3 block_dim(nbThreads,nbThreads);
  
  streaming_kernel<<<grid_size,block_dim>>>(params,v,fout_d,fin_d);


} // streaming
