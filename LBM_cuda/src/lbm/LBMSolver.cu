#include <cstdlib> // for malloc
#include <iostream>
#include <vector>
#include <sstream>

#include "LBMSolver.h"

#include "lbmFlowUtils.h"

#include "writePNG/lodepng.h"
#include "writeVTK/saveVTK.h"

#include "cuda_error.h"

#include <sys/time.h>
struct timeval t1, t2;

// ======================================================
// ======================================================
LBMSolver::LBMSolver(const LBMParams& params) :
  params(params)
{

  const int nx = params.nx;
  const int ny = params.ny;
  const int npop = LBMParams::npop;

  // memory allocations

  // distribution functions
  fin  = (real_t*) malloc(nx*ny*npop * sizeof(real_t));
  fout = (real_t*) malloc(nx*ny*npop * sizeof(real_t));
  feq  = (real_t*) malloc(nx*ny*npop * sizeof(real_t));

  cudaMalloc((real_t**)&fin_d, nx*ny*npop * sizeof(real_t));
  cudaMalloc((real_t**)&fout_d, nx*ny*npop * sizeof(real_t));
  cudaMalloc((real_t**)&feq_d, nx*ny*npop * sizeof(real_t));

  // macroscopic variables
  rho = (real_t*) malloc(nx*ny * sizeof(real_t));
  ux  = (real_t*) malloc(nx*ny * sizeof(real_t));
  uy  = (real_t*) malloc(nx*ny * sizeof(real_t));
  
  cudaMalloc((real_t**)&rho_d, nx*ny*npop * sizeof(real_t));
  cudaMalloc((real_t**)&ux_d, nx*ny*npop * sizeof(real_t));
  cudaMalloc((real_t**)&uy_d, nx*ny*npop * sizeof(real_t));

  // obstacle
  obstacle = (int *) malloc(nx*ny * sizeof(int));
  
  cudaMalloc((real_t**)&obstacle_d, nx*ny* sizeof(int));

} // LBMSolver::LBMSolver

// ======================================================
// ======================================================
LBMSolver::~LBMSolver()
{
  // free memory

  // distribution functions
  delete[] fin;
  delete[] fout;
  delete[] feq;
  
  cudaFree(fin_d);
  cudaFree(fout_d);
  cudaFree(feq_d);
  
  // macroscopic variables
  delete[] rho;
  delete[] ux;
  delete[] uy;
  
  cudaFree(rho_d);
  cudaFree(ux_d);
  cudaFree(uy_d);

  // obstacle
  delete[] obstacle;
  
  cudaFree(obstacle_d);
  
} // LBMSolver::~LBMSolver

// ======================================================
// ======================================================
void LBMSolver::initialize()
{

  // initialize obstacle mask array
  init_obstacle_mask(params, obstacle, obstacle_d);

  // initialize macroscopic velocity
  initialize_macroscopic_variables(params, 
                                   rho, rho_d, 
                                   ux, ux_d, 
                                   uy, uy_d);

  // Initialization of the populations at equilibrium 
  // with the given macroscopic variables.
  equilibrium(params, v, t, rho_d, ux_d, uy_d, fin_d);
  
} // LBMSolver::initialize

// ======================================================
// ======================================================
void LBMSolver::run()
{
  //Uncomment for time measurment 
  //gettimeofday(&t1, 0);

  initialize();
  
  // time loop
  for (int iTime=0; iTime<params.maxIter; ++iTime) {

    if (iTime % 100 == 0) {
      output_png(iTime);
      output_vtk(iTime);
    }

    // Right wall: outflow condition.
    // we only need here to specify distrib. function for velocities
    // that enter the domain (other that go out, are set by the streaming step)
    border_outflow(params, fin_d);
      
    // Compute macroscopic variables, density and velocity.
    macroscopic(params, v, fin_d, rho_d, ux_d, uy_d);
      
    // Left wall: inflow condition.
    border_inflow(params, fin_d, rho_d, ux_d, uy_d);

    // Compute equilibrium.
    equilibrium(params, v, t, rho_d, ux_d, uy_d, feq_d);
    update_fin_inflow(params, feq_d, fin_d);

    // Collision step.
    compute_collision(params, fin_d, feq_d, fout_d);

    // Bounce-back condition for obstacle.
    // in python language, we "slice" fout by obstacle
    update_obstacle(params, fin_d, obstacle_d, fout_d);

    // Streaming step.
    streaming(params, v, fout_d, fin_d);

  } // end for iTime
  // Uncomment for time measurment
  /*
  gettimeofday(&t2, 0);
  double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
  printf("Time to generate:  %3.1f ms \n", time);
  */
} // LBMSolver::run

// ======================================================
// ======================================================
void LBMSolver::output_png(int iTime)
{

  std::cout << "Output data (PNG) at time " << iTime << "\n";

  const int nx = params.nx;
  const int ny = params.ny;

  // copy data device to host
  cudaMemcpy(ux, ux_d, nx*ny*sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(uy, uy_d, nx*ny*sizeof(real_t), cudaMemcpyDeviceToHost);

  real_t* u2 = (real_t *) malloc(nx*ny*sizeof(real_t));

  // compute velocity norm, as well as min and max values
  real_t min_value = sqrt(ux[0]*ux[0] + uy[0]*uy[0]);
  real_t max_value = min_value;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      u2[index] = sqrt(ux[index]*ux[index] + uy[index]*uy[index]);

      if (u2[index]<min_value)
        min_value = u2[index];

      if (u2[index]>max_value)
        max_value = u2[index];

    } // end for i

  } // end for j

  // create png image buff
  std::vector<unsigned char> image;
  image.resize(nx * ny * 4);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {

      int index = i + nx * j;

      // rescale velocity in 0-255 range
      unsigned char value = static_cast<unsigned char>((u2[index]-min_value)/(max_value-min_value)*255);
      image[0 + 4*i + 4*nx*j] = value; 
      image[1 + 4*i + 4*nx*j] = value; 
      image[2 + 4*i + 4*nx*j] = value; 
      image[3 + 4*i + 4*nx*j] = value; 
    }
  }

  std::ostringstream iTimeNum;
  iTimeNum.width(7);
  iTimeNum.fill('0');
  iTimeNum << iTime;

  std::string filename  = "vel_" + iTimeNum.str() + ".png";

  // encode the image
  unsigned error = lodepng::encode(filename, image, nx, ny);

  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;


  delete[] u2;

} // LBMSolver::output_png

// ======================================================
// ======================================================
void LBMSolver::output_vtk(int iTime)
{

  std::cout << "Output data (VTK) at time " << iTime << "\n";

  bool useAscii = false; // binary data leads to smaller files
  
  const int nx = params.nx;
  const int ny = params.ny;
  
  // copy data device to host  
  cudaMemcpy(ux, ux_d, nx*ny*sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(ux, ux_d, nx*ny*sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(rho, rho_d, nx*ny*sizeof(real_t), cudaMemcpyDeviceToHost);

  saveVTK(rho, ux, uy, params, useAscii, iTime);

} // LBMSolver::output_vtk
