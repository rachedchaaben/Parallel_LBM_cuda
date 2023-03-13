# Parallel_LBM_cuda
This is an academic project, part of HPC-AI Advanced Master's degree. 
The project consisits of parallelizing Lattice Boltzman Method with cuda. In this repo we have the original seqential C++ code and the parallelized cuda version. 
## Requirements

- cmake version >= 3.18 (when using nvcc)
- cmake version >= 3.22 (when using nvc++/nvhpc compiler)
- cuda toolkit


## Building :

### cpp :

```bash
mkdir -p build/
cd build/
cmake ..
make
# then you can run the application
./src/lbmFlowAroundCylinder
```
### cuda :
## Build with nvcc (Nvidia compiler from CUDA toolkit)


```bash
#  minimum cmake version required is 3.18
module load cmake/3.22.0

# if you want to build with nvcc
module load cuda/11.5
```

```bash
mkdir -p build/nvcc
cd build/nvcc
cmake -DCMAKE_CUDA_ARCHITECTURES="80" ../..
make
# then you can run the application
./src/lbmFlowAroundCylinder
```


## Build with nvc++ (Nvidia compiler from hpcsdk package)

```bash
#  minimum cmake version required is 3.22
module load cmake/3.22.0

# if you want to build with nvc++ (from Nvidia hpcsdk)
module load nvhpc/21.11
```

```bash
mkdir -p build/nvcc
cd build/nvcc
export CXX=nvc++
cmake -DCMAKE_CUDA_HOST_COMPILER=nvc++ -DCMAKE_CUDA_ARCHITECTURES="80" ../..
make
# then you can run the application
./src/lbmFlowAroundCylinder
```
