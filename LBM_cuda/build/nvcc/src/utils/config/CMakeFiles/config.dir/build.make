# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc

# Include any dependencies generated for this target.
include src/utils/config/CMakeFiles/config.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/utils/config/CMakeFiles/config.dir/compiler_depend.make

# Include the progress variables for this target.
include src/utils/config/CMakeFiles/config.dir/progress.make

# Include the compile flags for this target's objects.
include src/utils/config/CMakeFiles/config.dir/flags.make

src/utils/config/CMakeFiles/config.dir/ConfigMap.cpp.o: src/utils/config/CMakeFiles/config.dir/flags.make
src/utils/config/CMakeFiles/config.dir/ConfigMap.cpp.o: ../../src/utils/config/ConfigMap.cpp
src/utils/config/CMakeFiles/config.dir/ConfigMap.cpp.o: src/utils/config/CMakeFiles/config.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/utils/config/CMakeFiles/config.dir/ConfigMap.cpp.o"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/utils/config/CMakeFiles/config.dir/ConfigMap.cpp.o -MF CMakeFiles/config.dir/ConfigMap.cpp.o.d -o CMakeFiles/config.dir/ConfigMap.cpp.o -c /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/ConfigMap.cpp

src/utils/config/CMakeFiles/config.dir/ConfigMap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/config.dir/ConfigMap.cpp.i"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/ConfigMap.cpp > CMakeFiles/config.dir/ConfigMap.cpp.i

src/utils/config/CMakeFiles/config.dir/ConfigMap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/config.dir/ConfigMap.cpp.s"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/ConfigMap.cpp -o CMakeFiles/config.dir/ConfigMap.cpp.s

src/utils/config/CMakeFiles/config.dir/inih/ini.cpp.o: src/utils/config/CMakeFiles/config.dir/flags.make
src/utils/config/CMakeFiles/config.dir/inih/ini.cpp.o: ../../src/utils/config/inih/ini.cpp
src/utils/config/CMakeFiles/config.dir/inih/ini.cpp.o: src/utils/config/CMakeFiles/config.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/utils/config/CMakeFiles/config.dir/inih/ini.cpp.o"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/utils/config/CMakeFiles/config.dir/inih/ini.cpp.o -MF CMakeFiles/config.dir/inih/ini.cpp.o.d -o CMakeFiles/config.dir/inih/ini.cpp.o -c /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/inih/ini.cpp

src/utils/config/CMakeFiles/config.dir/inih/ini.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/config.dir/inih/ini.cpp.i"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/inih/ini.cpp > CMakeFiles/config.dir/inih/ini.cpp.i

src/utils/config/CMakeFiles/config.dir/inih/ini.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/config.dir/inih/ini.cpp.s"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/inih/ini.cpp -o CMakeFiles/config.dir/inih/ini.cpp.s

src/utils/config/CMakeFiles/config.dir/inih/INIReader.cpp.o: src/utils/config/CMakeFiles/config.dir/flags.make
src/utils/config/CMakeFiles/config.dir/inih/INIReader.cpp.o: ../../src/utils/config/inih/INIReader.cpp
src/utils/config/CMakeFiles/config.dir/inih/INIReader.cpp.o: src/utils/config/CMakeFiles/config.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/utils/config/CMakeFiles/config.dir/inih/INIReader.cpp.o"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/utils/config/CMakeFiles/config.dir/inih/INIReader.cpp.o -MF CMakeFiles/config.dir/inih/INIReader.cpp.o.d -o CMakeFiles/config.dir/inih/INIReader.cpp.o -c /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/inih/INIReader.cpp

src/utils/config/CMakeFiles/config.dir/inih/INIReader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/config.dir/inih/INIReader.cpp.i"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/inih/INIReader.cpp > CMakeFiles/config.dir/inih/INIReader.cpp.i

src/utils/config/CMakeFiles/config.dir/inih/INIReader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/config.dir/inih/INIReader.cpp.s"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config/inih/INIReader.cpp -o CMakeFiles/config.dir/inih/INIReader.cpp.s

# Object files for target config
config_OBJECTS = \
"CMakeFiles/config.dir/ConfigMap.cpp.o" \
"CMakeFiles/config.dir/inih/ini.cpp.o" \
"CMakeFiles/config.dir/inih/INIReader.cpp.o"

# External object files for target config
config_EXTERNAL_OBJECTS =

src/utils/config/libconfig.a: src/utils/config/CMakeFiles/config.dir/ConfigMap.cpp.o
src/utils/config/libconfig.a: src/utils/config/CMakeFiles/config.dir/inih/ini.cpp.o
src/utils/config/libconfig.a: src/utils/config/CMakeFiles/config.dir/inih/INIReader.cpp.o
src/utils/config/libconfig.a: src/utils/config/CMakeFiles/config.dir/build.make
src/utils/config/libconfig.a: src/utils/config/CMakeFiles/config.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libconfig.a"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && $(CMAKE_COMMAND) -P CMakeFiles/config.dir/cmake_clean_target.cmake
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/config.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/utils/config/CMakeFiles/config.dir/build: src/utils/config/libconfig.a
.PHONY : src/utils/config/CMakeFiles/config.dir/build

src/utils/config/CMakeFiles/config.dir/clean:
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config && $(CMAKE_COMMAND) -P CMakeFiles/config.dir/cmake_clean.cmake
.PHONY : src/utils/config/CMakeFiles/config.dir/clean

src/utils/config/CMakeFiles/config.dir/depend:
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/src/utils/config /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cuda/build/nvcc/src/utils/config/CMakeFiles/config.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/utils/config/CMakeFiles/config.dir/depend

