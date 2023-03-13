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
CMAKE_SOURCE_DIR = /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build

# Include any dependencies generated for this target.
include src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/compiler_depend.make

# Include the progress variables for this target.
include src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/progress.make

# Include the compile flags for this target's objects.
include src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/flags.make

src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o: src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/flags.make
src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o: ../src/utils/monitoring/OpenMPTimer.cpp
src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o: src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o -MF CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o.d -o CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o -c /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/src/utils/monitoring/OpenMPTimer.cpp

src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.i"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/src/utils/monitoring/OpenMPTimer.cpp > CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.i

src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.s"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/src/utils/monitoring/OpenMPTimer.cpp -o CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.s

src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o: src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/flags.make
src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o: ../src/utils/monitoring/SimpleTimer.cpp
src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o: src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o -MF CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o.d -o CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o -c /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/src/utils/monitoring/SimpleTimer.cpp

src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.i"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/src/utils/monitoring/SimpleTimer.cpp > CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.i

src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.s"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/src/utils/monitoring/SimpleTimer.cpp -o CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.s

# Object files for target lbmMonitoring
lbmMonitoring_OBJECTS = \
"CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o" \
"CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o"

# External object files for target lbmMonitoring
lbmMonitoring_EXTERNAL_OBJECTS =

src/utils/monitoring/liblbmMonitoring.a: src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/OpenMPTimer.cpp.o
src/utils/monitoring/liblbmMonitoring.a: src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/SimpleTimer.cpp.o
src/utils/monitoring/liblbmMonitoring.a: src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/build.make
src/utils/monitoring/liblbmMonitoring.a: src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library liblbmMonitoring.a"
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && $(CMAKE_COMMAND) -P CMakeFiles/lbmMonitoring.dir/cmake_clean_target.cmake
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lbmMonitoring.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/build: src/utils/monitoring/liblbmMonitoring.a
.PHONY : src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/build

src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/clean:
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring && $(CMAKE_COMMAND) -P CMakeFiles/lbmMonitoring.dir/cmake_clean.cmake
.PHONY : src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/clean

src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/depend:
	cd /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/src/utils/monitoring /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring /home/rchaaben/MS-HPC-AI-GPU/code/mini-projet/LBM/cpp/LBM_cpp/build/src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/utils/monitoring/CMakeFiles/lbmMonitoring.dir/depend

