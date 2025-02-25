# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/quan/workspace/cuda-learn/03-matmul-basic

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/quan/workspace/cuda-learn/03-matmul-basic/build

# Include any dependencies generated for this target.
include CMakeFiles/trt_cuda.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/trt_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/trt_cuda.dir/flags.make

CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.o: CMakeFiles/trt_cuda.dir/flags.make
CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.o: ../src/matmul_gpu_basic.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/quan/workspace/cuda-learn/03-matmul-basic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.o"
	/usr/local/cuda-11.5/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/quan/workspace/cuda-learn/03-matmul-basic/src/matmul_gpu_basic.cu -o CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.o

CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/trt_cuda.dir/src/main.cpp.o: CMakeFiles/trt_cuda.dir/flags.make
CMakeFiles/trt_cuda.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/quan/workspace/cuda-learn/03-matmul-basic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/trt_cuda.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/trt_cuda.dir/src/main.cpp.o -c /home/quan/workspace/cuda-learn/03-matmul-basic/src/main.cpp

CMakeFiles/trt_cuda.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/trt_cuda.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/quan/workspace/cuda-learn/03-matmul-basic/src/main.cpp > CMakeFiles/trt_cuda.dir/src/main.cpp.i

CMakeFiles/trt_cuda.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/trt_cuda.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/quan/workspace/cuda-learn/03-matmul-basic/src/main.cpp -o CMakeFiles/trt_cuda.dir/src/main.cpp.s

CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.o: CMakeFiles/trt_cuda.dir/flags.make
CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.o: ../src/matmul_cpu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/quan/workspace/cuda-learn/03-matmul-basic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.o -c /home/quan/workspace/cuda-learn/03-matmul-basic/src/matmul_cpu.cpp

CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/quan/workspace/cuda-learn/03-matmul-basic/src/matmul_cpu.cpp > CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.i

CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/quan/workspace/cuda-learn/03-matmul-basic/src/matmul_cpu.cpp -o CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.s

CMakeFiles/trt_cuda.dir/src/utils.cpp.o: CMakeFiles/trt_cuda.dir/flags.make
CMakeFiles/trt_cuda.dir/src/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/quan/workspace/cuda-learn/03-matmul-basic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/trt_cuda.dir/src/utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/trt_cuda.dir/src/utils.cpp.o -c /home/quan/workspace/cuda-learn/03-matmul-basic/src/utils.cpp

CMakeFiles/trt_cuda.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/trt_cuda.dir/src/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/quan/workspace/cuda-learn/03-matmul-basic/src/utils.cpp > CMakeFiles/trt_cuda.dir/src/utils.cpp.i

CMakeFiles/trt_cuda.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/trt_cuda.dir/src/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/quan/workspace/cuda-learn/03-matmul-basic/src/utils.cpp -o CMakeFiles/trt_cuda.dir/src/utils.cpp.s

# Object files for target trt_cuda
trt_cuda_OBJECTS = \
"CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.o" \
"CMakeFiles/trt_cuda.dir/src/main.cpp.o" \
"CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.o" \
"CMakeFiles/trt_cuda.dir/src/utils.cpp.o"

# External object files for target trt_cuda
trt_cuda_EXTERNAL_OBJECTS =

CMakeFiles/trt_cuda.dir/cmake_device_link.o: CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.o
CMakeFiles/trt_cuda.dir/cmake_device_link.o: CMakeFiles/trt_cuda.dir/src/main.cpp.o
CMakeFiles/trt_cuda.dir/cmake_device_link.o: CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.o
CMakeFiles/trt_cuda.dir/cmake_device_link.o: CMakeFiles/trt_cuda.dir/src/utils.cpp.o
CMakeFiles/trt_cuda.dir/cmake_device_link.o: CMakeFiles/trt_cuda.dir/build.make
CMakeFiles/trt_cuda.dir/cmake_device_link.o: CMakeFiles/trt_cuda.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/quan/workspace/cuda-learn/03-matmul-basic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CUDA device code CMakeFiles/trt_cuda.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/trt_cuda.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/trt_cuda.dir/build: CMakeFiles/trt_cuda.dir/cmake_device_link.o

.PHONY : CMakeFiles/trt_cuda.dir/build

# Object files for target trt_cuda
trt_cuda_OBJECTS = \
"CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.o" \
"CMakeFiles/trt_cuda.dir/src/main.cpp.o" \
"CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.o" \
"CMakeFiles/trt_cuda.dir/src/utils.cpp.o"

# External object files for target trt_cuda
trt_cuda_EXTERNAL_OBJECTS =

trt_cuda: CMakeFiles/trt_cuda.dir/src/matmul_gpu_basic.cu.o
trt_cuda: CMakeFiles/trt_cuda.dir/src/main.cpp.o
trt_cuda: CMakeFiles/trt_cuda.dir/src/matmul_cpu.cpp.o
trt_cuda: CMakeFiles/trt_cuda.dir/src/utils.cpp.o
trt_cuda: CMakeFiles/trt_cuda.dir/build.make
trt_cuda: CMakeFiles/trt_cuda.dir/cmake_device_link.o
trt_cuda: CMakeFiles/trt_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/quan/workspace/cuda-learn/03-matmul-basic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable trt_cuda"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/trt_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/trt_cuda.dir/build: trt_cuda

.PHONY : CMakeFiles/trt_cuda.dir/build

CMakeFiles/trt_cuda.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/trt_cuda.dir/cmake_clean.cmake
.PHONY : CMakeFiles/trt_cuda.dir/clean

CMakeFiles/trt_cuda.dir/depend:
	cd /home/quan/workspace/cuda-learn/03-matmul-basic/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/quan/workspace/cuda-learn/03-matmul-basic /home/quan/workspace/cuda-learn/03-matmul-basic /home/quan/workspace/cuda-learn/03-matmul-basic/build /home/quan/workspace/cuda-learn/03-matmul-basic/build /home/quan/workspace/cuda-learn/03-matmul-basic/build/CMakeFiles/trt_cuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/trt_cuda.dir/depend

