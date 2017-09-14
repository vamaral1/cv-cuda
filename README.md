## Parallel Programming 600.420 Final Project
Victor Amaral, Timothy Ng

As a final project, we had to implement something using one of the parallel processing frameworks we learned about in class including OpenMP, MPI, Hadoop, Spark, and CUDA. Since we were both interested in computer vision, we chose to explore the CUDA library for motion tracking and compare GPU performance with CPU performance. For more info, please refer to writeup.pdf

Install OpenCV 2.4.10 and Cuda 5.5
* run `cmake .` in the main project directory
* run `make`
* run `./run.sh` to run all files or look inside for an example to run one file

Directories:
* src - contains all source files
* bin - contains all executables
* doc - contains all necessary files to compile the writeup with latex including writeup.pdf which is the paper for the project
* video - contains the video files used for testing
* output - output from all runs and from profiler
