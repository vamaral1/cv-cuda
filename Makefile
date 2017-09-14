# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/victor/Dropbox/parallel_programing/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/victor/Dropbox/parallel_programing/project

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running interactive CMake command-line interface..."
	/usr/bin/cmake -i .
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/victor/Dropbox/parallel_programing/project/CMakeFiles /home/victor/Dropbox/parallel_programing/project/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/victor/Dropbox/parallel_programing/project/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named TAGS

# Build rule for target.
TAGS: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 TAGS
.PHONY : TAGS

# fast build rule for target.
TAGS/fast:
	$(MAKE) -f CMakeFiles/TAGS.dir/build.make CMakeFiles/TAGS.dir/build
.PHONY : TAGS/fast

#=============================================================================
# Target rules for targets named lk_parallel

# Build rule for target.
lk_parallel: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 lk_parallel
.PHONY : lk_parallel

# fast build rule for target.
lk_parallel/fast:
	$(MAKE) -f CMakeFiles/lk_parallel.dir/build.make CMakeFiles/lk_parallel.dir/build
.PHONY : lk_parallel/fast

#=============================================================================
# Target rules for targets named lk_serial

# Build rule for target.
lk_serial: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 lk_serial
.PHONY : lk_serial

# fast build rule for target.
lk_serial/fast:
	$(MAKE) -f CMakeFiles/lk_serial.dir/build.make CMakeFiles/lk_serial.dir/build
.PHONY : lk_serial/fast

#=============================================================================
# Target rules for targets named sub_parallel

# Build rule for target.
sub_parallel: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 sub_parallel
.PHONY : sub_parallel

# fast build rule for target.
sub_parallel/fast:
	$(MAKE) -f CMakeFiles/sub_parallel.dir/build.make CMakeFiles/sub_parallel.dir/build
.PHONY : sub_parallel/fast

#=============================================================================
# Target rules for targets named sub_serial

# Build rule for target.
sub_serial: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 sub_serial
.PHONY : sub_serial

# fast build rule for target.
sub_serial/fast:
	$(MAKE) -f CMakeFiles/sub_serial.dir/build.make CMakeFiles/sub_serial.dir/build
.PHONY : sub_serial/fast

# target to build an object file
src/lk_parallel.o:
	$(MAKE) -f CMakeFiles/lk_parallel.dir/build.make CMakeFiles/lk_parallel.dir/src/lk_parallel.o
.PHONY : src/lk_parallel.o

# target to preprocess a source file
src/lk_parallel.i:
	$(MAKE) -f CMakeFiles/lk_parallel.dir/build.make CMakeFiles/lk_parallel.dir/src/lk_parallel.i
.PHONY : src/lk_parallel.i

# target to generate assembly for a file
src/lk_parallel.s:
	$(MAKE) -f CMakeFiles/lk_parallel.dir/build.make CMakeFiles/lk_parallel.dir/src/lk_parallel.s
.PHONY : src/lk_parallel.s

# target to build an object file
src/lk_serial.o:
	$(MAKE) -f CMakeFiles/lk_serial.dir/build.make CMakeFiles/lk_serial.dir/src/lk_serial.o
.PHONY : src/lk_serial.o

# target to preprocess a source file
src/lk_serial.i:
	$(MAKE) -f CMakeFiles/lk_serial.dir/build.make CMakeFiles/lk_serial.dir/src/lk_serial.i
.PHONY : src/lk_serial.i

# target to generate assembly for a file
src/lk_serial.s:
	$(MAKE) -f CMakeFiles/lk_serial.dir/build.make CMakeFiles/lk_serial.dir/src/lk_serial.s
.PHONY : src/lk_serial.s

# target to build an object file
src/sub_parallel.o:
	$(MAKE) -f CMakeFiles/sub_parallel.dir/build.make CMakeFiles/sub_parallel.dir/src/sub_parallel.o
.PHONY : src/sub_parallel.o

# target to preprocess a source file
src/sub_parallel.i:
	$(MAKE) -f CMakeFiles/sub_parallel.dir/build.make CMakeFiles/sub_parallel.dir/src/sub_parallel.i
.PHONY : src/sub_parallel.i

# target to generate assembly for a file
src/sub_parallel.s:
	$(MAKE) -f CMakeFiles/sub_parallel.dir/build.make CMakeFiles/sub_parallel.dir/src/sub_parallel.s
.PHONY : src/sub_parallel.s

# target to build an object file
src/sub_serial.o:
	$(MAKE) -f CMakeFiles/sub_serial.dir/build.make CMakeFiles/sub_serial.dir/src/sub_serial.o
.PHONY : src/sub_serial.o

# target to preprocess a source file
src/sub_serial.i:
	$(MAKE) -f CMakeFiles/sub_serial.dir/build.make CMakeFiles/sub_serial.dir/src/sub_serial.i
.PHONY : src/sub_serial.i

# target to generate assembly for a file
src/sub_serial.s:
	$(MAKE) -f CMakeFiles/sub_serial.dir/build.make CMakeFiles/sub_serial.dir/src/sub_serial.s
.PHONY : src/sub_serial.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... TAGS"
	@echo "... edit_cache"
	@echo "... lk_parallel"
	@echo "... lk_serial"
	@echo "... rebuild_cache"
	@echo "... sub_parallel"
	@echo "... sub_serial"
	@echo "... src/lk_parallel.o"
	@echo "... src/lk_parallel.i"
	@echo "... src/lk_parallel.s"
	@echo "... src/lk_serial.o"
	@echo "... src/lk_serial.i"
	@echo "... src/lk_serial.s"
	@echo "... src/sub_parallel.o"
	@echo "... src/sub_parallel.i"
	@echo "... src/sub_parallel.s"
	@echo "... src/sub_serial.o"
	@echo "... src/sub_serial.i"
	@echo "... src/sub_serial.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

