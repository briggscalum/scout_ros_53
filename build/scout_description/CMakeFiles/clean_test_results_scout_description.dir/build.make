# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/Docker/scout_ros_53/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/Docker/scout_ros_53/build

# Utility rule file for clean_test_results_scout_description.

# Include the progress variables for this target.
include scout_description/CMakeFiles/clean_test_results_scout_description.dir/progress.make

scout_description/CMakeFiles/clean_test_results_scout_description:
	cd /root/Docker/scout_ros_53/build/scout_description && /usr/bin/python2 /opt/ros/melodic/share/catkin/cmake/test/remove_test_results.py /root/Docker/scout_ros_53/build/test_results/scout_description

clean_test_results_scout_description: scout_description/CMakeFiles/clean_test_results_scout_description
clean_test_results_scout_description: scout_description/CMakeFiles/clean_test_results_scout_description.dir/build.make

.PHONY : clean_test_results_scout_description

# Rule to build all files generated by this target.
scout_description/CMakeFiles/clean_test_results_scout_description.dir/build: clean_test_results_scout_description

.PHONY : scout_description/CMakeFiles/clean_test_results_scout_description.dir/build

scout_description/CMakeFiles/clean_test_results_scout_description.dir/clean:
	cd /root/Docker/scout_ros_53/build/scout_description && $(CMAKE_COMMAND) -P CMakeFiles/clean_test_results_scout_description.dir/cmake_clean.cmake
.PHONY : scout_description/CMakeFiles/clean_test_results_scout_description.dir/clean

scout_description/CMakeFiles/clean_test_results_scout_description.dir/depend:
	cd /root/Docker/scout_ros_53/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/Docker/scout_ros_53/src /root/Docker/scout_ros_53/src/scout_description /root/Docker/scout_ros_53/build /root/Docker/scout_ros_53/build/scout_description /root/Docker/scout_ros_53/build/scout_description/CMakeFiles/clean_test_results_scout_description.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : scout_description/CMakeFiles/clean_test_results_scout_description.dir/depend

