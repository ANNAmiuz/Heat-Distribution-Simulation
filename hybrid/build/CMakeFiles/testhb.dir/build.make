# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

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
CMAKE_COMMAND = /opt/cmake/bin/cmake

# The command to remove a file.
RM = /opt/cmake/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/119010114/.mmap/hybrid

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/119010114/.mmap/hybrid/build

# Include any dependencies generated for this target.
include CMakeFiles/testhb.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/testhb.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/testhb.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testhb.dir/flags.make

CMakeFiles/testhb.dir/src/graphic.cpp.o: CMakeFiles/testhb.dir/flags.make
CMakeFiles/testhb.dir/src/graphic.cpp.o: ../src/graphic.cpp
CMakeFiles/testhb.dir/src/graphic.cpp.o: CMakeFiles/testhb.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/119010114/.mmap/hybrid/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testhb.dir/src/graphic.cpp.o"
	/usr/local/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testhb.dir/src/graphic.cpp.o -MF CMakeFiles/testhb.dir/src/graphic.cpp.o.d -o CMakeFiles/testhb.dir/src/graphic.cpp.o -c /home/119010114/.mmap/hybrid/src/graphic.cpp

CMakeFiles/testhb.dir/src/graphic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testhb.dir/src/graphic.cpp.i"
	/usr/local/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/119010114/.mmap/hybrid/src/graphic.cpp > CMakeFiles/testhb.dir/src/graphic.cpp.i

CMakeFiles/testhb.dir/src/graphic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testhb.dir/src/graphic.cpp.s"
	/usr/local/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/119010114/.mmap/hybrid/src/graphic.cpp -o CMakeFiles/testhb.dir/src/graphic.cpp.s

CMakeFiles/testhb.dir/src/main.cpp.o: CMakeFiles/testhb.dir/flags.make
CMakeFiles/testhb.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/testhb.dir/src/main.cpp.o: CMakeFiles/testhb.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/119010114/.mmap/hybrid/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/testhb.dir/src/main.cpp.o"
	/usr/local/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testhb.dir/src/main.cpp.o -MF CMakeFiles/testhb.dir/src/main.cpp.o.d -o CMakeFiles/testhb.dir/src/main.cpp.o -c /home/119010114/.mmap/hybrid/src/main.cpp

CMakeFiles/testhb.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testhb.dir/src/main.cpp.i"
	/usr/local/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/119010114/.mmap/hybrid/src/main.cpp > CMakeFiles/testhb.dir/src/main.cpp.i

CMakeFiles/testhb.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testhb.dir/src/main.cpp.s"
	/usr/local/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/119010114/.mmap/hybrid/src/main.cpp -o CMakeFiles/testhb.dir/src/main.cpp.s

# Object files for target testhb
testhb_OBJECTS = \
"CMakeFiles/testhb.dir/src/graphic.cpp.o" \
"CMakeFiles/testhb.dir/src/main.cpp.o"

# External object files for target testhb
testhb_EXTERNAL_OBJECTS =

testhb: CMakeFiles/testhb.dir/src/graphic.cpp.o
testhb: CMakeFiles/testhb.dir/src/main.cpp.o
testhb: CMakeFiles/testhb.dir/build.make
testhb: libcore.a
testhb: /usr/lib64/libfreetype.so
testhb: /usr/lib64/libSDL2.so
testhb: /usr/lib64/libGLX.so
testhb: /usr/lib64/libOpenGL.so
testhb: /usr/local/lib/libmpi.so
testhb: CMakeFiles/testhb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/119010114/.mmap/hybrid/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable testhb"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testhb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testhb.dir/build: testhb
.PHONY : CMakeFiles/testhb.dir/build

CMakeFiles/testhb.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testhb.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testhb.dir/clean

CMakeFiles/testhb.dir/depend:
	cd /home/119010114/.mmap/hybrid/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/119010114/.mmap/hybrid /home/119010114/.mmap/hybrid /home/119010114/.mmap/hybrid/build /home/119010114/.mmap/hybrid/build /home/119010114/.mmap/hybrid/build/CMakeFiles/testhb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testhb.dir/depend
