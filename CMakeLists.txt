# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(nvsample_cudaprocess)
# from v3.10 of cmake can use this instead of find_package of cuda (and then use add_executable and add_library)
# ther add_executable and add_library(for making shared and dynamic libraries) ------------> making targets 
enable_language(CUDA)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/include/trackers)
include_directories(/usr/local/include/jetson-utils)

link_directories(/usr/lib/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)
link_directories(/usr/local/lib)

##-------------------------------------------------------------------------
#FILE(GLOB SRC "src/*.cpp")
FILE(GLOB CUDA_SRC  "*.cu")
#SET(SRC ${SRC} ${CUDA_SRC})
SET(SRC ${CUDA_SRC})

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    include path: ${OpenCV_LIBS}")####-----added
#SET(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS "-O2")

#----------------------------------------------------------------------------
# Declare the executable target built from your sources
#add_executable(${PROJECT_NAME} ${SRC})
add_library(${PROJECT_NAME} SHARED
    ${SRC}
    )

# Link your application with OpenCV and cuda libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} -lcufft -lEGL -lGLESv2 -lcuda -lrt -ljetson-utils -lKCF)
#target_link_libraries(-lcufft -lEGL -lGLESv2 -lcuda -lrt -ljetson-utils -lKCF)


# Run After
# mkdir build && cd build
# cmake ..
# make ( or cmake --build . --config Release)

