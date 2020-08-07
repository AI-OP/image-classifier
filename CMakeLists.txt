# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(planner_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

aux_source_directory(cc DIR_CC)

set(TFLite_DIR third_party/tensorflow/lite)
set(TFLite_LIB_DIR ${TFLite_DIR}/gen/linux_x86_64/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(TFLite_LIBS ${CMAKE_SOURCE_DIR}/${TFLite_LIB_DIR}/libtensorflow-lite.a)

include_directories( "cc" )
include_directories( "third_party")
include_directories( ${TFLite_DIR} )
include_directories( ${TFLite_DIR}/tools/make/downloads/flatbuffers/include )

add_executable(demo.exe  demo.cc ${DIR_CC} )
add_executable(minimal.exe  test.cc ${DIR_CC} )

target_link_libraries(demo.exe LINK_PRIVATE ${TFLite_LIBS} ${OpenCV_LIBS} -ldl)
target_link_libraries(minimal.exe LINK_PRIVATE ${TFLite_LIBS} ${OpenCV_LIBS} -ldl)
