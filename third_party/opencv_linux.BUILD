# Description:
#   OpenCV libraries for video/image processing on Linux

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by instructions:
# `$ git clonegit clone https://github.com/Itseez/opencv.git `
# `$ cd opencv/ `
# `$ mkdir build install `
# `$ cd build `
# `$ cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/path/to/opencv/install .. `
# `$ make install `

cc_library(
    name = "opencv",
    srcs = glob(
        [
            "local/lib/libopencv_*.so",
        ],
    ),
    hdrs = glob(["local/include/opencv4/opencv2/**/*.h*"]),
    includes = ["local/include/opencv4"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
