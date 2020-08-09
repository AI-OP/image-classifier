# Description:
#   OpenCV libraries for video/image processing on MacOS

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

# The following build rule assumes that OpenCV is installed by
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
#            "local/opt/opencv@3/lib/libopencv_core.dylib",
#            "local/opt/opencv@3/lib/libopencv_calib3d.dylib",
#            "local/opt/opencv@3/lib/libopencv_features2d.dylib",
#            "local/opt/opencv@3/lib/libopencv_highgui.dylib",
#            "local/opt/opencv@3/lib/libopencv_imgcodecs.dylib",
#            "local/opt/opencv@3/lib/libopencv_imgproc.dylib",
#            "local/opt/opencv@3/lib/libopencv_video.dylib",
#            "local/opt/opencv@3/lib/libopencv_videoio.dylib",
            "local/lib/libopencv_core.dylib",
            "local/lib/libopencv_calib3d.dylib",
            "local/lib/libopencv_features2d.dylib",
            "local/lib/libopencv_highgui.dylib",
            "local/lib/libopencv_imgcodecs.dylib",
            "local/lib/libopencv_imgproc.dylib",
            "local/lib/libopencv_video.dylib",
            "local/lib/libopencv_videoio.dylib",
        ],
    ),
    hdrs = glob(["local/include/opencv4/opencv2/**/*.h*"]),
    includes = ["local/include/opencv4"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
