cc_library(
    name = "opencv",
    visibility = ["//visibility:public"],
    deps = select({
        # "//mediapipe:android_x86": [
        #     "@android_opencv//:libopencv_x86",
        # ],
        # "//mediapipe:android_x86_64": [
        #     "@android_opencv//:libopencv_x86_64",
        # ],
        # "//mediapipe:android_armeabi": [
        #     "@android_opencv//:libopencv_armeabi-v7a",
        # ],
        # "//mediapipe:android_arm": [
        #     "@android_opencv//:libopencv_armeabi-v7a",
        # ],
        # "//mediapipe:android_arm64": [
        #     "@android_opencv//:libopencv_arm64-v8a",
        # ],
        # "//mediapipe:ios": [
        #     "@ios_opencv//:opencv",
        # ],
		# "//mediapipe:windows": [
        #     "@windows_opencv//:opencv",
        # ],
        # "//mediapipe:macos": [
        #     "@macos_opencv//:opencv",
        # ],
        "//conditions:default": [
            "@linux_opencv//:opencv",
        ],
    }),
)