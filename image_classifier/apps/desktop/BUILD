# Copyright 2020 Sun Aries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "image_classifier",
    deps = [
        "image_classifier_cc",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "image_classifier_benchmark",
    deps = [
        "image_classifier_benchmark_cc",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "image_classifier_cc",
    srcs = ["main.cc"],
    deps = [
        "@//third_party:opencv",
        "//image_classifier/cc:image_classifier",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "image_classifier_benchmark_cc",
    srcs = ["benchmark.cc"],
    deps = [
        "@//third_party:opencv",
        "//image_classifier/cc:image_classifier"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "benchmark_edgetpu_cc",
    srcs = ["benchmark.cc"],
    copts = ["-DWITH_EDGE_TPU=1"],
    deps = [
        "@//third_party:opencv",
        "//image_classifier/cc:image_classifier_edgetpu"
    ],
    visibility = ["//visibility:public"],
)

