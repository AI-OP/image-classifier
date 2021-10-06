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

cc_library(
  name = "image_classifier",
  srcs = [
    "image_classifier.cc",
    "image_classifiers.cc",
    "image_classify_service.cc",
    "classifier_float_mobile_net.cc",
    "classifier_float_efficient_net.cc",
    "classifier_quantized_mobile_net.cc",
    "classifier_quantized_efficient_net.cc",
  ],
  hdrs = [
    "utils.h",
    "image_classfier.h",
    "image_classifiers.h",
    "image_classify_service.h",
    "classifier_float_mobile_net.h",
    "classifier_float_efficient_net.h",
    "classifier_quantized_mobile_net.h",
    "classifier_quantized_efficient_net.h",
  ],
  visibility = ["//visibility:public"],
  deps = [
    "@com_google_absl//absl/memory",
    "@org_tensorflow//tensorflow/lite:builtin_op_data",
    "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    "@org_tensorflow//tensorflow/lite:framework",
    "@//third_party:opencv",
  ],
)

cc_library(
   name = "image_classifier_edgetpu",
   srcs = [
     "image_classifier.cc",
     "image_classifiers.cc",
     "image_classify_service.cc",
     "classifier_quantized_mobile_net.cc",
     "classifier_quantized_efficient_net.cc",
     "classifier_quantized_mobile_net_edge_tpu.cc",
     "classifier_quantized_efficient_net_edge_tpu.cc",
   ],

   hdrs = [
     "utils.h",
     "image_classfier.h",
     "image_classifiers.h",
     "image_classify_service.h",
     "classifier_quantized_mobile_net.h",
     "classifier_quantized_efficient_net.h",
     "classifier_quantized_mobile_net_edge_tpu.h",
     "classifier_quantized_efficient_net_edge_tpu.h",
   ],

   copts = ["-DWITH_EDGE_TPU"],

   deps = [
     "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
     "@com_google_absl//absl/memory",
     "@org_tensorflow//tensorflow/lite:builtin_op_data",
     "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
     "@org_tensorflow//tensorflow/lite:framework",
     "@//third_party:opencv",
   ],
)

