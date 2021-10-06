// Copyright 2021 Sun Aries.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef WITH_EDGE_TPU

#ifndef IMAGE_CLASSIFIER_CC_CLASSIFIER_QUANTIZED_MOBILE_NET_EDGE_TPU_H_
#define IMAGE_CLASSIFIER_CC_CLASSIFIER_QUANTIZED_MOBILE_NET_EDGE_TPU_H_
#include "classifier_quantized_mobile_net_edge_tpu.h"
class ClassifierQuantizedMobileNetEdgeTPU
    : public ClassifierQuantizedMobileNet {
 public:
  ClassifierQuantizedMobileNetEdgeTPU();
  ~ClassifierQuantizedMobileNetEdgeTPU() = default;
};

#endif  // IMAGE_CLASSIFIER_CC_CLASSIFIER_QUANTIZED_MOBILE_NET_EDGE_TPU_H_

#endif  // WITH_EDGE_TPU
