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

#ifndef CLASSIFIER_QUANTIZED_EFFICIENT_NET_EDGE_TPU_H_
#define CLASSIFIER_QUANTIZED_EFFICIENT_NET_EDGE_TPU_H_

#include "classifier_quantized_efficient_net.h"

class ClassifierQuantizedEfficientNetEdgeTPU
    : public ClassifierQuantizedEfficientNet {
 public:
  ClassifierQuantizedEfficientNetEdgeTPU();
  ClassifierQuantizedEfficientNetEdgeTPU() = default;
};

#endif  // CLASSIFIER_QUANTIZED_EFFICIENT_NET_EDGE_TPU_H_

#endif  // WITH_EDGE_TPU