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

#include "image_classifiers.h"

#include "classifier_float_efficient_net.h"
#include "classifier_float_mobile_net.h"
#include "classifier_quantized_efficient_net.h"
#include "classifier_quantized_mobile_net.h"

#ifdef WITH_EDGE_TPU
#include "classifier_quantized_efficient_net_edge_tpu.h"
#include "classifier_quantized_mobile_net_edge_tpu.h"
#endif  // WITH_EDGE_TPU

std::unique_ptr<ImageClassifier> ImageClassifiers::CreateImageClassifier(
    const Model method) {
  std::unique_ptr<ImageClassifier> classifier = nullptr;

  switch (method) {
    case Model::kFloatMobileNet:
      classifier = absl::make_unique<ClassifierFloatMobileNet>();
      break;
    case Model::kQuantizedMobileNet:
      classifier = absl::make_unique<ClassifierQuantizedMobileNet>();
      break;
    case Model::kFloatEfficientNet:
      classifier = absl::make_unique<ClassifierFloatEfficientNet>();
      break;
    case Model::kQuantizedEfficientNet:
      classifier = absl::make_unique<ClassifierQuantizedEfficientNet>();
      break;
#ifdef WITH_EDGE_TPU
    case Model::kQuantizedMobileNetEdgeTPU:
      classifier = absl::make_unique<ClassifierQuantizedMobileNetEdgeTPU>();
      break;
    case Model::kQuantizedEfficientNetEdgeTPU:
      classifier = absl::make_unique<ClassifierQuantizedEfficientNetEdgeTPU>();
      break;
#endif  // WITH_EDGE_TPU
    default:
      classifier = nullptr;
  }

  return classifier;
}
