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

#ifndef IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIERS_H_
#define IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIERS_H_

#include "image_classifier.h"

enum class Model {
  kFloatMobileNet = 0,
  kQuantizedMobileNet,
  kFloatEfficientNet,
  kQuantizedEfficientNet,
#ifdef WITH_EDGE_TPU
  kQuantizedMobileNetEdgeTPU,
  kQuantizedEfficientNetEdgeTPU,
#endif  // WITH_EDGE_TPU
};

class ImageClassifiers {
 public:
  static std::unique_ptr<ImageClassifier> CreateImageClassifier(const Model);

 public:
  ImageClassifiers() = default;
  ~ImageClassifiers() = default;
};

#endif  // IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIERS_H_
