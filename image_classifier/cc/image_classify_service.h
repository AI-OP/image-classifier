// Copyright 2020 Sun Aries.
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

#ifndef IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFY_SERVICE_H_
#define IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFY_SERVICE_H_

#include "image_classifiers.h"
#include "utils.h"

class ImageClassifyService {
 public:
  ImageClassifyService() = default;
  ~ImageClassifyService() = default;

 public:
  bool Init(std::string model_dir, Model method, Device device,
            int num_threads);

  //  return the sorted list.
  std::vector<std::pair<std::string, float>> RecognizeImage(
      Bytes rgb_image_data);
  void Close();

 public:
  int GetModelInputSizeX();
  int GetModelInputSizeY();

 private:
  std::unique_ptr<ImageClassifier> classifier_;
};

#endif  // IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFY_SERVICE_H_
