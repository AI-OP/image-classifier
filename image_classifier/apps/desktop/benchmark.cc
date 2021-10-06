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

#include <opencv2/opencv.hpp>

#include "absl/memory/memory.h"
#include "image_classifier/cc/image_classifiers.h"
#include "image_classifier/cc/utils.h"

namespace {
const std::string command =
    "{i |         | image path}"
    "{m | models  | model folder}";
}

int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, command);

  CHECK(parser.has("i"), "Has no image");
  const std::string kImagePath = parser.get<std::string>("i");
  cv::Mat image = cv::imread(kImagePath);
  if (image.empty()) {
    printf("Error: Image %s can not be read.\n", kImagePath.c_str());
    return -1;
  }

  CHECK(parser.has("m"), "Has no model...");
  const std::string kModelDir = parser.get<std::string>("m");

  const int kTopN = 5;
  const int kTestCount = 10;

  for (int model = (int)Model::kFloatMobileNet;
       model <= (int)Model::kQuantizedEfficientNet; model++) {
    std::unique_ptr<ImageClassifier> image_classifier =
        ImageClassifiers::CreateImageClassifier((Model)model);

    image_classifier->Init(kModelDir);

    cv::TickMeter tick_meter;
    std::vector<std::pair<std::string, float>> results;
    for (int i = 0; i < kTestCount; i++) {
      tick_meter.start();
      results = image_classifier->Classify(image);
      tick_meter.stop();
    }

    printf("%s results:\n", image_classifier->GetModelName().c_str());
    printf("Running time: %lf ms with %lf fps.\n", tick_meter.getAvgTimeMilli(),
           tick_meter.getFPS());

    CHECK(results.size() >= kTopN, "error in results.size() >= kTopN.");

    for (int i = 0; i < kTopN; i++) {
      printf("Top-%d is %s with %f \n", i + 1, results[i].first.c_str(),
             results[i].second);
    }
  }

  return 0;
}
