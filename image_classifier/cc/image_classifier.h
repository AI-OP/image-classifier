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

#ifndef IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_
#define IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "utils.h"

#ifdef WITH_EDGE_TPU
#include "tflite/public/edgetpu.h"
#endif // WITH_EDGE_TPU


class ImageClassifier {
 public:
  virtual bool Init(std::string model_dir);
  virtual std::vector<std::pair<std::string, float>> Classify(
      const cv::Mat& image);

 public:
  virtual int GetThreads();
  virtual bool SetThreads(const int);

  virtual Device GetDevice();
  virtual bool SetDevice(const Device);

 public:
  virtual int GetModelInputSizeX();
  virtual int GetModelInputSizeY();

 public:
  virtual bool SetModelName(const std::string);
  virtual std::string GetModelName() const;

  virtual bool SetLabelName(const std::string);
  virtual std::string GetLabelName() const;

  virtual bool SetImageParameters(const float image_mean,
                                  const float image_std);
  virtual bool GetImageParameters(float& image_mean, float& image_std);

  virtual bool SetOutputParameters(const float probability_mean,
                                   const float probability_std);
  virtual bool GetOutputParameters(float& probability_mean,
                                   float& probability_std);

 public:
  ImageClassifier();
  virtual ~ImageClassifier() = default;

 protected:
  int input_tensor_index_;
  int output_tensor_index_;

 private:
  bool LoadLabelsFile(std::string label_file_path);

 private:
  float image_std_;
  float image_mean_;
  float probability_std_;
  float probability_mean_;

  std::string model_name_;
  std::string label_name_;
  std::vector<std::string> labels_;

  // TensorFlow Lite Settings
  Device device_;
  int num_threads_;

  // std::vector<Delegate> delegates_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;

#ifdef WITH_EDGE_TPU
 protected:
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context_;
#endif // WITH_EDGE_TPU
};

#endif  // IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_
