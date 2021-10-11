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

#include "image_classifier.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <string>

ImageClassifier::ImageClassifier() : device_(Device::kCPU), num_threads_(0) {
  ;
}

ImageClassifier::~ImageClassifier() {
#ifdef WITH_EDGE_TPU
  // TODO: Figure out why we need to release interpreter first,
  // and edgetpu_context_ secondly?
  interpreter_ = nullptr;
  edgetpu_context_ = nullptr;
#endif  // WITH_EDGE_TPU
}

bool ImageClassifier::SetThreads(const int threads) {
  CHECK(threads > 0, "Error: threads should be > 0 on SetThreads.");
  CHECK(interpreter_ != nullptr,
        "Interpreter is null. Please init interpreter before setting threads.");

  num_threads_ = threads;
  interpreter_->SetNumThreads(num_threads_);

  return true;
}

bool ImageClassifier::SetDevice(const Device device) {
  // TODO: Add configurations of delegates.
  device_ = device;
  return true;
}

int ImageClassifier::GetThreads() { return num_threads_; }

Device ImageClassifier::GetDevice() { return device_; }

bool ImageClassifier::SetModelName(const std::string model_name) {
  model_name_ = model_name;
  return true;
}

std::string ImageClassifier::GetModelName() const { return model_name_; }

bool ImageClassifier::SetLabelName(const std::string label_name) {
  label_name_ = label_name;
  return true;
}

std::string ImageClassifier::GetLabelName() const { return label_name_; }

bool ImageClassifier::LoadLabelsFile(std::string label_path) {
  std::ifstream label_file(label_path);
  if (!label_file) {
    printf("%s cannot be opened.\n", label_path.c_str());
    return false;
  }

  std::string line;

  while (getline(label_file, line)) {
    labels_.emplace_back(line);
  }

  return !labels_.empty();
}

#ifdef WITH_EDGE_TPU
// TODO: It should be deprecated by using libcoral.
std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}
#endif  // WITH_EDGE_TPU

bool ImageClassifier::Init(std::string model_dir) {
  // Load labels file
  std::string label_path = model_dir + '/' + GetLabelName();
  labels_.clear();
  CHECK(LoadLabelsFile(label_path), "Error load labels file.");

  // Set model path
  std::string model_path = model_dir + '/' + GetModelName();

  // Load model
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  CHECK(model_ != nullptr, "Model cannot be built.");

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);
  CHECK(this->interpreter_ != nullptr, "Interpreter is null.");

#ifdef WITH_EDGE_TPU
  if (GetDevice() == Device::kEdgeTPU) {
    edgetpu_context_ = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    interpreter_ = nullptr;
    interpreter_ = BuildEdgeTpuInterpreter(*model_, edgetpu_context_.get());
    CHECK(this->interpreter_ != nullptr, "Edge TPU Interpreter is null.");
  }
#endif  // WITH_EDGE_TPU

  input_tensor_index_ = interpreter_->inputs()[0];
  output_tensor_index_ = interpreter_->outputs()[0];

  CHECK(kTfLiteOk == interpreter_->AllocateTensors(),
        "Can not allocate tensors.");

  //    tflite::PrintInterpreterState(interpreter_.get());
  return true;
}

int ImageClassifier::GetModelInputSizeX() {
  return interpreter_->tensor(input_tensor_index_)->dims->data[2];
}

int ImageClassifier::GetModelInputSizeY() {
  return interpreter_->tensor(input_tensor_index_)->dims->data[1];
}

bool ImageClassifier::SetImageParameters(const float image_mean,
                                         const float image_std) {
  image_mean_ = image_mean;
  image_std_ = image_std;
  return true;
}

bool ImageClassifier::GetImageParameters(float& image_mean, float& image_std) {
  image_std = image_std_;
  image_mean = image_mean_;
  return true;
}

bool ImageClassifier::SetOutputParameters(const float probability_mean,
                                          const float probability_std) {
  probability_mean_ = probability_mean;
  probability_std_ = probability_std;
  return true;
}

bool ImageClassifier::GetOutputParameters(float& probability_mean,
                                          float& probability_std) {
  probability_mean = probability_mean_;
  probability_std = probability_std_;
  return true;
}

std::vector<std::pair<std::string, float>> ImageClassifier::Classify(
    const cv::Mat& image) {
  if (image.empty()) return {};

  // Input shape: {1, height, width, 3}
  const int kNetworkInputWidth =
      interpreter_->tensor(input_tensor_index_)->dims->data[2];
  const int kNetworkInputHeight =
      interpreter_->tensor(input_tensor_index_)->dims->data[1];
  const int kNetworkInputChannels =
      interpreter_->tensor(input_tensor_index_)->dims->data[3];
  const cv::Size input_size(kNetworkInputWidth, kNetworkInputHeight);

  cv::Mat input_image;
  cv::resize(image, input_image, input_size);
  cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

  TfLiteType image_tf_data_type =
      interpreter_->tensor(input_tensor_index_)->type;

  float kImageMean, kImageStd;
  GetImageParameters(kImageMean, kImageStd);

  CHECK(kTfLiteFloat32 == image_tf_data_type ||
            kTfLiteUInt8 == image_tf_data_type,
        " image_tf_data_type is not uint8 or float32.");

  int cv_data_type = kTfLiteFloat32 == image_tf_data_type ? CV_32F : CV_8U;

  input_image.convertTo(input_image, cv_data_type, 1. / kImageStd,
                        -kImageMean / kImageStd);

  void* input_tensor_buffer =
      interpreter_->tensor(input_tensor_index_)->data.data;

  size_t bytes_size =
      kTfLiteFloat32 == image_tf_data_type ? sizeof(float) : sizeof(uint8_t);
  size_t buffer_size = bytes_size * kNetworkInputWidth * kNetworkInputHeight *
                       kNetworkInputChannels;

  memcpy(input_tensor_buffer, input_image.data, buffer_size);

  // Runing inference.
  CHECK(kTfLiteOk == interpreter_->Invoke(), "Inference invoke error.");

  // Output shape: {1, NUM_CLASSES}
  const TfLiteType output_tf_data_type =
      interpreter_->tensor(output_tensor_index_)->type;
  const int kNumClasses =
      interpreter_->tensor(output_tensor_index_)->dims->data[1];
  const void* output = interpreter_->tensor(output_tensor_index_)->data.data;

  auto compare = [](std::pair<std::string, float> a,
                    std::pair<std::string, float> b) {
    return a.second < b.second;
  };

  std::priority_queue<std::pair<std::string, float>,
                      std::vector<std::pair<std::string, float>>,
                      decltype(compare)>
      pq(compare);

  float kProbabilityMean, kProbabilityStd;
  GetOutputParameters(kProbabilityMean, kProbabilityStd);

  for (int i = 0; i < kNumClasses; i++) {
    float output_value = output_tf_data_type == kTfLiteFloat32
                             ? ((float*)output)[i]
                             : (float)((uint8_t*)output)[i];
    pq.push(std::make_pair(
        labels_[i],
        (float)(output_value - kProbabilityMean) / kProbabilityStd));
  }

  std::vector<std::pair<std::string, float>> label_prob;
  for (int i = 0; i < kNumClasses; i++) {
    label_prob.emplace_back(pq.top());
    pq.pop();
  }

  return label_prob;
}
