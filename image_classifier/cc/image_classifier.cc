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

#include <string>
#include <fstream>

#include "classifier_float_mobilenet.h"

ImageClassifier::ImageClassifier(): device_(kCPU), num_threads_(0)


bool ImageClassifier::SetThreads(const int threads) {
    CHECK(threads > 0, "Error: threads should be > 0 on SetThreads.");
    CHECK(interpreter_ != nullptr, "Interpreter is null. Please init interpreter before setting threads.");
    
    num_threads_ = threads;
    interpreter_ ->SetNumThreads(num_threads_);
    
    return true;
}

bool ImageClassifier::SetDevice(const Device device) {
    // TODO: Add configurations of delegates. 
    device_ = device;
    return true;
}

int ImageClassifier::GetThreads() {
    return num_threads_; 
}

Device ImageClassifier::GetDevice() {
    return device_;
}

bool ImageClassifier::SetModelName(const std::string model_name) {
    model_name_ = model_name;
    return true;
}

std::string ImageClassifier::GetModelName() {
    return model_name_;
}

bool ImageClassifier::SetLabelName(cosnt std::string label_name) {
    label_name_ = label_name;
    return true;
}

bool ImageClassifier::LoadLabelsFile(std::string label_path) {
    std::ifstream label_file(label_path);
    if(!label_file) {
        printf("%s cannot be opened.\n", label_path.c_str());
        return false;
    }

    std::string line; 

    while(getline(label_file, line)) {
        labels_.emplace_back(line);
    }

    return !labels_.empty();
}

bool ImageClassifier::Init(std::string model_dir) {
    
    // Load labels file
    std::string label_path = model_dir + '/' + GetLabelName();
    labels_.clear();
    CHECK(LoadLabelsFile(label_path), "Error load labels file.");

    // Set model path
    std::string model_path = model_dir + '/' + GetModelName();
    std::cout<<model_path<<std::endl;

    // Load model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    CHECK(model_ != nullptr, "Model cannot be built.");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_); 
    CHECK(this->interpreter_ != nullptr, "Interpreter is null.");

    input_tensor_index_ = interpreter_ -> inputs()[0];
    output_tensor_index_ = interpreter_ -> outputs()[0];
 
    CHECK(kTfLiteOk == interpreter_ -> AllocateTensors(), "Can not allocate tensors.");
   
//    tflite::PrintInterpreterState(interpreter_.get());
    return true;
}

int ImageClassifier::GetModelInputSizeX() {
    return interpreter_ -> tensor(input_tensor_index_) -> dims -> data[2];
}

int ImageClassifier::GetModelInputSizeY() {
    return interpreter_ -> tensor(input_tensor_index_) -> dims -> data[1];
}

void ImageClassifier::SetThreads(int threads) {
    interpreter_->SetNumThreads(threads);
}

bool SetImageParameters(const float image_mean, const float image_std) {
    image_mean_ = image_mean;
    image_std_ = image_std;
    return true; 
}

bool GetImageParameters(float& image_mean, float& image_std) {
    image_std = image_std_;
    image_mean = image_mean_;
    return true;
}

bool SetOutputParameters(const float probability_mean,
        const float probability_std) {
    probability_mean_ = probability_mean;
    probability_std_ = probability_std;
    return true;
}

bool GetOutputParameters(float& probability_mean,
        float& probability_std) {
    probability_mean = probability_mean_;
    probability_std = probability_std_;
    return true;
}

std::vector<std::pair<std::string, float>>
ImageClassifier::classify(const cv::Mat& image) {

    if(image.empty())
        return {};
    
    // Input shape: {1, height, width, 3}
    const int kNetworkInputWidth = interpreter_ -> tensor(input_tensor_index_) -> dims -> data[2];
    const int kNetworkInputHeight = interpreter_ -> tensor(input_tensor_index_) -> dims -> data[1];
    const int kNetworkInputChannels = interpreter_ -> tensor(input_tensor_index_) -> dims -> data[3];
    const cv::Size input_size(kNetworkInputWidth, kNetworkInputHeight);
    
    cv::Mat input_image;
    cv::resize(image, input_image, input_size);
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
    

    float kImageMean, kImageStd;
    GetImageParameters(kImageMean, kImageStd);

    input_image.convertTo(input_image, CV_32F, 1. / kImageStd, - kImageMean / kImageStd);
    float* input_tensor_buffer = interpreter_ -> typed_tensor<float>(input_tensor_index_);
    int buffer_size = sizeof(float)*kNetworkInputWidth*kNetworkInputHeight*kNetworkInputChannels;
    memcpy((uchar*)(input_tensor_buffer), input_image.data, buffer_size);

    // Runing inference.
    CHECK(kTfLiteOk == interpreter_->Invoke(), "Inference invoke error.");

    // Output shape: {1, NUM_CLASSES}
    const float* output = interpreter_->typed_tensor<float>(output_tensor_index_);
    const int kNumClasses = interpreter_->tensor(output_tensor_index_)->dims->data[1]; 

    auto compare = 
        [](std::pair<std::string, float> a, std::pair<std::string, float> b){
            return a.second < b.second;
        };

    std::priority_queue< std::pair<std::string,float>,
        std::vector<std::pair<std::string, float>>,  
        decltype(compare) > pq(compare);

    float kProbabilityMean, kProbabilityStd;
    GetOutputParameters(kProbabilityMean, kProbabilityStd);

    for(int i = 0 ; i < kNumClasses; i ++) {
       pq.push(std::make_pair(labels_[i],
               (float) (output[i] - kProbabilityMean) / kProbabilityStd);
    }

    std::vector<std::pair<std::string, float>> label_prob;
    for(int i = 0; i < kNumClasses; i ++) {
        label_prob.emplace_back(pq.top());
        pq.pop();
    }

    return label_prob; 
}


