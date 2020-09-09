#include "classifier_float_mobilenet.h"

bool ClassifierFloatMobileNet::Init(std::string model_dir) {
        
    std::string model_path = model_dir+"/mobilenet_v1_1.0_224.tflite";
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

int ClassifierFloatMobileNet::GetModelInputSizeX() {
    return interpreter_ -> tensor(input_tensor_index_) -> dims -> data[2];
}

int ClassifierFloatMobileNet::GetModelInputSizeY() {
    return interpreter_ -> tensor(input_tensor_index_) -> dims -> data[1];
}

void ClassifierFloatMobileNet::SetThreads(int threads) {
    interpreter_->SetNumThreads(threads);
}

void ClassifierFloatMobileNet::SetDevice(Device device) {
    ;
}

std::vector<std::pair<std::string, float>>
ClassifierFloatMobileNet::classify(const cv::Mat& image) {

    if(image.empty())
        return {};

    const int kNetworkInputWidth = interpreter_ -> tensor(input_tensor_index_) -> dims -> data[2];
    const int kNetworkInputHeight = interpreter_ -> tensor(input_tensor_index_) -> dims -> data[1];
    const int kNetworkInputChannels = interpreter_ -> tensor(input_tensor_index_) -> dims -> data[3];
    const cv::Size input_size(kNetworkInputWidth, kNetworkInputHeight);
    
    cv::Mat input_image;
    cv::resize(image, input_image, input_size);
    const float kImageMean = 127.5f;
    const float kImageStd = 127.5f;
    input_image.convertTo(input_image, CV_32F, 1. / kImageStd, - kImageMean);
    float* input_tensor_buffer = interpreter_ -> typed_tensor<float>(input_tensor_index_);
    int buffer_size = sizeof(float)*kNetworkInputWidth*kNetworkInputHeight*kNetworkInputChannels;
    memcpy((uchar*)(input_tensor_buffer), input_image.data, buffer_size);
    CHECK(kTfLiteOk == interpreter_->Invoke(), "Inference invoke error.");

    const TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_index_);
    return {};  
}


