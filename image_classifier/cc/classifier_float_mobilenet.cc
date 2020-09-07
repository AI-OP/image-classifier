#include "classifier_float_mobilenet.h"

bool ClassifierFloatMobileNet::Init(std::string model_dir) {
    
    std::string model_name = model_dir+"/mobilenet_v1_1.0_224.tflite";

    model_ = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_); 
    
    assert(kTfLiteOK == interpreter_ -> AllocateTensors());

    input_tensor_index_ = interpreter_ -> inputs()[0];
    output_tensor_index_ = interpreter_ -> outputs()[0];
    
    std::cout<<"input tensor name: "<<interpreter_->tensor(input_tensor_index_)->name<<std::endl;
    std::cout<<"output tensor name: "<<interpreter_->tensor(output_tensor_index_)->name<<std::endl;

    return true;
}

int ClassifierFloatMobileNet::GetModelInputSizeX() {
    return interpreter_ -> tensor(input_tensor_index_) -> dims -> data[1];
}

int ClassifierFloatMobileNet::GetModelInputSizeY() {
    return interpreter_ -> tensor(input_tensor_index_) -> dims -> data[0];
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

    float* input_tensor_buffer = interpreter_ -> typed_input_tensor<float>(input_tensor_index_);
    int buffer_size = sizeof(float)*kNetworkInputWidth*kNetworkInputHeight*kNetworkInputChannels;
    memcpy((uchar*)(input_tensor_buffer), input_image.data, buffer_size);
    assert(kTfLiteStatusOk == interpreter_->Invoke());

    const TfLiteTensor* output_tensor = interpreter_->tensor(output_tensor_index_);
    return {};  
}


