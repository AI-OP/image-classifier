#include "classifier_float_mobilenet.h"

bool ClassifierFloatMobileNet::Init(std::string model_dir) {
    
    std::string model_name = model_dir+"/mobilenet_v1_1.0_224.tflite";

    model_ = tflite::FlatBufferModel::BuildFromFile(model_name);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(interpreter_); 
    
    assert(kTfLiteOK == interpreter_ -> AllocateTensors());

    input_tensor_index_ = interpreter_ -> inputs()[0];
    outpu_tensor_index_ = interpreter_ -> outputs()[0];
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

    const int kNetworkInputWidth = interpreter_ -> dims -> data[2];
    const int kNetworkInputHeight = interpreter_ -> dims -> data[1];

    const cv::Size image_size(kNetworkInputWidth, kNetworkInputHeight);

}


