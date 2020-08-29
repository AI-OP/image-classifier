#ifndef IMAGE_CLASSIFIER_CC_CLASSIFIER_FLOAT_MOBILENET_H_
#define IMAGE_CLASSIFIER_CC_CLASSIFIER_FLOAT_MOBILENET_H_
#include "image_classifier.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

class ClassifierFloatMobileNet {
public:
    ClassifierFloatMobileNet() = default;
    ~ClassifierFloatMobileNet() = default;

    bool Init(std::string model_dir);
    std::vector<std::pair<std::string, float>>
        classify(const cv::Mat& image);

    int GetModelInputSizeX() override;
    int GetModelInputSizeY() override;

private:
    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    //std::vector<Delegate> delegates_;
    std::vector<std::string> labels_;
    int input_tensor_index_;
    int ouput_tensor_index_;
};


#endif //IMAGE_CLASSIFIER_CC_CLASSIFIER_FLOAT_MOBILENET_H_

