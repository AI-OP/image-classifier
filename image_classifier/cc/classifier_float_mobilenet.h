#ifndef IMAGE_CLASSIFIER_CC_CLASSIFIER_FLOAT_MOBILENET_H_
#define IMAGE_CLASSIFIER_CC_CLASSIFIER_FLOAT_MOBILENET_H_
#include "image_classifier.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

class ClassifierFloatMobileNet: public ImageClassifier {
public:
    ClassifierFloatMobileNet() = default;
    ~ClassifierFloatMobileNet() = default;

    bool Init(std::string model_dir) override;
    std::vector<std::pair<std::string, float>>
        classify(const cv::Mat& image) override;

    void SetThreads(int threads) override;
    void SetDevice(Device device) override;

    int GetModelInputSizeX() override;
    int GetModelInputSizeY() override;

private:
    bool LoadLabelsFile(std::string label_file_path);

private:
    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    //std::vector<Delegate> delegates_;
    std::vector<std::string> labels_;
    int input_tensor_index_;
    int output_tensor_index_;
};


#endif //IMAGE_CLASSIFIER_CC_CLASSIFIER_FLOAT_MOBILENET_H_

