#ifndef IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_
#define IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_
#include "utils.h"

class ImageClassifier {
public:
    ImageClassifier() = default;
    ~ImageClassifier() = default;

public:
    virtual bool Init(std::string model_dir) = 0;
    virtual int GetModelInputSizeX() = 0;
    virtual int GetModelInputSizeY() = 0;
    virtual std::vector<std::pair<std::string, float>>
        classify(const cv::Mat& image) = 0;

public:
    virtual void SetThreads(int);
    virtual void SetDevice(Device);
};

#endif // IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_

