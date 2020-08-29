#ifndef IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_
#define IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_
#include <cstdio>

class ImageClassifier {
public:
    ImageClassifier() = default;
    ~ImageClassifier() = default;
    bool Init(std::string model_dir) = 0;
    void SetThreads(int);
    void SetDevice(Device);
    virtual std::vector<std::pair<std::string, float>>
        classify(const cv::Mat& image) = 0;
    
    int GetModelInputSizeX();
    int GetModelInputSizeY();

};

#endif // IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFIER_H_

