#ifndef IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFY_SERVICE_H_
#define IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFY_SERVICE_H_

#include "utils.h"
#include "image_classifier.h"

enum class Model {
    kFloatMobileNet = 0,
    kQuantizedMobileNet = 1,
    kFloatEfficientNet = 2,
    kQuantizedEfficientNet =3,
};

class ImageClassifyService {

public:
    ImageClassifyService() = default;
    ~ImageClassifyService() = default;

public:
	bool Init(std::string model_dir, Model method, Device device, int num_threads);

    //  return the sorted list.
	std::vector<std::pair<std::string, float>> RecognizeImage(Bytes rgb_image_data); 
	
    void Close();
	int GetModelInputSizeX();
	int GetModelInputSizeY();

private:
    std::unique_ptr<ImageClassifier> classifier_;
};

#endif //IMAGE_CLASSIFIER_CC_IMAGE_CLASSIFY_SERVICE_H_

 
