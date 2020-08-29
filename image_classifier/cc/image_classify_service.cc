#include "image_classify_service.h"


bool ImageClassifyService::Init(std::string model_dir,
        Model method, Device device, int num_threads) {

    switch(method) {
        case kFloatMobileNet:
            classifier_ = new ClassifierFloatMobileNet();
            break;
//        case kQuantizedMobileNet:
//            classifier_ = new ClassifierQuantizedMobileNet();
//            break;
//        case kFloatEfficientNet:
//            classifier_ = new ClassifierFloatEfficientNet();
//            break;
//        case kQuantizedEfficientNet:
//            classifier_ = new ClassifierQuantizedEfficientNet();
//            break;
        default:
            ;
    }

    classifier_ -> Init(model_dir);
    classifier_ -> SetThreads(num_threads);
    classifier_ -> SetDevice(device);

    return true;
}

void ImageClassifyService::Close() {
    delete classifier;
}

int ImageClassifyService::GetModelInputSizeX() {
    return classifier_->GetModelInputSizeX();
}

int ImageClassifyService::GetModelInputSizeY() {
    return classifier_->GetModelInputSizeY();
}

