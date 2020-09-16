#include "image_classify_service.h"
#include "classifier_float_mobilenet.h"

bool ImageClassifyService::Init(std::string model_dir,
        Model method, Device device, int num_threads) {

    switch(method) {
        case Model::kFloatMobileNet:
            classifier_ = absl::make_unique<ClassifierFloatMobileNet>();
            break;
//        case Model::kQuantizedMobileNet:
//            classifier_ = absl::make_unique<ClassifierQuantizedMobileNet>();
//            break;
//        case Model::kFloatEfficientNet:
//            classifier_ = absl::make_unique<ClassifierFloatEfficientNet>();
//            break;
//        case Model::kQuantizedEfficientNet:
//            classifier_ = absl::make_unique<ClassifierQuantizedEfficientNet>();
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
    ;
}

int ImageClassifyService::GetModelInputSizeX() {
    return classifier_->GetModelInputSizeX();
}

int ImageClassifyService::GetModelInputSizeY() {
    return classifier_->GetModelInputSizeY();
}

std::vector<std::pair<std::string, float>> 
ImageClassifyService::RecognizeImage(Bytes rgb_image_data) {
    cv::Mat image = cv::imdecode(rgb_image_data, cv::IMREAD_COLOR);
    return classifier_->classify(image);
}  
