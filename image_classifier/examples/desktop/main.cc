#include <opencv2/opencv.hpp>
#include "absl/memory/memory.h"

#include "image_classifier/cc/utils.h"
#include "image_classifier/cc/image_classify_service.h"

namespace {
    const std::string command = 
        "{c | 0       | Camera device}"
        "{i |         | image path}"
        "{m | models  | model folder}";
}

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv, command);

    CHECK(parser.has("m"), "Has no model...");

    auto image_classify_service = absl::make_unique<ImageClassifyService>();
    std::string model_dir = parser.get<std::string>("m");
    image_classify_service->Init(model_dir, Model::kFloatMobileNet, Device::kCPU, 2); 

    if(parser.has("i")) {

        std::string image_path = parser.get<std::string>("i"); 
        cv::Mat image = cv::imread(image_path);
        assert(!image.empty());
        std::vector<uchar> rgb_image_buffer;
        cv::imencode(".bmp", image, rgb_image_buffer);
        std::vector<std::pair<std::string, float>> result = image_classify_service -> RecognizeImage(rgb_image_buffer);
        CHECK(!result.empty(), "Results should not be empty!");
        std::cout<<"Top1 is "<< result[0].first << " with score "<< result[0].second <<std::endl;

        return 0;
    }

    CHECK(parser.has("c"), "Has no camera device input..."); 

    int device = parser.get<int>("c");

    cv::VideoCapture capture;
    CHECK(capture.open(device), "Can not open devices");

    cv::Mat frame;
    for(;;) { 
        capture >> frame;
        CHECK(!frame.empty(), "Frame is empty has no data.");

        cv::imshow("Debug Image Show", frame);
        char key = cv::waitKey(10);
        if(key == 'q')
            break;
    }

    return 0;
}
