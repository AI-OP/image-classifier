#include <opencv2/opencv.hpp>

#include "image_classifier/cc/utils.h"


namespace {
    const std::string command = 
        "{c | 0       | Camera devices}"
        "{m | models  | model_dir}";
}

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv, command);

    CHECK(parser.has("m"), "Has no model...");
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
