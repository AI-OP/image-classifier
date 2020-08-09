#include <opencv2/opencv.hpp>

int main(int argc, char**){
    cv::Mat image = cv::Mat::ones(300, 300, CV_8UC3);
    image.setTo(cv::Scalar(0,255,0));

    cv::imshow("aaa",image);
    cv::waitKey();
    return 0;
}
