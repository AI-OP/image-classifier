#include <opencv2/opencv.hpp>

int main(int argc, char** argv){

	cv::Mat image = cv::Mat::zeros(200, 200, CV_8UC3);

	cv::imshow("image", image);
	
	return 0;
}
