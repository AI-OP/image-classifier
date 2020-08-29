#ifndef IMAGE_CLASSIFIER_CC_UTILS_H_
#define IMAGE_CLASSIFIER_CC_UTILS_H_

#include <cstdio>
#include <vector>

#define CHECK(condition, error_info) \
    if(!condition) { \
        std::cout<<error_info<<std::endl;\
        exit(-1); \
    }\

// Todo:
// Find more easy way to print our content
#define LOG_INFO(content) \
    std::cout<<content<<std::endl;

enum class Device{
    kCPU = 0,
    kNNAPI = 1,
    kGPU = 2,
};

typedef std::vector<uchar> Bytes;

#endif //IMAGE_CLASSIFIER_CC_UTILS_H_ 
