#ifndef IMAGE_CLASSIFIER_CC_UTILS_H_
#define IMAGE_CLASSIFIER_CC_UTILS_H_

#define CHECK(condition, error_info) \
    if(!condition) { \
        std::cout<<error_info<<std::endl;\
        exit(-1); \
    }\

#endif //IMAGE_CLASSIFIER_CC_UTILS_H_ 
