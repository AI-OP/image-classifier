#include "classifier_float_efficient_net.h"

ClassifierFloatEfficientNet::ClassifierFloatEfficientNet() {
    SetModelName("efficientnet-lite0-fp32.tflite");
    SetLabelName("labels.txt");
    SetImageParameters(127.f, 127.f);
    SetOutputParameters(0.f, 1.f);
}
