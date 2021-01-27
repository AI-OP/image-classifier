#include "classifier_float_mobile_net.h"

ClassifierFloatMobileNet::ClassifierFloatMobileNet() {
    SetModelName("mobilenet_v1_1.0_224.tflite");
    SetLabelName("labels.txt");
    SetImageParameters(127.f, 127.f);
    SetOutputParameters(0.f, 1.f);
}
