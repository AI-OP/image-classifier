#include "classifier_quantized_mobile_net.h"

ClassifierQuantizedMobileNet::ClassifierQuantizedMobileNet() {
    SetModelName("mobilenet_v1_1.0_224_quant.tflite");
    SetLabelName("labels.txt");
    SetImageParameters(0.f, 1.f);
    SetOutputParameters(0.f, 255.f);
}
