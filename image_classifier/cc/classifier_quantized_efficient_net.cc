#include "classifier_quantized_efficient_net.h"

ClassifierQuantizedEfficientNet::ClassifierQuantizedEfficientNet() {
    SetModelName("efficientnet-lite0-int8.tflite");
    SetLabelName("labels.txt");
    SetImageParameters(0.f, 1.f);
    SetOutputParameters(0.f, 255.f);
}
