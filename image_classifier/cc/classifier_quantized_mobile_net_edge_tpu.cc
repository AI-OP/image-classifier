#ifdef WITH_EDGE_TPU

#include "classifier_quantized_mobile_net_edge_tpu.h"

ClassifierQuantizedMobileNetEdgeTPU::ClassifierQuantizedMobileNetEdgeTPU() {
  SetModelName("mobilenet_v1_1.0_224_quant_edgetpu.tflite");
  SetDevice(Device::kEdgeTPU);
}

#endif  // WITH_EDGE_TPU
