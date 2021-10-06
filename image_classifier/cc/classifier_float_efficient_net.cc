// Copyright 2021 Sun Aries.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "classifier_float_efficient_net.h"

ClassifierFloatEfficientNet::ClassifierFloatEfficientNet() {
  SetModelName("efficientnet-lite0-fp32.tflite");
  SetLabelName("labels.txt");
  SetImageParameters(127.f, 127.f);
  SetOutputParameters(0.f, 1.f);
}
