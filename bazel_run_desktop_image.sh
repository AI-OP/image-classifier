# Copyright 2021 Sun Aries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

bazel run -c opt --experimental_repo_remote_exec //image_classifier/apps/desktop:image_classifier -- -i=data/goldfish-alcohol.jpg -m=models  --verbose_failures ;
bazel run -c opt --experimental_repo_remote_exec //image_classifier/apps/desktop:image_classifier -- -c=0 -m=models
