bazel build -c opt --experimental_repo_remote_exec //image_classifier/apps/desktop:image_classifier_benchmark ;
./bazel-bin/image_classifier/apps/desktop/image_classifier_benchmark -i=data/goldfish-alcohol.jpg

