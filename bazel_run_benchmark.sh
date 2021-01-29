bazel run -c opt --experimental_repo_remote_exec //image_classifier/apps/desktop:image_classifier_benchmark  -- -i=$PWD/data/goldfish-alcohol.jpg -m=$PWD/models

