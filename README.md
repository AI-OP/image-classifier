# image-classifier

## Framework (WIP)

![](https://github.com/SunAriesCN/image-classifier/blob/master/docs/image_classifier_framework.png)


## Usage
For a PC desktop platform, we firstly install the OpenCV.
Your can build it from sources by following instructions.
```
$ git clone https://github.com/Itseez/opencv.git 
$ cd opencv/ 
$ mkdir build install
$ cd build 
$ cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/path/to/opencv/install .. 
$ make install 
```
Otherwise, you can also get it from apt on Linux or homebrew from MacOS.

Then run the applications on this repo.
```
$ ./bazel_run_benchmark.sh
$ ./bazel_run_desktop_image.sh
```

For Raspberry Pi platform, we can run the following command.
```
$ ./build_rpi.sh
```

## TODO
1. Please write the README.md seriously ok....o(-.-)o  
  [a] Well, framework graph is out, it seem better now.
2. Android app waiting for you, Sun, come on!
3. Anyone can write the IOS app for this repo, Sun is not well on IOS development.
4. [Completed]Raspberry Pi compilation files seem not be found on this repo. Sun!!! DO IT QUICKLY!!! OK? (T.T)  
  [a] Usage has showed the all.  
  [b] You can also go to my friend's [repo](https://github.com/Duan-JM/bazel-crosstools-compiler) for more details.
   
