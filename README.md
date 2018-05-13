# Predict Facial Attractiveness

> Using OpenCV and Dlib to predict facial attractiveness.

![landmarks](https://raw.githubusercontent.com/LiuXiaolong19920720/predict-facial-attractiveness/master/image/landmarks.JPG)

# Init env on MacOS

This project was written by python2, but it has been refactored by python3, and it runs with opencv and dlib, so we should init the environment at first.

## Install Python3

```shell
$ brew install python3
```

## Install Numpy & Sklearn

```shell
$ pip3 install numpy
$ pip3 install sklearn
```

## Install & Config OpenCV2

### Install OpenCV2

```shell
$ brew install opencv
```

### Config OpenCV2 into Python Env

Enter into your python site-packages path, I use python virtual env, and it is `~/Projects/pyvenv/lib/python3.6/site-packages`, then create soft link for cv2.so.

```shell
$ ln -s /usr/local/Cellar/opencv/3.4.0_1/lib/python3.6/site-packages/cv2.cpython-36m-darwin.so cv2.so
```

After that, you can use cv2 in your python project.

```python
import cv2
```

## Install Dlib

Execute commands below, it may spend some time to setup dlib, be patient.

```shell
$ brew install cmake
$ brew install dlib
$ pip install dlib
```

## Download shape_predictor_68_face_landmarks.dat

Download data file from [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

And replace the file: `data/shape_predictor_68_face_landmarks.dat`

# Run

The workflow of this project is:

```
1. Train Model
2. Get Landmarks
3. Generate Features
4. Predict
```

Every step has a python file, you can run them one by one.

### Train Model

Generate the model which will be used to predict facial attractiveness later, it will read data from `data/train_features` and `/data/train_ratings`.

```shell
$ python source/trainModel.py
```

### Get Landmarks

Get the landmarks of the faces detected in the `image/test.jpg`, it will read data from `data/shape_predictor_68_face_landmarks.dat` and then output a file named `data/my_landmarks`.

```shell
$ python source/getLandmarks.py
```

###  Generate Features

Generate the features which will be as the input of the model we get before, it will output a file named `data/my_features`.

```shell
$ python source/generateFeatures.py
```

### Predict

Predict facial attractiveness

```shell
$ python source/myPredict.py
```

### Attention

If you want to run `source/run.py` to reduce operations, you should pay attention to that maybe program read a empty file which is generated by the previous step.

```shell
$ python source/run.py
```

# Refactor Records

1. Use the grammer of python3 to make it run, such as `print` to `print()`, change `min_samples_split = 1` into `min_samples_split = 1.0` and reshape the dimensions of array data in some necessary place.

2. There should be a loop in the `trainModel` process, if not, the predict result will be all the same, because the `my_face_rating.pkl` was generated incompletely.

```
[
  [ 2.70908844]
  [ 2.70908844]
  [ 2.70908844]
  [ 2.70908844]
  [ 2.70908844]
  [ 2.70908844]
]
```

3. If just one face was detected, it would be error. Fix it by reshape array data in this case.

4. Rename some files to make the structure of this project more clear.
