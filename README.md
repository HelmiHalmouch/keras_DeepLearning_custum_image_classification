# cnn-image-classification-model

This project involves the development of a deep learning model for custom image classification using Convolutional Neural Networks (CNN) with the Keras library.

The model utilizes a sequential CNN architecture implemented using Keras for classifying images into four predefined categories.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.5 or higher
- Pytorch
- Matplotlib
- Scikit-learn (sklearn)
- OpenCV

You can install the necessary packages using the following commands:

pip install keras matplotlib scikit-learn opencv-python
```bash
data/
  ├── class1/
  ├── class2/
  ├── class3/
  └── class4/
```
## Architecture of CNN and the parmeters 
![summary_of_model](https://user-images.githubusercontent.com/40611217/50387867-0a457380-0707-11e9-9ec3-ba00c0ef2585.JPG)
## Run the code
```bash
python main.py
```

## Tensorboard 
You can run the code, and view TensorBoard logs using the command:
```bash
tensorboard --logdir=runs
```

if you do not have a dataset and would like to request one, feel free to contact me at helmirealmadrid@gmail.com.

