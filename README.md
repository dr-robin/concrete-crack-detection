# Concrete-crack-detection

This is a deep learning model for concrete crack detection using Tensor Flow. The base-model was Inception v3 (https://keras.io/api/applications/inceptionv3/). For transfer learning, I added a feature detection layer and a classification layer to train on concrete wall images from the SDNET2018 dataset which reached an accuracy of 80% after 8 epoches. The training was automatically stopped when the loss did not improve between 2 consecutive epoches. For fine-tuning, I retrained the last 100 layers of the Inception v3 model, which resulted in an accuracy of 90% on the test dataset, with minimal additional training.

## Dataset
The model is trained using the SDNET2018 dataset which is open-source. The directory W contains wall images, C is the prefix used for cracked surfaces and U is the prefix used for uncracked surfaces. The dataset can be downloaded from the link below.

```python
curl https://digitalcommons.usu.edu/cgi/viewcontent.cgi?filename=2&article=1047&context=all_datasets&type=additional
```

## Installation

To run the jupyter notebook use pip to install the requirements.txt. The code was written using Tensorflow v2.

```bash
pip install requirements.txt
```

## Usage

```python
cd concrete-crack-detection
jupyter notebook
```

## Saved model
The trained model is in the ./model directory and saved in the .tflite format due to limited github space.

## Deployment
The model can be deployed on google cloud as a REST-api using the AI prediction platform, using the free tier.

```python
https://cloud.google.com/ai-platform/prediction/docs/deploying-models
```

## Additional
After detecting cracks with the model, I did some additional exploration of ways to assess crack dimension using tools in the skimage library. You can find these also in jupyter notebook. Canny edge detection and OTSU threshold were both promising ways to simply segment the pixels containing a crack.
