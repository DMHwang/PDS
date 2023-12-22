# PDS: Solar Flare Detection Using Machine Learning

Raymond Blaha, Paul Hwang, David Nezelek

## Project Overview

Solar flares have important implications with our solar system. We will attempt to improve upon the prexisting model that NASA has using machine learning tools and algorithms.

## Project Contributions

*Raymond Blaha*
- Autoencoder:
- Encoder:
Conv1D Layers:
The first Conv1D layer has 16 filters, a kernel size of 3, and utilizes “ReLU” activation. Ensuring that the model is able to capture local patterns in the XSRA AND XRSB solar flare flux data. The “same” padding is used to ensure that the output size is the same as the input size. 
The second Conv1D later, also utilizes “ReLU” activation and increases the filter count to 32. This layer in particular refines the feature extraction process by capturing more complex features in the data.
MaxPooling1D Layers: 
Following each instance of the Conv1D layer, a MaxPooling1D layer with a pool size of 2 is used. The importance of this layer is to ensure that the dimensionality is reduced allowing for features to be visible to the model and computational costs. 
Dropout Layers:
A dropout layer of 0.3 is selected after the MaxPooling1D. The dropout layer ensures that the model does not fall victim to overfitting. This is conducted by randomly setting a fraction of the input units to 0 at each update during training.
Flatten Layers: 
The Flatten layer converts the 2D feature maps into a 1D feature vector. This is important for the transition from convolutional layers to dense layers. 
Dense Layers (Latent Space Representation)
A Dense layer with the latent_dim units of 10 and “ReLU” activation. The units in this particular case represent the 10-dimensional latent space vector that contains the salient features of the input data. The 10-dimensional vector was chosen in this application to be the ideal balance between simplicity and sufficient complexity to represent the features while ensuring the model does not overfit.

- Decoder:
Dense and Reshape Layers:
The decoder starts with the Dense layer sized to upscale the latent representation, followed by the Reshape later to prepare for transposed convolutional layers.
BatchNormalization Layer:
This layer normalizes the activations of the previous layers, which improves the stability and speed of the model’s training. 
Conv1DTranspose Layers: 
This reverses the operations of the Conv1D, upsampling back to the original dimension. The first instance of the Conv1DTranspose layer contains 32 filters and the second Conv1D has 16 filters, which both employ “ReLU” activation and “same” padding. 
Final Conv1D Layer:
The final Conv1D layer with a linear activation is used to reconstruct the original input dimensions allowing for the reconstruction of the solar flux data. 


*Paul Hwang*

- Data Preprocessing
- Time Series Model Building

*David Nezelek*
- 


## Project Summary

Using the GOES dataset (https://www.ngdc.noaa.gov/stp/satellite/goes-r.html), we preprocessed the data to grab time, fluxes (XRSA and XRSB), and instances of solar flares and used different time series models (LSTM, Conv1D, and mixture) and an autoencoder to detect instances of solar flares.


## About this Repository

This repository is broken up into folders based on focus.

*Data EDA* - Files focused on analyzing and visualing data.

*Data Preprocessing* - Files focused on preprocessing data.

*Models and Analysis* - Files focused on running different models and analysis.

*Data* - Link to google drive: https://drive.google.com/drive/folders/1s-yTlgDNXA3VpdiMw6VrwFG0WDTREnb_?usp=drive_link

*Presentations* - Link to google drive: https://drive.google.com/drive/folders/1Ct5E5UBtPsyY8oXjNH1aFva_iZNr5AU5?usp=drive_link
