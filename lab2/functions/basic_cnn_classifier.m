function layers = basic_cnn_classifier()
% Function Name: basic_cnn_classifier
%
% Description: 
%   This function defines a basic convolutional neural network (CNN)
%   architecture for image classification on the MNIST dataset. The CNN
%   consists of an image input layer, a 2D convolutional layer, a ReLU
%   activation layer, a 2D max pooling layer, a fully connected layer, a
%   softmax layer, and a classification output layer.
%
% Inputs:
%   None
%
% Output:
%   layers - a layer array defining the architecture of the CNN
%
%
% Example Usage:
%   % Define the basic CNN classifier architecture
%   layers = basic_cnn_classifier();
%
% Author: Daniele Murgolo
% Date: March 1st, 2023


layers = [; ...
    imageInputLayer([28, 28, 1]); ...
    convolution2dLayer(3, 25); ...
    reluLayer(); ...
    maxPooling2dLayer(2, 'Stride', 2); ...
    fullyConnectedLayer(10); ...
    softmaxLayer(); ...
    classificationLayer(); ...
    ];
end