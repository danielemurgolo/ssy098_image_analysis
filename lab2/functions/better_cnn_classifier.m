function layers = better_cnn_classifier()
% Function Name: better_cnn_classifier
%
% Description:
%   This function defines an improved convolutional neural network (CNN)
%   architecture for image classification on the MNIST dataset. The CNN
%   consists of an image input layer, a 2D convolutional layer with 25
%   filters of size 3x3, a ReLU activation layer, a 2D max pooling layer
%   with stride 2, another 2D convolutional layer with 20 filters of size
%   5x5, another ReLU activation layer, another 2D max pooling layer with
%   stride 2, two fully connected layers with 50 and 10 neurons,
%   respectively, a softmax layer, and a classification output layer.
%
% Inputs:
%   None
%
% Output:
%   layers - a layer array defining the architecture of the CNN
%
% Example Usage:
%   % Define the improved CNN classifier architecture
%   layers = better_cnn_classifier();
%
% Author: Daniele Murgolo
% Date: March 1st, 2023


layers = [; ...
    imageInputLayer([28, 28, 1]); ...
    convolution2dLayer(3, 25); ...
    reluLayer(); ...
    maxPooling2dLayer(2, 'Stride', 2); ...
    convolution2dLayer(5, 20); ...
    reluLayer(); ...
    maxPooling2dLayer(2, 'Stride', 2); ...
    fullyConnectedLayer(50); ...
    fullyConnectedLayer(10); ...
    softmaxLayer(); ...
    classificationLayer(); ...
    ];
end