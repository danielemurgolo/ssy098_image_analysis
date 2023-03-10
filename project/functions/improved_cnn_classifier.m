function layers = improved_cnn_classifier()
% Function Name: basic_cnn_classifier
%
% Description:
%   This function defines a basic convolutional neural network (CNN)
%   architecture for image classification on the CIFAR10 dataset. The CNN
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
    imageInputLayer([32, 32, 3],"Normalization","zscore",Mean=0,StandardDeviation=1); ...
    convolution2dLayer(3, 32, "Padding", "same", "WeightsInitializer", "he"); ...
    reluLayer(); ...
    batchNormalizationLayer();...
    dropoutLayer(0.2);...
    convolution2dLayer(3, 32, "Padding", "same", "WeightsInitializer", "he"); ...
    reluLayer(); ...
    batchNormalizationLayer(); ...
    dropoutLayer(0.3);...
    maxPooling2dLayer(2); ...
    fullyConnectedLayer(128); ...
    reluLayer(); ...
    dropoutLayer(0.4);...
    fullyConnectedLayer(10); ...
    softmaxLayer(); ...
    classificationLayer(); ...
    ];
end