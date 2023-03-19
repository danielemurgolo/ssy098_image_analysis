function layers = basic_cnn_classifier_v2()
% Function Name: basic_cnn_classifier_v2
%
% Description: 
%   This function returns a CNN architecture with three
%   convolutional layers, three max pooling layers, and two fully connected
%   layers. The function assumes that the input images have a height and width
%   of 32 pixels, and three color channels.
%
% Inputs:
%   None
%
% Output:
%   layers - a layer array defining the architecture of the CNN
%
% Example Usage:
%      % Load the CIFAR-10 dataset
%      [trainImages, trainLabels, testImages, testLabels] = load_data(...);
%
%      % Train the network
%      layers = basic_cnn_classifier_v2();
%      options = trainingOptions('sgdm', 'MaxEpochs', 20, 'Verbose', false);
%      trainedNet = trainNetwork(trainImages, categorical(trainLabels), layers, options);
%
% Author: Daniele Murgolo
% Date: March 19th, 2023

height =32;
width =32;
numChannels = 3;
imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize);

% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [
convolution2dLayer(filterSize, numFilters, 'Padding', 2);

reluLayer();

maxPooling2dLayer(3, 'Stride', 2);

convolution2dLayer(filterSize, numFilters, 'Padding', 2);

reluLayer();

maxPooling2dLayer(3, 'Stride',2);

convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2);

reluLayer();

maxPooling2dLayer(3, 'Stride',2);
];

finalLayers = [

fullyConnectedLayer(64);

reluLayer();

fullyConnectedLayer(10);

softmaxLayer();
classificationLayer;
];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ];
layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

