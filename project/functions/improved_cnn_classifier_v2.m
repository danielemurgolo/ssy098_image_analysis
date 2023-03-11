function layers = improved_cnn_classifier_v2()
%BASIC_CNN_CLASSIFIERV2 Summary of this function goes here
%   Detailed explanation goes here
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

batchNormalizationLayer("Epsilon",numFilters);

maxPooling2dLayer(3, 'Stride', 2);

convolution2dLayer(filterSize, numFilters, 'Padding', 2);
reluLayer();
maxPooling2dLayer(3, 'Stride',2);
dropoutLayer(0.1);

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

