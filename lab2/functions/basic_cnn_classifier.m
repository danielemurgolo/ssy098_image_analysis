function layers = basic_cnn_classifier()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
layers = [
    imageInputLayer([28,28,1]);
    convolution2dLayer(3,25);
    reluLayer();
    maxPooling2dLayer(2, 'Stride',2);
    fullyConnectedLayer(10);
    softmaxLayer();
    classificationLayer()
];
end