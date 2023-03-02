function net = train_classifier(layers, imgs_train, labels_train, imgs_val, labels_val)
% Function Name: train_classifier
%
% Description:
%   This function trains a convolutional neural network (CNN) classifier on
%   a set of training images and labels, and validates the performance on a
%   separate set of validation images and labels.
%
% Inputs:
%   layers - a layer array defining the architecture of the CNN
%   imgs_train - a 4D array of training images (height x width x numChannels x numImages)
%   labels_train - a categorical vector of training labels
%   imgs_val - a 4D array of validation images (height x width x numChannels x numImages)
%   labels_val - a categorical vector of validation labels
%
% Output:
%   net - a trained CNN classifier
%
% Example Usage:
%   layers = [imageInputLayer([28 28 1])
%             convolution2dLayer(5,20)
%             reluLayer()
%             maxPooling2dLayer(2,'Stride',2)
%             fullyConnectedLayer(10)
%             softmaxLayer()
%             classificationLayer()];
%   net = train_classifier(layers, imgs_train, labels_train, imgs_val, labels_val);
%   % The trained CNN classifier can now be used to classify new images
%   labels_pred = net.classify(imgs_test);
%   accuracy = nnz(labels_pred == labels_test) / length(labels_test);
%   disp(['The network achieved an accuracy of: ' num2str((accuracy)*100),'%'])
%
% Author: Daniele Murgolo
% Date: March 1st, 2023

epochs = 25;
disp('Training the network')
options = trainingOptions('sgdm', 'ExecutionEnvironment', 'gpu', 'MaxEpochs', epochs, 'Verbose', false);
net = trainNetwork(imgs_train, labels_train, layers, options);
tp = nnz(net.classify(imgs_val) == labels_val);
accuracy = tp / length(labels_val);
disp(['The network achieved an accuracy of: ', num2str((accuracy)*100), '%'])
end