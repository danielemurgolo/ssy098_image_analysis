function [X_train, X_test, Y_train, Y_test] = train_test_split(data, labels, test_size)
% Function Name: train_test_split
%
% Description: This function splits a dataset into training and testing sets
%              The input data is assumed to be a 4-dimensional matrix with
%              the dimensions [height, width, channels, samples]. The input
%              labels must be a vector with one label for each sample in the
%              data. The test_size parameter can be either a fraction
%              between 0 and 1 representing the proportion of samples to
%              include in the test set, or an integer representing the exact
%              number of samples to use for the test set.
%
% Inputs:
%   - data:           32 x 32 x 3 x N matrix of images
%   - labels:         1 x N vector of labels for each image
%   - test_size:      fraction or number of observations in test set
%
% Outputs:
%   - X_train           32 x 32 x 3 x M matrix of training images
%   - X_test            32 x 32 x 3 x (N-M) matrix of testing images
%   - Y_train           1 x M vector of training labels
%   - Y_test            1 x (N-M) vector of testing labels
%
% Example:
%   >> [X_train, X_test, Y_train, Y_test] = train_test_split(data, labels, 0.2);
%
%
% Author: Daniele Murgolo
% Date: March 7th, 2023
%
% See also: cvpartition

N = length(labels);
partition = cvpartition(N, "HoldOut", test_size);

X_train = data(:, :, :, partition.training);
Y_train = labels(partition.training);

X_test = data(:, :, :, partition.test);
Y_test = labels(partition.test);

end