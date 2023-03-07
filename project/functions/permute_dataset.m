function [permutedImageData, categoricalLabels] = permute_dataset(data, labels)
% Function Name: permute_dataset
%
% Description: This function reshapes the input data so that the last
%              dimension of the output shape corresponds to the index.
%              Additionally, it converts the input labels to categorical format.
%
% Inputs:
%   - data: matrix of type uint8 of size N x 3072 representing the images in the dataset
%   - labels: array of length N representing the labels of the images
%
% Outputs:
%   - permutedImageData: matrix of type double of size 32 x 32 x 3 x N
%   - categoricalLabels: categorical array of length N
%
% Example Usage:
%   >> load(data_batch_1.mat);
%   >> [permutedImageData, categoricalLabels] = permute_dataset(data, labels);
%
% Author: Daniele Murgolo
% Date: March 6th, 2023

N = size(data, 1);
data = reshape(data, [N, 32, 32, 3]);
permutedImageData = im2double(permute(data, [3, 2, 4, 1]));
categoricalLabels = categorical(labels);
end