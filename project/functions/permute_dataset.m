function [permutedImageData, categoricalLabels] = permute_dataset(data, labels)
% Function Name: permute_dataset
%
% Description: This function reshapes the input data so that the last
%              dimension of the output shape corresponds to the index.
%              Additionally, it converts the input labels to categorical format.
%
% Inputs:
%   - data: matrix of size N x 3072
%   - labels: array of length N
%
% Outputs:
%   - permutedImageData: matrix of size 32 x 32 x 3 x N
%   - categoricalLabels: categorical array of length N
%
% Author: Daniele Murgolo
% Date: March 6th, 2023

N = size(data, 1);
data = data';
permutedImageData = reshape(data, [32, 32, 3, N]);
categoricalLabels = categorical(labels);
end