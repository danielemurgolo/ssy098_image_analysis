function [data,labels] = permute_data(data,labels)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
N = size(data, 1);
data = data';
data = reshape(data, [32,32,3,N]);
labels = categorical(labels);
end