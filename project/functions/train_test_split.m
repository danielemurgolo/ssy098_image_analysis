function [X_train, X_test, Y_train, Y_test] = train_test_split(data, labels, test_size)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = length(labels);
partition = cvpartition(N, "HoldOut", test_size);

X_train = data(:,:,:,partition.training);
Y_train = labels(partition.training);

X_test = data(:,:,:,partition.test);
Y_test = labels(partition.test);

end