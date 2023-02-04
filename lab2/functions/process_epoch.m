function [w, w0] = process_epoch(w, w0, lrate, examples_train, labels_train)
%PROCESS_EPOCH Summary of this function goes here
%   Detailed explanation goes here

idx = randperm(size(examples_train,2));

% idx = 1:size(labels_train,2);

for i=idx
    [wgrad,w0grad] = partial_gradient(w,w0,cell2mat(examples_train(i)), labels_train(i));

    w = w - lrate*wgrad;

    w0 = w0 - lrate*w0grad;

end

