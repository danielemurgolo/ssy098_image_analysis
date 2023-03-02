function [w, w0] = process_epoch(w, w0, lrate, examples_train, labels_train)
% Function Name: process_epoch
%
% Description: This function performs one epoch of stochastic gradient descent.
%              At each iteration of stochastic gradient descent, a training example, i,
%              is chosen at random. For this example the gradient of the partial loss, L_i,
%              is computed and the parameters are updated according to this gradient.
%              The most common way to introduce the randomness is to make a random
%              reordering of the data and then going through it in the new order.
%
% Inputs:
%     - w: a weight vector
%     - w0: a scalar bias value
%     - lrate: the learning rate
%     - examples_train: a cell array of training examples
%     - labels_train: a vector of training labels
%
% Outputs:
%     - w: the updated weight vector after one epoch
%     - w0: the updated bias scalar after one epoch
%
% Author: Your Name
% Date: March 1st, 2023

idx = randperm(size(examples_train, 2));

% idx = 1:size(labels_train,2);

for i = idx
    [wgrad, w0grad] = partial_gradient(w, w0, cell2mat(examples_train(i)), labels_train(i));

    w = w - lrate * wgrad;

    w0 = w0 - lrate * w0grad;

end
