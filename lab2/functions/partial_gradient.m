function [wgrad, w0grad] = partial_gradient(w, w0, example_train, label_train)
% Function Name: partial_gradient
%
% Description: This function computes the derivatives of the partial loss L_i
%              with respect to each of the classifier parameters. It takes in
%              a weight vector w, a bias scalar w0, a training example example_train,
%              and its corresponding label label_train.
%
% Inputs:
%     - w: a weight vector
%     - w0: a scalar bias value
%     - example_train: a vector representing a training example
%     - label_train: a scalar representing the label of the training example
%
% Outputs:
%     - wgrad: a vector of the same size as w containing the gradient of L_i with
%              respect to each element of w
%     - w0grad: a scalar containing the gradient of L_i with respect to w0
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: train_logistic, evaluate_logistic

y = dot(example_train, w) + w0;

p = exp(y) / (1 + exp(y));

if label_train == 1

    wgrad = (p - 1) * example_train;
    w0grad = (p - 1);

else

    wgrad = p * example_train;
    w0grad = p;

end


end
