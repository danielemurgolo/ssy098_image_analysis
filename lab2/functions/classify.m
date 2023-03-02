function predicted_labels = classify(examples_val, w, w0)
% Function Name: classify
%
% Description: This function applies the logistic regression classifier to the
%              example data.
%
% Inputs:
%     - examples_val: a cell array containing validation examples
%     - w: a weight vector
%     - w0: a scalar bias value
%
% Outputs:
%     - predicted_labels: a vector containing the predicted labels for each example
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: train_logistic, evaluate_logistic
N = size(examples_val, 2);

predicted_labels = zeros(1, N);

for i = 1:N

    y = dot(cell2mat(examples_val(i)), w) + w0;

    p = exp(y) / (1 + exp(y));

    if p > 0.5

        predicted_labels(i) = 1;

    else
        predicted_labels(i) = 0;

    end

end

end