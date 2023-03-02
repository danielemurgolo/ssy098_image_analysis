function [examples_train_aug, labels_train_aug] = augment_data(examples_train, labels_train, M)
% Function Name: augment_data
%
% Description: This function takes each sample of the original training data
%              and applies M random rotations using the imrotate function, 
%              from which result M new examples. The new examples are stored
%              in examples_train_aug and their corresponding labels are stored 
%              in labels_train_aug.
%
% Inputs:
%     - examples_train: a cell array containing the original training data samples
%     - labels_train: a vector containing the corresponding labels for the original training data samples
%     - M: the number of random rotations to apply to each sample
%
% Outputs:
%     - examples_train_aug: a cell array containing the augmented training data samples
%     - labels_train_aug: a vector containing the corresponding labels for the augmented training data samples
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: train_logistic, evaluate_logistic
examples_train_aug = {};
labels_train_aug = [];

N = size(examples_train, 2);

for i = 1:N

    for j = 1:M

        angle = 90 * randi(3);
        img = imrotate(cell2mat(examples_train(i)), angle, 'bilinear', 'crop');
        idx = i * M + j - M;
        examples_train_aug{idx} = img;
        labels_train_aug(idx) = [labels_train(i)];

    end

end
