function label = classify_digit(digit_img, position, radius, digits_training)
% Function Name: classify_digit
%
% Description: This function computes a descriptor for the given digit image,
%              goes through all the digits in digits_training to find the one
%              with the most similar descriptor and outputs the label of that digit.
%
% Inputs:
%     - digit_img: a grayscale image of a single digit
%     - position: a 2-element vector representing the center of the region of interest
%     - radius: the radius of the regions used to compute the descriptor
%     - digits_training: an array of structs containing training digit images and their labels
%
% Outputs:
%     - label: the predicted label of the input digit image
%
% Example Usage:
%     >> img = imread('digit_0.png');
%     >> digits_training = load('digits_training.mat').digits_training;
%     >> label = classify_digit(img, [20, 20], 8, digits_training);
%     >> disp(label);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: gradient_descriptor, place_regions, gaussian_gradients,
%           gradient_histogram, get_patch


desc = gradient_descriptor(digit_img, position, radius);

best_idx = 1;

best_dist = inf;

N = length(digits_training);

for i = 1:N

    dist = norm(desc-digits_training(i).descriptor);
    if dist < best_dist
        best_dist = dist;
        best_idx = i;
    end
end

label = digits_training(best_idx).label;

end