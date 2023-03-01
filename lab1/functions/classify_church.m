function label = classify_church(image, feature_collection)
% Function Name: classify_church
%
% Description: This function classifies a new image by computing feature points
%              for the image, matching them to the features in the feature
%              collection, and letting each match vote for the correct church.
%              The output is the label of the most likely church.
%
% Inputs:
%   - image: The new image to be classified
%   - feature_collection: A structure containing a set of features and descriptors
%       for each church, as well as the labels for the churches.
%
% Outputs:
%   - label: The label of the most likely church for the new image.
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: detectSIFTFeatures, extractFeatures, matchFeatures

n_labels = length(feature_collection.names);

matches_per_label = zeros(n_labels, 1);

points = detectSIFTFeatures(image);

[features, validPoints] = extractFeatures(image, points);

for i = 1:n_labels

    desc = feature_collection.descriptors(:, feature_collection.labels == i);

    match_coords = matchFeatures(features, desc', 'MatchThreshold', 100, 'MaxRatio', 0.7);

    matches_per_label(i) = size(match_coords, 1);

end

[~, label] = max(matches_per_label);

end