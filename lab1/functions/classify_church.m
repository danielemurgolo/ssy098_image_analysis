function label = classify_church(image, feature_collection)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

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