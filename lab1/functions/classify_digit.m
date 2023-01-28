function label = classify_digit(digit_img, position, radius, digits_training)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


desc = gradient_descriptor(digit_img, position, radius);

best_idx = 1;

best_dist = inf;

N = length(digits_training);

for i=1:N

    dist = norm(desc - digits_training(i).descriptor);
    if dist<best_dist
        best_dist = dist;
        best_idx = i;
    end
end

label = digits_training(best_idx).label;

end