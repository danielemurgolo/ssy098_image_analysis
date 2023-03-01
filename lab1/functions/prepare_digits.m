%image size is 39x39 so
mid = ceil(39/2);

center = [mid, mid];

%since the center is (19,19) and the radius to not create an error in
% get_patch should be <10
radius = 6;

N = length(digits_training);

for i = 1:N

    image = digits_training(i).image;
    digits_training(i).descriptor = gradient_descriptor(image, center, radius);

end