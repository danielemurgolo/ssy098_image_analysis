function patch = get_patch(image, x, y, patch_radius)
%GET_PATCH Summary of this function goes here
%   Detailed explanation goes here

x_min = x - patch_radius;
x_max = x + patch_radius;

y_min = y - patch_radius;
y_max = y + patch_radius;

if (x_min<=0) || (y_min <= 0) || (x_max > size(image,2)) || (y_max > size(image, 1))

    error('Patch outside image border')

end

patch = image(y_min:y_max, x_min:x_max, :);

end

