function desc = gradient_descriptor(image, position, radius)
%GRADIENT_DESCRIPTOR Summary of this function goes here
%   Detailed explanation goes here
std = radius * 0.1;
[grad_x, grad_y] = gaussian_gradients(image, std);
desc = [];

centres = place_regions(position, radius);

for c=centres

    
    x_patch = get_patch(grad_x, c(1), c(2), radius);
    y_patch = get_patch(grad_y, c(1), c(2), radius);
    
    hist = gradient_histogram(x_patch, y_patch);
    
    desc = [desc; hist];

end

desc = desc(:);
desc = desc/norm(desc);

end

