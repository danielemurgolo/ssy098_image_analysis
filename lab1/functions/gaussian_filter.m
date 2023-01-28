function filtered = gaussian_filter(image,std)
%GAUSSIAN_FILTER Summary of this function goes here
%   Detailed explanation goes here
filter = fspecial('gaussian', ceil(4*std), std);

filtered = imfilter(image, filter, 'symmetric');
end

