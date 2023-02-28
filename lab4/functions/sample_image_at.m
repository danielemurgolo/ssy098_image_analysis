function value = sample_image_at(img, position) 
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
y = round(position(1));
x = round(position(2));
h = size(img,1);
w = size(img,2);
if(y<=0) || (y>h) || isnan(y)
    value = [0;0;0];
elseif (x<=0) || (x>w) || isnan(x)
    value = [0;0;0];
else
    value = squeeze(img(y,x,:));
end