function histogram = gradient_histogram(grad_x,grad_y)
%GRADIENT_HISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
bins = 8;
histogram = 1:bins;
angles = atan2(grad_y, grad_x);

for n = 1:bins

    low = (n-1)*2*pi / bins - pi ; 
    high = n * 2 * pi / bins - pi;

    idx = angles>=low & angles < high;

    gx = grad_x(idx);
    gy = grad_y(idx);

    if isempty(gx)
        gx = 1e-31;
    end

    if isempty(gy)
        gy = 1e-31;
    end

    magnitudes = sqrt(gx.^2 + gy.^2);

    histogram(n) = sum(magnitudes);

end


end

