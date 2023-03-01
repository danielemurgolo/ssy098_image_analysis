function region_centres = place_regions(centre, radius)
% Function Name: place_regions
%
% Description: creates 3 x 3 square regions for the descriptor. The function
%              takes in a 1x2 array `centre` representing the center of the
%              region pattern, and a scalar `radius` specifying the radius
%              of each square region. The output is a 2x9 array with columns
%              being the 2D center points of the 9 regions. Use the provided
%              function plot_squares(img, region_centres, region_radius) to
%              plot your regions in an example image. Increasing the input
%              radius with a factor K should scale the whole region pattern
%              with a factor K.
%
% Inputs:
%     - centre: a 1x2 array representing the center of the region pattern
%     - radius: a scalar specifying the radius of each square region
%
% Outputs:
%     - region_centres: a 2x9 array with columns being the 2D center points of the 9 regions
%
% Example Usage:
%     >> centre = [50, 50];
%     >> radius = 10;
%     >> regions = place_regions(centre, radius);
%     >> img = imread('example.jpg');
%     >> plot_squares(img, regions, radius);
%
% Author: Daniele Murgolo
%
% Date: March 1st, 2023
%
% See also: plot_squares

region_centres = zeros(2, 9);
idx = 1;
for i = -1:1

    for j = -1:1

        x = centre(1) + i * 2 * radius;
        y = centre(2) + j * 2 * radius;

        region_centres(:, idx) = [x; y];
        idx = idx + 1;

    end

end
%region_centres = region_centres';
end
