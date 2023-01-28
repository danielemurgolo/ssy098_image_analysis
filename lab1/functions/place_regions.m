function region_centres = place_regions(centre,radius)
%PLACE_REGIONS Summary of this function goes here
%   Detailed explanation goes here
region_centres = zeros(2,9);
idx = 1;
for i = -1:1

    for j = -1:1
    
        x = centre(1) + i*2*radius;
        y = centre(2) + j*2*radius;

        region_centres(:,idx) = [x;y];
        idx = idx + 1;

    end

end
%region_centres = region_centres';
end

