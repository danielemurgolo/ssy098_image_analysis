tp = 0;

N = length(digits_validation);

for i=1:N

    center = ceil(size(digits_validation(i).image)/2);
    radius = floor((min(center)-1)/3);
    label = classify_digit(digits_validation(i).image, center, radius, digits_training);
    if label == digits_validation(i).label
        tp = tp + 1;
    end
    
end

disp(['Accuracy ' num2str((tp/N)*100) '%']);