function errors = reprojection_errors(Ps, us, U)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    N_cameras = length(Ps);
    positive = check_depths(Ps, U);
    errors = zeros(N_cameras,1);
    
    for i=1:N_cameras
    
        if positive(i)

            u = Ps{i}*[U;1];
            u = u(1:2)/u(3);
            errors(i) = sqrt(sum(us(:,i)-u).^2);

        else
            errors(i) = inf;

        end
    
    end
end