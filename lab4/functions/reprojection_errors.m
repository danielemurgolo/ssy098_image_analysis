function errors = reprojection_errors(Ps, us, U)
% The function takes N camera matrices, Ps, N image points, us,
% and a 3D point, U, and computes a vector with the reprojection errors.
% If a point has negative depth, reprojection error is set to Inf
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