function all_residuals = compute_residuals(Ps, us, U)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    N_cameras = length(Ps);
    Us = [U;1];
    all_residuals = zeros(2*N_cameras,1);
    
    for i=1:2:2*N_cameras
  
        P = Ps{round(i/2)};
        a = P(1,:);
        b = P(2,:);
        c = P(3,:);
        positive = check_depths(Ps, U);

        if positive

            x = (a*Us)/(c*Us) - us(1,round(i/2));
            y = (b*Us)/(c*Us) - us(2,round(i/2));
            all_residuals(i) = x;
            all_residuals(i+1) = y;
            
        else

            all_residuals(i) = inf;
            all_residuals(i+1) = inf;

        end

    end

end