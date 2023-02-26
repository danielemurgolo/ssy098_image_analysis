function U = refine_triangulation(Ps, us, Uhat)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    n_iter = 5;
    
    for i=1:n_iter

        r = compute_residuals(Ps,us,Uhat);
        J = compute_jacobian(Ps,Uhat);
        Uhat = Uhat - ((J'*J)\J')*r;

    end

    U = Uhat;

end