function  [wgrad, w0grad] = partial_gradient(w, w0, example_train, label_train)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

y = dot(example_train,w) + w0;

p = exp(y)/(1+exp(y));

if label_train == 1

    wgrad = (p-1) * example_train;
    w0grad = p-1;

else

    wgrad = p * example_train;
    w0grad = p;

end

end