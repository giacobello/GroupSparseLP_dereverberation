function y = S(x, t)
% 2-norm
mx = max(1- t/norm(x, 2), 0);
y = x*mx;
