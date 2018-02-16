function Y = SV(X, t)
% 2-norm
nX = sqrt(sum(abs(X).^2, 2));
mx = max(1- t./nX, 0);
Y = bsxfun(@times, X, mx); % Y = diag(mx)*X;
