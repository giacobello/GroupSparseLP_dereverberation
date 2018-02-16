function Y = bflipud(s, X)
%
% Performs a block flip upside down
%
%

p = size(X, 1)/s;
Xp = reshape(X', s, s, p);
Y = reshape(Xp(:, :, p:-1:1), s, size(X, 1))';