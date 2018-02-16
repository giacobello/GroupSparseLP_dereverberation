%
% Example of solving a specific block Toeplitz system
%
% This implementation follows the algorithm outlined in
%
%  Efficient Solution of a Toeplitz-plus-Hankel Coefficient
%   Matrix System of Equations
%  Merchant and Parks
%  IEEE-AASP 30, 1982, pp.40--44
%
%

clear all

load('DeRev_Example.mat')

M = size(Xref, 2);
LgM = size(XX, 2);

HH = XX'*XX;
H = HH;

m = M;
B = randn(size(H, 1), m) + 1j*randn(size(H, 1), m);

Xs = H\B;

% Hermitian block Levinson algorithm


% Interleaver
s = M;

n = size(H, 1)/M;

q = (1:s:s*n)';
qq = [q; q+1; q+2; q+3];
I = eye(s*n); Q = I(:, qq);

BB = Q*B;
A = Q*H*Q';

Rij = @(i,j) A(s*i+1:s*(i+1), s*j+1:s*(j+1));


X = eye(s);
Y = eye(s);

P = Rij(0, 0)\BB(1:s,:);
p = P;

Vx = Rij(0, 0);
Vy = Vx;

for i=1:n-1

    %Ex = zeros(s, s);
    %for j=0:i-1
    %    Ex = Ex + Rij(i, j)*X(s*j+1:s*(j+1), :);
    %end
    Ex = A(i*s+1:s*(i+1), 1:i*s) * X(1:i*s, :);

    %Ey = zeros(s, s);
    %for j=1:i
    %    Ey = Ey + Rij(0, j)*Y(s*(i-j)+1:s*((i-j)+1), :);
    %end
    Ey = Ex';
    
    %ep = zeros(s, 1);
    %for j=0:i-1
    %    ep = ep + Rij(i, j)*p(j*s+1:((j+1)*s));
    %end
    ep = A(i*s+1:s*(i+1), 1:i*s) * p(1:i*s, :);
    
    Bx = Vy\Ex;
    By = Vx\Ey;

    Xp = [X(1:s*i, :); zeros(s, s)] ...
        - [zeros(s, s); bflipud(s, Y(1:s*i, :));] *Bx;

    Yp = [zeros(s, s); bflipud(s, Y(1:s*i, :));] ...
        - [X(1:s*i, :); zeros(s, s)]*By;

    X = Xp;
    Y = bflipud(s, Yp);
    
    Vx = Vx - Ey*Bx;
    Vy = Vy - Ex*By;

    g = Vy\(BB(i*s+1:(i+1)*s, :) - ep);
    
    p = [p(1:s*i, :); zeros(s, m)] ...
         + [bflipud(s, Y(1:(i+1)*s, :));]*g;
end

Xsp = Q'*p;
