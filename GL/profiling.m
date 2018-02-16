
load('DeRev_Example.mat')

M = size(Xref, 2);
LgM = size(XX, 2);
    
Gkp = zeros(LgM, M);


profile on
Gkp = gl_admm(XX, Xref, zeros(size(Xref)), 1.7, 1000);
profile off


