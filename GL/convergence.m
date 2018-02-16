

load('DeRev_Example.mat')

M = size(Xref, 2);
LgM = size(XX, 2);

itmax = 200;
c = 50;
m = 2;
[Gkp, info0] = gl_admm(XX, Xref, zeros(size(Xref)), c, 50*itmax);

[Gkp, info1] = gl_admm(XX, Xref, zeros(size(Xref)), c, itmax);

alpha = 1.7;
[Gkp, info2] = gl_admm(XX, Xref, zeros(size(Xref)), c, ...
                       itmax, alpha);

fs = info0.fk(end);

figure(1)
clf
semilogy((info1.fk - fs)/fs)
hold on;
plot((info2.fk - fs)/fs)
legend('alpha=1.0', sprintf('overrelaxation alpha=%.2f', alpha))