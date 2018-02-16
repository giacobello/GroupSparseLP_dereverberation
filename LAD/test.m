function tests = test()
%
% Test script for evaluating the LAD dereverberation formulation
%
%  Can be executed as runtests('test')
%
% Author:
%    T. L. Jensen,  tlj@es.aau.dk    
%    Aalborg University, 2017
%      

% Seeding may be used to make debugging easier.
    rng(100);

    tests = functiontests(localfunctions);
end    

function test_S_real(testCase)
% Testing of softhresholding
    n = 100;
    x = randn(n, 1);
    t = 1.1;

    %% Case for prox_t||u||_1 (x)
    cvx_begin
    cvx_quiet(true)
    variable u(n)
    minimize t*norm(u, 1) + 0.5*square_pos(norm(u - x, 2))
    cvx_end

    uh = Soft(x, t);

    testCase.assertEqual(u, uh, 'absTol', 1e-4)
end

function test_S_complex(testCase)
% Testing of softhresholding
    n = 100;
    x = randn(n, 1) + 1j*randn(n, 1);
    t = 1.1;

    %% Case for prox_t||u||_1 (x)
    cvx_begin
    cvx_quiet(true)
    variable u(n) complex
    minimize t*norm(u, 1) + 0.5*square_pos(norm(u - x, 2))
    cvx_end

    uh = Soft(x, t);

    testCase.assertEqual(u, uh, 'absTol', 1e-4)
end

function test_random_real(testCase)

    m = 100;
    n = 30;
    
    C = randn(m, n);
    b = randn(m, 1);
    
    cvx_begin
    cvx_quiet(true)
    variable x(n)
    minimize norm(C*x - b, 1)
    cvx_end
   
    xp = lad_admm(C, b, C\b, 1.7, 1000);
    
    testCase.assertEqual(x, xp, 'absTol', 1e-3)
end

function test_random_complex(testCase)

    m = 100;
    n = 30;
    
    C = randn(m, n) + 1j*randn(m, n);
    b = randn(m, 1) + 1j*randn(m, 1);
    
    cvx_begin
    cvx_quiet(true)
    variable x(n) complex
    minimize norm(C*x - b, 1)
    cvx_end
   
    xp = lad_admm(C, b, C\b, 2, 1000, 1.7);
    
    testCase.assertEqual(x, xp, 'absTol', 1e-4)
end

function test_dereverb_complex_case(testCase)

    load('DeRev_LSA_Example.mat')

    M = size(Xref, 2);
    LgM = size(XX, 2);
    cvx_begin quiet
    variable Gk(LgM, M) complex
    minimize(  sum(sum(abs(XX*Gk - Xref))) )
    cvx_end
    
    Gkp = zeros(size(Gk));

    for m = 1:M
        Gkp(:, m) = lad_admm(XX, Xref(:, m), XX\Xref(:, m), 2, ...
                             1000, 1.7);
    end

    f = @(G) sum(sum(abs(XX*G - Xref)));

    testCase.assertEqual(f(Gk), f(Gkp), 'relTol', 1e-5)
    
    testCase.assertEqual(Gk, Gkp, 'relTol', 5e-2)

end

function test_random_real_reg(testCase)

    m = 100;
    n = 30;
    
    C = randn(m, n);
    b = randn(m, 1);
    
    alpha = 1.2;
    cvx_begin
    cvx_quiet(true)
    variable x(n)
    minimize norm(C*x - b, 1) + alpha*norm(x, 1)
    cvx_end
   
    xp = lad_reg_admm(C, b, C\b, alpha, 2, 3000);
    
    testCase.assertEqual(x, xp, 'absTol', 1e-3)
end

function test_random_complex_ref(testCase)

    m = 100;
    n = 30;
    
    C = randn(m, n) + 1j*randn(m, n);
    b = randn(m, 1) + 1j*randn(m, 1);
    
    alpha = 3.2;
    
    cvx_begin
    cvx_quiet(true)
    variable x(n) complex
    minimize norm(C*x - b, 1) + alpha*norm(x, 1)
    cvx_end
   
    xp = lad_reg_admm(C, b, C\b, alpha, 2, 1000, 1.7);
    
    testCase.assertEqual(x, xp, 'absTol', 1e-4)
end

function test_dereverb_complex_case_reg(testCase)

    load('DeRev_LSA_Example.mat')

    M = size(Xref, 2);
    LgM = size(XX, 2);
    alpha = 2.2;
    cvx_begin quiet
    variable Gk(LgM, M) complex
    minimize(  sum(sum(abs(XX*Gk - Xref))) + alpha * ...
               sum(sum(abs(Gk))) )
    cvx_end
    
    Gkp = zeros(size(Gk));

    for m = 1:M
        [Gkp(:, m), info] = lad_reg_admm(XX, Xref(:, m), zeros(LgM, 1), alpha, 2, ...
                             1000, 1.7);
    end

    f = @(G) sum(sum(abs(XX*G - Xref))) + alpha*sum(sum(abs(G)));

    testCase.assertEqual(f(Gk), f(Gkp), 'relTol', 1e-5)
    
    testCase.assertEqual(Gk, Gkp, 'absTol', 5e-2)

end

