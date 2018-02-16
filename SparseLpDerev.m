function [D,Dtot] = SparseLpDerev(X,paramderev)
% DEREVERBERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = size(X,1);  % number of frequency bins
N = size(X,2);  % number of time frames
M = size(X,3);  % number of channels (mics)


% estimated desired speech signal (MIMO)
D = zeros(K,N,M);

% final output
Dtot  = zeros(K,N);

% length of the prediction filter for each channel
if numel(paramderev.Lg) == 1
    Lg = paramderev.Lg * ones(K,1);
elseif numel(paramderev.Lg) ~= K
    error('Parameter Lg should consist of 1 or K elements.')
end

% main loop for each FFT bin
for k=1:K 
    
    fprintf('k = %d/%d  \r', k, K);
    
    % initialize convolution matrices
    if strcmp(paramderev.WINDOWING, 'pre-window')
        % PRE-WINDOWED CASE 
        Xref = squeeze( X(k,1:N,1:M) );
        XX   = zeros(N, M*Lg(k));
        
        for m=1:M    
            tmp = convmtx( [ zeros( paramderev.tau, 1 ) ; squeeze(X(k,:,m)).' ] , Lg(k) );
            tmp = tmp(1:N,:);
            XX( 1:N , (m-1)*Lg(k)+1 : m*Lg(k) ) = tmp;
        end
        
    elseif strcmp(paramderev.WINDOWING, 'autocorrelation')
        % AUTOCORRELATION MATRIX
        Xref = [squeeze( X(k,1:N,1:M) ); zeros(Lg(k)+1, M) ];
        XX   = zeros(N+Lg(k)+1, M*Lg(k));
        
        for m=1:M
            tmp = convmtx( [ zeros( paramderev.tau, 1 ) ; squeeze(X(k,:,m)).' ] , Lg(k) );
            XX( : , (m-1)*Lg(k)+1 : m*Lg(k) ) = tmp;
        end
        
    else
        error('Unknown windowing type')
    end
    
    % initial estimate of the dereverberated signal
    Dk = Xref;
    
    
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% LEAST ABS SUM ABS %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if strcmp(paramderev.METHOD, 'LeastAbsSumCVX') 

            cvx_begin quiet
            
            if isreal(XX)
                variable Gk(Lg(k)*M, M)
            else
                variable Gk(Lg(k)*M, M) complex
            end
            
               minimize(  sum(sum(abs(XX*Gk-Xref)))) 
            
            cvx_end

            Dk = Xref - XX * Gk;


    elseif strcmp(paramderev.METHOD, 'LeastAbsSumCVXreg')   
        
            cvx_begin quiet
            if isreal(XX)
                variable Gk(Lg(k)*M, M)
            else
                variable Gk(Lg(k)*M, M) complex
            end
            
              minimize(  sum(sum(abs(XX*Gk-Xref))) + paramderev.alphaREG *  sum(sum(abs(Gk))) )
            cvx_end

            
            Dk = Xref - XX * Gk;

            
    elseif strcmp(paramderev.METHOD, 'LeastAbsSumADMM')
        
        Gk = zeros(Lg(k)*M, M);

        for m = 1:M
            Gk(:, m) = lad_admm(XX, Xref(:, m), zeros(size(XX, 1),1), 2, 1000, 1.7);
        end
        
        Dk = Xref - XX * Gk;
        
    elseif strcmp(paramderev.METHOD, 'LeastAbsSumADMMreg')
        
        Gk = zeros(Lg(k)*M, M);

        for m = 1:M
            Gk(:, m) = lad_reg_admm(XX, Xref(:, m), 0, paramderev.alphaREG, 2, 1000, 1.7);
        end
        
        Dk = Xref - XX * Gk;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% GROUP LASSO %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
    elseif strcmp(paramderev.METHOD, 'GroupLassoCVX') 
        
        cvx_begin quiet
        
            if isreal(XX)
                variable Gk(Lg(k)*M, M)
            else
                variable Gk(Lg(k)*M, M) complex
            end
            
            minimize( sum(norms(XX*Gk - Xref, 2, 2))  ) 
        
        cvx_end
        
        Dk = Xref - XX * Gk;
          
     elseif strcmp(paramderev.METHOD, 'GroupLassoCVXreg') 
        
        cvx_begin quiet
        
            if isreal(XX)
                variable Gk(Lg(k)*M, M)
            else
                variable Gk(Lg(k)*M, M) complex
            end
            
           minimize( sum(norms(XX*Gk - Xref, 2, 2))  + paramderev.alphaREG *   sum(norms(Gk, 2, 2)) ) 
        
        cvx_end
        
        Dk = Xref - XX * Gk;
            
    elseif strcmp(paramderev.METHOD, 'GroupLassoADMM') 
        
        Gk = gl_admm(XX, Xref, zeros(size(Xref)), 50, 1000, 1.7);
        
        Dk = Xref - XX * Gk;
        
    elseif strcmp(paramderev.METHOD, 'GroupLassoADMMreg') 
        
        Gk = gl_reg_admm(XX, Xref, zeros(size(Xref)), paramderev.alphaREG, 2, 1000, 1.7);

        Dk = Xref - XX * Gk;
   
    else
        
        error('Unknown model order selection method')
    
    end
    

    % save the result
    D(k,:,:) = Dk(1:N, :);
    
    % final results
    Dtot(k, :) = mean(Dk(1:N, :), 2);
    
end
fprintf('\n')
    