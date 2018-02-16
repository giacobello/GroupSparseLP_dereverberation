clear; close all; clc;

load Xdata

METHODS = {'LeastAbsSumCVX','LeastAbsSumCVXreg','LeastAbsSumADMM','LeastAbsSumADMMreg',...
           'GroupLassoCVX','GroupLassoCVXreg','GroupLassoADMM','GroupLassoADMMreg'}; 

                        
%%%%%%%%%%% PARAMETERS DEREV %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

paramderev.tau            = 2;             % delay for prediction 
paramderev.Lg             = 10;            % predictor order


% IRLS only
paramderev.IRLS.myeps     = 1e-8;          % to avoid division by 0 in weighting calcualtion
paramderev.IRLS.itMax     = 5;             % iterations of IRLS 
paramderev.IRLS.itTol     = 1e-6;          % IRLS convergence check 
paramderev.IRLS.p         = 0;             % p-norm to minimize

paramderev.alphaREG       = 0.1;           % used for regularized type of optimization


% methods to use
% METHODS = {'IRLS','IRLSjoint','GLjoint','FLSAjoint'};
WINDOWING = {'autocorrelation','pre-window'};
% choose method here
paramderev.METHOD = METHODS{5};
paramderev.WINDOWING = WINDOWING{1};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% call the main function (D is the mimo output, Dtot is the average single channel)
[D,Dtot] = SparseLpDerev(X,paramderev);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




figure;

subplot(2,1,1);
imagesc(db(abs(X(:,:,1))), [-180 50])
axis xy; axis tight; colormap(jet); view(0,90); axis off

subplot(2,1,2);
imagesc(db(abs(D(:,:,1))), [-180 50])
axis xy; axis tight; colormap(jet); view(0,90); axis off

