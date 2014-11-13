function [alpha beta] = m_step_linear_regression(X, Y, m, s)
% M-step of EM algorithm
%
% @param X      : design matrix for regression (n x d, includes intercept)
% @param Y      : target vector
% @param m      : mean of weight vector
% @param s      : covariance matrix of weight vector
%
% @return alpha : weight precision = 1/(weight variance)
% @return beta  : noise precision = 1 / (noise variance)

[n,d] = size(X);

%% trace(s) = sum_i=1^d 1/(lambda_i + alpha) = sum of eigenvalues of s
alpha = d/(m'*m + trace(s));

%% d(ln |A|)/d(beta) = sum_i=1^d gamma_i/(lambda_i + alpha),
%% where, gamma_i = eig(X'X).
%% since, 1/(lambda_i+alpha) = eig(s = beta*X'X + alpha*I)
%%        ==> eig(s*X'X) = gamma_i/(lambda_i+alpha)
beta  = n/(norm(Y-X*m)^2 + trace(s*(X'*X)));
