function [lp] = log_predictive_collapsed_gmm(x, K, N, Nk, X_bar_k, Sk, alpha, beta, Lambda_0, nu)
% Compute the (unnormalized) log predictive for a single class; that is,
% 
%   p(z_new=k | data, alpha, beta, Lambda_0, nu)
%
% where the dirichlet prior pi, and the cluster means and precisions mu_k
% and Lambda_k have been marginalized out analytically.
%
% x: new data point, D x 1 vector
% K: (maximum) number of classes
% N: total number of observations
% Nk: number of observations within this class (not including x)
% X_bar_k: mean value of observations within this class (not including x)
% Sk: empirical covariance matrix of observations within this class (not including x)
% alpha: prior parameter for dirichlet pi
% beta: hyperparameter on precision of cluster means
% Lambda_0: DxD matrix hyperparameter for wishart
% nu: degrees of freedom hyperparameter for wishart

log_dirichlet = log_predictive_dirichlet(Nk,N,K,alpha);
log_mvt       = log_predictive_mvt(x, Nk, X_bar_k, Sk, beta, Lambda_0, nu);
lp = log_dirichlet + log_mvt;
