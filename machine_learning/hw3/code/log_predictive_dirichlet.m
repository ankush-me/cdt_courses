function [lp] = log_predictive_dirichlet(Nk,N,K,alpha)
% Computes p(z_i=k | z{\i} (all z's except z_i))
%
% @param
% K: (maximum) number of classes
% N: total number of observations
% Nk: number of observations within this class (not including x)
% alpha: prior parameter for dirichlet pi
%
% @return
% lp = log( p(z_i=k | z{\i} )) %% log conditional probability
lp = log(alpha + Nk) - log(K*alpha + N);
