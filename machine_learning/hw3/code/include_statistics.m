function [N_n, mu_n, S_n] = include_statistics(x,k_x, N, mu, S)
% Include the statistics of a new data-point x in the class-statistics
% of class k_x, given the class-statistics of all classes -- N,mu,S:
%
% x : [dx1] data-point
% k_x: class of this data-point
% N: [K x 1] number of data-points in each class
% mu: [d x K] mean of each class
% S:  cell of [d x d] matrices -- covariance of each class

mu(:,k_x) = (mu(:,k_x)*N(k_x) + x)/(N(k_x)+1);
S{k_x}    = (S{k_x}*N(k_x) + (x'-mu(:,k_x))*(x'-mu(:,k_x))')/(N(k_x)+1);
N(k_x)    = N(k_x)+1;

