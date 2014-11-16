function [N_n, mu_n, S_n] = exclude_statistics(x, N, mu, S)
% Return the excluded the statistics of a new data-point x 
% from the class-statistics of class k_x, given the old class-stats.
%
% x : [dx1] data-point
% N:  [scalar] number of data-points in this class
% mu: [d x 1] mean of this class
% S:  [d x d] matrix -- covariance of this class
%
% @return : updated N,mu,S
%

if N > 1
	S_n  = (S*N - (x-mu)*(x-mu)')/(N-1);
	mu_n = (mu*N - x)/(N-1);
	N_n  = N-1;
else
	D    = size(x,1);
	S_n  = zeros(D);
	mu_n = zeros(D,1);
	N_n  = 0;
end
