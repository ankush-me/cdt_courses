function [N, mu, S] = update_statistics(x,z_old,z_new,data, N_o, mu_o, S_o)
	update_statistics(data,z_n_previous, z_n_new,N,mu,S);
% Update the statistics of a new data-point x in the class-statistics
% of class k_x, given the class-statistics of all classes -- N,mu,S:
%
% x : [dx1] data-point
% k_x: class of this data-point
% N: [K x 1] number of data-points in each class
% mu: [d x K] mean of each class
% S:  cell of [d x d] matrices -- covariance of each class



[N(z_old), mu(:,z_old), S{z_old}] = exclude_statistics(,z_old,N_o,mu_o,S_o);


mu(:,k_x) = (mu(:,k_x)*N(k_x) + x)/(N(k_x)+1);
S{k_x}    = (S{k_x}*N(k_x) + (x'-mu(:,k_x))*(x'-mu(:,k_x))')/(N(k_x)+1);
N(k_x)    = N(k_x)+1;

