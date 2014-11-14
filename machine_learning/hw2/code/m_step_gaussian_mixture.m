function [mu,sigma,Pi] = m_step_gaussian_mixture(data,gamma)
% Performs the M-step of the EM algorithm for gaussain mixture model.
%
% @param data   : n x d matrix with rows as d dimensional data points
% @param gamma  : n x k matrix of resposibilities
%
% @return mu    : d x k matrix of maximized cluster centers
% @return sigma : cell array of maximized 
%

[n,d] = size(data);
[n,k] = size(gamma);

%% soft-number of points per class:
N_k = sum(gamma, 1);

%% class means:
mu = (data'*gamma)./repmat(N_k, d, 1); %% (d x k)

%% class variances:
sigma = {};
for i=1:k
	x_c = data - repmat(mu(:,i)', n, 1);
	sigma{i} = (x_c' * (diag(gamma(:,i))*x_c)   )/N_k(i);
end

Pi = N_k/(n+0.0);
