function [N_k, mu_k, S_k] = calc_statistics(data_k);
% Compute the number, mean and covriance of the data-ponts in a class.
% data_k : [N_k x d] matrix of data-points in the class
%
% @return
% N_k : scalar : number of points in the class
% mu_k : mean of the points in the class
% S_k : covariance of the points in the class

N_k  = size(data_k,1);
if N_k~=0
	mu_k = sum(data_k, 1)'/N_k;
	data_c = data_k - repmat(mu_k', N_k, 1);
	S_k = (data_c'*data_c)/N_k;
else
	D    = 4;
	S_k  = zeros(D);
	mu_k = zeros(D,1);
end