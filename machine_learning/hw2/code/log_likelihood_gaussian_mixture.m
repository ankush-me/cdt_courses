function ll = log_likelihood_gaussian_mixture(data,mu,sigma,Pi)
% Calculates the log likelihood of the data given the parameters of the
% model
%
% @param data   : each row is a d dimensional data point
% @param mu     : a d x k dimensional matrix with columns as the means of
% each cluster
% @param sigma  : a cell array of the cluster covariance matrices
% @param pi     : a column matrix of probabilities for each cluster
%
% @return ll    : the log likelihood of the data (scalar)

[N,D] = size(data);
[D,K] = size(mu);

P = zeros(N,K);
for k=1:K
	P(:,k) = mvnpdf(data, mu(:,k)', sigma{k}); % get the probabilities
end
P = P*diag(Pi); %% multiply with the prior probability

ll = sum(log(sum(P,2)));
