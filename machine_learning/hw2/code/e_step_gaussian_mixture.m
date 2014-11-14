function gamma = e_step_gaussian_mixture(data,Pi,mu,sigma)
% Returns a matrix of responsibilities.
%
% @param    data : data matrix n x d with rows as elements of data
% @param    pi   : column vector of probabilities for each class
% @param    mu   : d x k matrix of class centers listed as columns
% @param    sigma: cell array of class covariance matrices (d x d)
%
% @return   gamma: n x k matrix of responsibilities


[N,D] = size(data);
[D,K] = size(mu);

P = zeros(N,K);
for i=1:K
	P(:,i) = mvnpdf(data, mu(:,i)', sigma{i}); % get the probabilities
end
W = P*diag(Pi); %% multiply with the prior probability

%% normalize:
x_prob = sum(W,2);
gamma  = W./repmat(x_prob, [1, K]);

