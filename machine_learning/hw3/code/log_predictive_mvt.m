function [lp] = log_predictive_mvt(x, Nk, X_bar_k, Sk, beta, Lambda_0, nu)
% Computes the [log] conditional probability of x belonging to class k, given
% the class-assigments for the rest of the data-points, i.e.
% 	= p(x | z(x)=k, z{\x}, all_hyperparameters)	
%     [eq (27) in http://www.kamperh.com/notes/kamper_bayesgmm13.pdf]
%
% @param:
% x: new data point, D x 1 vector
% Nk: number of observations within this class (not including x)
% X_bar_k: mean value of observations within this class (not including x)
% Sk: empirical covariance matrix of observations within this class (not including x)
% beta: hyperparameter on precision of cluster means
% Lambda_0: DxD matrix hyperparameter for wishart
% nu: degrees of freedom hyperparameter for wishart
%
% @return:
% log porbability as defined above.

D = size(Sk, 1);
assert(size(X_bar_k,1) == D);

% compute parameters for student t
mu_star_k     = get_param_mu(beta,  Nk, X_bar_k);
Lambda_star_k = get_param_lambda(Lambda_0, beta, Nk, X_bar_k, Sk);
beta_star_k   = get_param_beta(beta, Nk);
nu_star_k     = get_param_nu(nu, Nk);

df = nu_star_k - D + 1;
W = Lambda_star_k * (beta_star_k + 1) / (beta_star_k * df);

lp = logmvtpdf(x, mu_star_k, W, df);


