function [z, lp] = run_gibbs_sweep(data, z, K, alpha, beta, Lambda_0, nu)
% Run a single Gibbs sampler sweep through the data points, resampling each
% of the N data points
%
% Returns:
% ========
% z: a Nx1 vector of class assignments after resampling
% lp: the log joint probability of the sampled z and the data
%
% Arguments:
% ==========
% data: (n x d) matrix of data-points
% z: a Nx1 vector with the current class label assignments
% K: maximum number of cluster i.e., size of categorical distribution of pi
% alpha: hyperparameter on dirichlet pi
% beta: hyperparameter on precision of cluster means
% Lambda_0: DxD matrix hyperparameter for wishart
% nu: degrees of freedom hyperparameter for wishart

[N, D] = size(data);

%% compute class-statistics:
Ns  = zeros(K,1);
mu = zeros(D,K);
S  = {}; 
for i=1:K
    Ns(i) = sum(z==i);
    if Ns(i) > 0
        [~, mu(:,i) ,S{i}] = calc_statistics(data(z==i,:));
    else
        S{i} = zeros(D);
    end
end

% loop through every data point
for n = 1:N
    x_n = data(n,:);
    z_n_previous = z(n);
 
    log_prob = zeros(K,1);
    % compute probability of each assignment
    for k=1:K
        if z(n)~=k
            log_prob(k) = log_predictive_collapsed_gmm(x_n',K,N,Ns(k),mu(:,k),S{k},alpha,beta,Lambda_0,nu);
        else
            [N_n,mu_n,S_n] = exclude_statistics(x_n', Ns(k), mu(:,k), S{k});
            log_prob(k)    = log_predictive_collapsed_gmm(x_n',K,N,N_n,mu_n,S_n,alpha,beta,Lambda_0,nu);
        end
    end

    % sample a new class assignment
    [z_n_new,~] = sample_from_unnormalized_log_prob(log_prob);
    %% change the class-statistics:
    if z_n_new ~= z_n_previous
        z(n) = z_n_new;
        z_p = z_n_previous; % alias
        [Ns(z_p), mu(:,z_p), S{z_p}] = exclude_statistics(x_n', Ns(z_p), mu(:,z_p), S{z_p});
        [Ns(z(n)), mu(:,z(n)), S{z(n)}] = calc_statistics(data(z==z(n),:));
    end
end

lp = log_joint_collapsed_gmm(data, z, Ns,mu',S,alpha,beta,Lambda_0,nu);
end
