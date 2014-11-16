function [z, Z_hat] = run_smc_sweep(num_particles, data, K, alpha, beta, Lambda_0, nu)
% Run SMC sweep over the data
%
% Returns:
%
% z: cell array (vector) of length num_particles, each a Nx1 vector of
%    class assignments
% Z_hat: (optional) estimate of the marginal likelihood p(data) computed
%    during this SMC sweep
%
% Inputs:
%
% num_particles: how many particles to run in this sweep
% data: NxD matrix of observed data
% K: (maximum) number of classes, i.e. length of dirichlet vector
% alpha: hyperparameter on dirichlet
% beta: hyperparameter on cluster means
% Lambda_0: DxD hyperparameter on cluster precisions
% nu: hyperparameter on cluster precisions
%


%%%%%%%%%%%%%%%%%%%%
%% The proposal distribution is chosen to be: the predictive
%% class-label distribution for z_n, given, (z_{1:n-1}, alpha).
%% --> (implemented in log_predictive_dirichlet.m)
%%
%% Hence, the weights are student-t distribution of x|z(x)=k
%% --> (implemented in log_predictive_mvt.m)
%%
%%%%%%%%%%%%%%%%%%%%

[N, D] = size(data);

z    = cell(num_particles,1);  %% particles
%% initialize class-statistics for each particle:
Ns = cell(num_particles,1);
mu = cell(num_particles,1);
S  = cell(num_particles,K);
for np=1:num_particles
	z{np}  = -1*ones(N,1); %% set-class assigments to be invalid
	Ns{np} = zeros(K,1);   %% set class-wise number of assignments to 0
	mu{np} = zeros(D,K);   %% set class-wise means to 0
	for i=1:K
		S{mp,i} = zeros(D); %% set class-wise covars to 0
	end
end


for n = 1:N
    fprintf('Data point %d of %d\n', n, N);
    x_n = x_n = data(n,:);
    % sample particles from proposal distribution
    % and compute importance weights
    z_p = -1*ones(num_particles,1); %% class-assignments sampled from proposal
    lg_w = zeros(num_particles,1); %% log-weights of each particle
    for ip = 1:num_particles
    	log_proposal = zeros(K,1);
    	for ik=1:K
	    	log_proposal(k) = log_predictive_dirichlet(Ns{ip}(ik),N,K,alpha);
	    end
	    z_p(ip)  = sample_from_unnormalized_log_prob(log_proposal);
	    lg_w(ip) = log_predictive_mvt(x_n',Ns{ip}(z_p(ip)),mu{ip}(:,z_p(ip)),S{ip,z_p(ip)},beta,Lambda_0,nu);
    end

    % update estimate of marginal likelihood
    %% [z{ip}(n),~]
    ?
    
    % resample
    ?
    ?
    ?
    ?

    % plot (for debugging and visualization)
    if mod(n, 10) == 0
        figure(1);
        plot_data(data(1:n,:),z{1}(1:n));
        drawnow;
    end
    
end

end
