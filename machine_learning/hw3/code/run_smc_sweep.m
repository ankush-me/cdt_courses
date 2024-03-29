function [z] = run_smc_sweep(num_particles, data, K, alpha, beta, Lambda_0, nu)
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
%% class-label distribution for z_n, given, [z_{1:n-1} and alpha].
%% --> (implemented in log_predictive_dirichlet.m)
%%
%% Hence, the weights are student-t distribution of x|z(x)=k
%% --> (implemented in log_predictive_mvt.m)
%%
%%%%%%%%%%%%%%%%%%%%

[N, D] = size(data);

history = zeros(N,num_particles);
W = zeros(N,num_particles); %% store the weights

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
		S{np,i} = zeros(D); %% set class-wise covars to 0
	end
end


for n = 1:N
    fprintf('Data point %d of %d\n', n, N);
    x_n = data(n,:);
    % sample particles from proposal distribution
    % and compute importance weights
    z_p = -1*ones(num_particles,1); %% class-assignments sampled from proposal
    lg_w = zeros(num_particles,1); %% log-weights of each particle
    for ip = 1:num_particles
    	log_proposal = zeros(K,1);
    	for ik=1:K
	    	log_proposal(ik) = log_predictive_dirichlet(Ns{ip}(ik),N,K,alpha);
	    end
	    [z_p(ip),~] = sample_from_unnormalized_log_prob(log_proposal);
	    lg_w(ip) = log_predictive_mvt(x_n',Ns{ip}(z_p(ip)),mu{ip}(:,z_p(ip)),S{ip,z_p(ip)},beta,Lambda_0,nu);
    end
    
    % update estimate of marginal likelihood [this is optional --> do later]
   
    % resample
    [~, resample_dist] = sample_from_unnormalized_log_prob(lg_w);
    N_eff = 1/sum(resample_dist.^2);
    resample_dist = resample_dist + 0.0005; %% add some noise to prevent particle deprivation
    resample_dist = resample_dist / sum(resample_dist);
    W(n,:) = resample_dist;
    num_resamples = low_variance_resampling(resample_dist');
    num_cumsum = cumsum(num_resamples);
    sample_idx = 1;
    for ip=1:num_particles
    	while ip > num_cumsum(sample_idx)
    		sample_idx = sample_idx + 1;
    	end
    	history(n,ip) = sample_idx; %% store the ancestor relationship
    	z_n_new = z_p(sample_idx);
    	z{ip}(n) = z_n_new;
    	%% update the class-statistic for the new class
    	particle_labels = get_particle_labels(n, ip, z, history);
        for kk=1:K
    	   [Ns{ip}(kk), mu{ip}(:,kk), S{ip,kk}] = calc_statistics(data(particle_labels==kk,:));
        end
        %[Ns{ip}(z_n_new), mu{ip}(:,z_n_new), S{ip,z_n_new}] = calc_statistics(data(particle_labels==z_n_new,:));
    end

    % plot the assigments as per the first particle: (for debugging and visualization)
    if mod(n, 10) == 0
        figure(1);
		subplot(121);
		l1 = get_particle_labels(n,1,z,history);
		plot_data(data(1:n,:),l1);
		subplot(122);
		hist(l1);
        drawnow;
    end
end
%% plot the history matrix:
% figure(4);
% imagesc(W);
% colorbar();
% pause();
save('H.mat', 'history');
zo = cell(num_particles,1);
for i=1:num_particles
	zo{i} = get_particle_labels(n, i, z, history);
end
z = zo;