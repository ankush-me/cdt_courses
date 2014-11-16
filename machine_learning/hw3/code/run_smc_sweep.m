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


[N, D] = size(data);

z = cell(num_particles,1);

?
?
?

for n = 1:N
    fprintf('Data point %d of %d\n', n, N);

    ?
    ?
    ?
    ?
    
    % sample particles from proposal distribution
    % and compute importance weights

    ?
    ?
    ?
    
    % update estimate of marginal likelihood
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
