function [ll] = get_particle_labels(n, particle_idx, assignment, history)
% Due to resampling, the assignments die out.
% True labels, need to be read from the ancestory matrix -- history.
%
% @param
% n : labels are returned for variable from x_1 to x_n
% assignment : a [num_particle x 1] cell array, each with [Nx1] vector of labels.
% history : a [N x num_particle] matrix of indices which encodes the node ancestory.
%
% @return
% ll : [n x 1] vector of labels for this particle.
ll(n) = assignment{particle_idx}(n);
current_idx = particle_idx;
for ni=n-1:-1:1
	ancestor_idx = history(ni+1, current_idx);
	current_idx = ancestor_idx;
	ll(ni) = assignment{ancestor_idx}(ni);
end
