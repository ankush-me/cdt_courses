function [hist] = low_variance_resampling(dist)
%% low-variance resampling.
% ref: http://www.cs.berkeley.edu/~pabbeel/cs287-fa11/slides/particle-filters++_v2.pdf
N = size(dist,2);
hist=zeros(1,N);
r = rand()/N;
rs= r + (0:N-1)/N;
cdist = cumsum(dist);
sample = zeros(1,N);
for i=1:N
	[~,sample(i)] = max(rs(i) < cdist);
end
for i=1:N
	hist(i) = sum(sample==i);
end

