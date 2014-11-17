% This main file is used to execute sequential Monte Carlo sampling for 
% gaussian mixture modeling.  The data is the fisher iris data where each 
% row of data are four measurements taken from the pedal of an iris flower.  
% Important variables are listed below.
%
% data  : data matrix N x D with rows as elements of data
% z     : vector N x 1, of cluster assignments in current sample
%


clear 

load fisheriris

data = meas;
clear species meas

[N, D] = size(data);
data = data(randperm(N),:);

% K is the MAXIMUM number of clusters to use
K = 10;

% we set a few hyperparameters (parameters of our prior)
alpha = 1; % dirichlet parameter
Lambda_0 = eye(D); % wishart parameter
nu = 5; % wishart degrees of freedom
beta = 1; % normal covariance parameter

%% 
num_particles = 20;
zs = run_smc_sweep(num_particles, data, K, alpha, beta, Lambda_0, nu);

% % plot the histograms of the labels:
% figure(2);
% sp = 1;
% for ip=1:10:100
% 	subplot(2,5,sp);
% 	sp = sp + 1;
% 	hist(zs{ip});
% end

%% plot the labels as per particle number 2:
figure(2);
subplot(121);
plot_data(data,zs{2});
subplot(122);
hist(zs{2});


