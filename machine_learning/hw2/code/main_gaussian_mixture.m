% This main file is used to execute the em algorithm for gaussian mixture
% modeling.  The data is the fisher iris data where each row of data are
% four measurements taken from the pedal of an iris flower.  The value is e
% is a small number to asses convergence of the algorithm.  Whent the
% likelihood of the data under the model ceases to increase by e every time
% the algorithm is assumed to have converged.  Important variables are
% listed below.
%
% data  : data matrix n x d with rows as elements of data
% gamma : a n x k matrix of responsilities.  each row should sum to 1.
% pi    : column vector of probabilities for each class
% param :  mu   : d x k matrix of class centers listed as columns
% sigma : k x 1 cell array of class covariance matrices (each are d x d)


clear 

% k is the number of clusters to use, you should experiment with this
% number and MAKE SURE YOUR CODE WORKS FOR ANY VALUE OF K >= 1
k = 3;
e = .001;

load fisheriris;

data = meas;
%clear species meas;

% this sets the initial values of the gamma matrix, the matrix of
% responsibilities, randomly based on independent draws from a dirichlet
% distribution.
gamma = gamrnd(ones(size(data,1),k),1);
gamma = gamma ./ repmat(sum(gamma,2),1,k);

% to facilitate visualization, we label each data point by the cluster
% which takes most responsibility for it.
[m labels] = max(gamma,[],2);


% this draws a plot of the initial labeling.
clf;
figure(1);
plot_data(data,labels);

% given the initial labeling we set mu, sigma, and pi based on the m step
% and calculate the likelihood.
ll = -inf;
[mu,sigma,Pi] = m_step_gaussian_mixture(data,gamma);
nll = log_likelihood_gaussian_mixture(data,mu,sigma,Pi);
disp(['the log likelihood = ' num2str(nll);])

%% plot ground-truth:
%%figure(3);
%%plot_data(data, species);

% the loop iterates until convergence as determined by e.
while ll + e < nll
    ll = nll;
    gamma = e_step_gaussian_mixture(data,Pi,mu,sigma);
    [mu,sigma,Pi] = m_step_gaussian_mixture(data,gamma);
    nll = log_likelihood_gaussian_mixture(data,mu,sigma,Pi);
    disp(['the log likelihood = ' num2str(nll)]);
    
    [m labels] = max(gamma,[],2);
    figure(2)
    hold on;
    pc = plot_data(data,labels);
    
    %% transform mu and sigma by (first two) principle components:
    pc = pc(:,1:2);
    mu_p = mu'*pc;
    sigma_p = {};
    for ii=1:k
        sigma_p{ii} = pc'*sigma{ii}*pc;
    end
    scatter(mu_p(:,1), mu_p(:,2), 200,'s', 'filled');
    
    %% plot pdf contours:
    nn = 100;
    x = linspace(2,10,nn);
    y = linspace(4,7,nn);
    [xx,yy] = meshgrid(x,y);
    zz = zeros(nn,nn);
    for xi=1:nn
        for yi=1:nn 
            zz(xi,yi) = gmm_pdf(x(xi),y(yi), mu_p, sigma_p, Pi);
        end
    end
    %surf(xx,yy,zz');
    contour(xx,yy,zz',50);
    %ezcontour(@(x,y)gmm_pdf(x,y,mu_p,sigma_p,Pi), [2,10,4,7]);
    pause(0.3);

    if ll+e < nll
        clf();
    end
end
