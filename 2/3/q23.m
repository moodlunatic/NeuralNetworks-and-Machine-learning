
%DEMARD	Automatic relevance determination using the MLP.
%
%	Description
%	This script demonstrates the technique of automatic relevance
%	determination (ARD) using a synthetic problem having three input
%	variables: X1 is sampled uniformly from the range (0,1) and has a low
%	level of added Gaussian noise, X2 is a copy of X1 with a higher level
%	of added noise, and X3 is sampled randomly from a Gaussian
%	distribution. The single target variable is determined by
%	SIN(2*PI*X1) with additive Gaussian noise. Thus X1 is very relevant
%	for determining the target value, X2 is of some relevance, while X3
%	is irrelevant. The prior over weights is given by the ARD Gaussian
%	prior with a separate hyper-parameter for the group of weights
%	associated with each input. A multi-layer perceptron is trained on
%	this data, with re-estimation of the hyper-parameters using EVIDENCE.
%	The final values for the hyper-parameters reflect the relative
%	importance of the three inputs.
%
%	See also
%	DEMMLP1, DEMEV1, MLP, EVIDENCE
%

%	Copyright (c) Ian T Nabney (1996-2001)

clc;


% Generate the data set.

x = trainset;
t = labels_train;

% Plot the data and the original function.




% Set up network parameters.
nin = 30;			% Number of inputs.
nhidden = 2;			% Number of hidden units.
nout = 1;			% Number of outputs.
aw1 = 0.01*ones(1, nin);	% First-layer ARD hyperparameters.
ab1 = 0.01;			% Hyperparameter for hidden unit biases.
aw2 = 0.01;			% Hyperparameter for second-layer weights.
ab2 = 0.01;			% Hyperparameter for output unit biases.
beta = 50.0;			% Coefficient of data error.

% Create and initialize network.
prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp(nin, nhidden, nout, 'linear', prior, beta);

% Set up vector of options for the optimiser.
nouter = 2;			% Number of outer loops
ninner = 10;		        % Number of inner loops
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence must occur
options(3) = 1.0e-7;
options(14) = 300;		% Number of training cycles in inner loop. 

% Train using scaled conjugate gradients, re-estimating alpha and beta.
for k = 1:nouter
  net = netopt(net, options, x, t, 'scg');
  [net, gamma] = evidence(net, x, t, ninner);
  fprintf(1, '\n\nRe-estimation cycle %d:\n', k);
  disp('The first three alphas are the hyperparameters for the corresponding');
  disp('input to hidden unit weights.  The remainder are the hyperparameters');
  disp('for the hidden unit biases, second layer weights and output unit')
  disp('biases, respectively.')
  fprintf(1, '  alpha =  %8.5f\n', net.alpha);
  fprintf(1, '  beta  =  %8.5f\n', net.beta);
  fprintf(1, '  gamma =  %8.5f\n\n', gamma);
  disp(' ')
  disp('Press any key to continue.')
  pause
end



fprintf(1, '    alpha1: %8.5f\n', net.alpha(1));
fprintf(1, '    alpha2: %8.5f\n', net.alpha(2));
fprintf(1, '    alpha3: %8.5f\n', net.alpha(3));


