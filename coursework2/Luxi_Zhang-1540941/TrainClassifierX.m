function parameters = TrainClassifierX(inputs, output)%,lambda,units,iter

    %implement normalization
    parameters.mu = mean(inputs,1);%normalization parameters
    inputs = bsxfun(@minus, inputs, parameters.mu);
    parameters.sigma = std(inputs,1);%normalization parameters
    inputs = bsxfun(@rdivide, inputs, parameters.sigma);
    
    %implement PCA
    [m, m1] = size(inputs);
    Sigma=1/m*(inputs')*inputs;
    [U,S,~]=svd(Sigma);%compute U S (V)
    %find the value k to remain 99% variance
    for K=1:m1
        if (sum(sum(S(1:K,1:K)))/sum(sum(S))>=0.99)
            break;
        end
    end
    inputs=inputs*U(:,1:K);%Pca on inputs
    parameters.U=U(:,1:K);%Pca paprameters
    
    %initialize papameters for bp network
    parameters.input_layer_size=K; %number of features after pca
    parameters.hidden_layer_size=30;%units; %units in hidden layer
    parameters.num_labels=5; % 5 labels
    
    %randam initialze weighs
    initial_Theta1 = randInitializeWeights(parameters.input_layer_size,parameters.hidden_layer_size);
    initial_Theta2 = randInitializeWeights(parameters.hidden_layer_size,parameters.num_labels);

    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];    
    %  set lambda and iter
    lambda = 1;
    iter=70;
    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                       parameters.input_layer_size, ...
                                       parameters.hidden_layer_size, ...
                                       parameters.num_labels,...
                                       inputs, output, lambda);

    % Now, costFunction takes in only one argument (the neural network parameters)
    [nn_params, ~] = fmincg(costFunction, initial_nn_params, iter);

    % Obtain Theta1 and Theta2 back from nn_params
    parameters.Theta1 = reshape(nn_params(1:parameters.hidden_layer_size * ...
        (parameters.input_layer_size + 1)), parameters.hidden_layer_size, ...
        (parameters.input_layer_size + 1));

    parameters.Theta2 = reshape(nn_params((1 + (parameters.hidden_layer_size * ...
        (parameters.input_layer_size + 1))):end),...
        parameters.num_labels, (parameters.hidden_layer_size + 1));
   
end
 function g = sigmoid(z)
%Compute sigmoid functoon
g = 1.0 ./ (1.0 + exp(-z));
end

function g = sigmoidGradient(z)
%returns the gradient of the sigmoid function evaluated at z
g=sigmoid(z).*(1-sigmoid(z));
end

function W = randInitializeWeights(L_in, L_out)
%Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%The first column of W corresponds to the parameters for the bias unit
epsilon_init=0.12;
W=rand(L_out, 1+L_in)*2*epsilon_init-epsilon_init;
end

function [J,grad] = nnCostFunction(nn_params,input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels,X, y, lambda)
%Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2,
%  the weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some variables
m = size(X, 1);        
J = 0;
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Theta2));

% Feedforward the neural network and return the cost in the variable J. 
a1=[ones(m,1) X];
z2=a1*Theta1';
a2=[ones(m,1) sigmoid(z2)];
z3=a2*Theta2';
a3=sigmoid(z3);

%cost without regularization
for k=1:size(Theta2,1)
    %index=find(y==k);
    J=J+1/m*sum(-(y==k)'*log(a3(:,k))-(1-(y==k))'*(log(1-a3(:,k))));
end

%cost with regularization
J=J+lambda/2/m*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

%bp
%d3=zeros(size(a3));
%d2=zeros(size(a2));
% Implement the backpropagation algorithm to compute the gradients
% Theta1_grad and Theta2_grad.
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for i=1:m
    yi=zeros(1,size(Theta2,1));
    yi(y(i))=1;
    delta3=a3(i,:)-yi;
    t=Theta2'*delta3';
    delta2 = t(2:end,:) .* sigmoidGradient(z2(i, :)');
%    delta2 = t(2:end,:) .* sigmoidGradient(z2(i, :)');
    Delta1 = Delta1 + delta2* a1(i, :);
%    Delta1 = Delta1 + delta2(2:end) * X(i, :);
    Delta2 = Delta2 + delta3' * a2(i,:);
end
%Implement regularization with the gradients.
Theta1_grad = Delta1 / m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m*Theta1(:, 2:end);
Theta2_grad = Delta2 / m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m*Theta2(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
function [X, fX, i] = fmincg(f, X, length)
% Minimize a continuous differentialble multivariate function. 

RHO = 0.01;   % a bunch of constants for line searches
SIG = 0.5;    % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;    % extrapolate maximum 3 times the current bracket
MAX = 20;     % max 20 function evaluations per line search
RATIO = 100;  % maximum allowed slope ratio

argstr = ['feval(f, X)'];  % compose string used to call function
i = 0;           % zero the run length counter
ls_failed = 0;   % no previous line search has failed
fX = [];
[f1,df1] = eval(argstr);  % get function value and gradient
i = i + (length<0);
s = -df1;% search direction is steepest
d1 = -s'*s; % this is the slope
z1 = 1/(1-d1);% initial step 

while i < abs(length)     % while not finished
  i = i + (length>0); 

  X0 = X; f0 = f1; df0 = df1;    % make a copy of current values
  X = X + z1*s;                  % begin line search
  [f2,df2] = eval(argstr);
  i = i + (length<0);  
  d2 = df2'*s;
  f3 = f1; d3 = d1; z3 = -z1;   % initialize point 3 equal to point 1
  %if length>0, M = MAX; else M = min(MAX, -length-i); end
  M = MAX;
  success = 0; limit = -1;% initialize quanteties
  while 1
    while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0) 
      limit = z1;   % tighten the bracket
      if f2 > f1
        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);  % quadratic fit
      else
        A = 6*(f2-f3)/z3+3*(d2+d3);  % cubic fit
        B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;
      end
      if isnan(z2) || isinf(z2)
        z2 = z3/2;  % if we had a numerical problem then bisect
      end
      z2 = max(min(z2, INT*z3),(1-INT)*z3);% don't accept too close to limits
      z1 = z1 + z2; % update the step
      X = X + z2*s;
      [f2,df2] = eval(argstr);
      M = M - 1; i = i + (length<0); 
      d2 = df2'*s;
      z3 = z3-z2; % z3 is now relative to the location of z2
    end
    if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
      break;    %  failure 
    elseif d2 > SIG*d1
      success = 1; break; % success
    elseif M == 0
      break;   % failure
    end
    A = 6*(f2-f3)/z3+3*(d2+d3);  % make cubic extrapolation
    B = 3*(f3-f2)-z3*(d3+2*d2);
    z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3)); % num. error possible - ok!
    if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0 
      if limit < -0.5  % if we have no upper limit
        z2 = z1 * (EXT-1); % the extrapolate the maximum amount
      else
        z2 = (limit-z1)/2; % otherwise bisect
      end
    elseif (limit > -0.5) && (z2+z1 > limit) 
      z2 = (limit-z1)/2;   % bisect
    elseif (limit < -0.5) && (z2+z1 > z1*EXT) % extrapolation beyond limit
      z2 = z1*(EXT-1.0);   % set to extrapolation limit
    elseif z2 < -z3*INT
      z2 = -z3*INT;
    elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT)) 
      z2 = (limit-z1)*(1.0-INT);
    end
    f3 = f2; d3 = d2; z3 = -z2; % set point 3 equal to point 2
    z1 = z1 + z2; X = X + z2*s; % update current estimates
    [f2,df2] = eval(argstr);
    M = M - 1; i = i + (length<0);       
    d2 = df2'*s;
  end  % end of line search

  if success  % if line search succeeded
    f1 = f2; fX = [fX' f1]';
%    fprintf('%4i,%4.6e;', i, f1);
%    scatter(i,f1);
    s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;   % Polack-Ribiere direction
    tmp = df1; df1 = df2; df2 = tmp;   % swap derivatives
    d2 = df1'*s;
    if d2 > 0            % new slope must be negative
      s = -df1;        % otherwise use steepest direction
      d2 = -s'*s;    
    end
    z1 = z1 * min(RATIO, d1/(d2-realmin));   % slope ratio but max RATIO
    d1 = d2;
    ls_failed = 0;  % this line search did not fail
  else
    fprintf('fail'); 
    X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
    if ls_failed || i > abs(length) % line search failed twice in a row
      break; % or we ran out of time, so we give up
    end
    tmp = df1; df1 = df2; df2 = tmp; % swap derivatives
    s = -df1; % try steepest
    d1 = -s'*s;
    z1 = 1/(1-d1);                     
    ls_failed = 1; % this line search failed
  end
end
end
