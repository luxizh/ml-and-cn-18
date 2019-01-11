function class = ClassifyX(input, parameters)
    input = bsxfun(@minus, input, parameters.mu);
    input = bsxfun(@rdivide, input, parameters.sigma);
    
    input=input*parameters.U;
    
    m = size(input, 1);
    %num_labels = size(parameters.Theta2, 1);

    % You need to return the following variables correctly 
    h1 = sigmoid([ones(m, 1) input] * parameters.Theta1');
    h2 = sigmoid([ones(m, 1) h1] * parameters.Theta2');
    %[dummy, p] = max(h2, [], 2);
    [~, class] = max(h2, [], 2);

end
function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
g = 1.0 ./ (1.0 + exp(-z));
end