function parameters = TrainsClassifierX_knn(inputs, output)
    parameters.mu = mean(inputs,1);
    inputs = bsxfun(@minus, inputs, parameters.mu);

    parameters.sigma = std(inputs,1);
    inputs = bsxfun(@rdivide, inputs, parameters.sigma);
    
    parameters.train_in=inputs;
    parameters.train_out=output;
    parameters.k=12;
end