function class = ClassifyX_knn(input, parameters)
    input = bsxfun(@minus, input, parameters.mu);
    input = bsxfun(@rdivide, input, parameters.sigma);
    idx=knnsearch1(input,parameters.train_in,parameters.k);
    %idx=knnsearch(parameters.train_in,input,parameters.k);
    class=mode(parameters.train_out(idx),2);
    %value=sum(parameters.train_out(idx));
    %if value>parameters.k/2
    %    class=1;
    %else
    %    class=0;
    %end
end