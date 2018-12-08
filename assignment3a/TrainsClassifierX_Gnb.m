function parameters = TrainsClassifierX_Gnb(inputs, output)
%{
%normalize performance no obviouse change timing a bit longer
    parameters.nmu = mean(inputs,1);
    inputs = bsxfun(@minus, inputs, parameters.nmu);

    parameters.nsigma = std(inputs,1);
    inputs = bsxfun(@rdivide, inputs, parameters.nsigma);
%}
%{
    %pca worse
    [m, n] = size(inputs);
    Sigma=1/m*(inputs')*inputs;
    [U,S,V]=svd(Sigma);
    for K=1:n
        %Z = projectData(X_norm, U, K);
        %X_rec  = recoverData(Z, U, K);
        if (sum(sum(S(1:K,1:K)))/sum(sum(S))>=0.99)
            break;
        end
    end
    parameters.k=K;
    parameters.U=U;
    inputs=inputs*U(:,1:K);
    %}
    c=tabulate(output);
    parameters.nclass=size(c,1);
    parameters.pclass=c(:,3);
    %parameters.mu=zeros(nclass,)
    
    for i=1:parameters.nclass
        idx=find(output==c(i,1));
        parameters.mu(i,:)=mean(inputs(idx,:),1);
        parameters.sigma(i,:)=var(inputs(idx,:),1);
        %parameters.sigma(i,:)=std(inputs(idx,:),1);
    end
    
    %parameters.mu=mean(inputs,2);
    
end