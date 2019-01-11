function parameters = TrainClassifierX(input, label)

%% All implementation should be inside the function.
%mu = mean(input);
%input = bsxfun(@minus, input, mu);

%sigma = std(input);
%input = bsxfun(@rdivide, input, sigma);
%{
[m, n] = size(input);
Sigma=1/m*(input')*input;
[U,S,V]=svd(Sigma);
for K=1:n
    %Z = projectData(X_norm, U, K);
    %X_rec  = recoverData(Z, U, K);
    if (sum(sum(S(1:K,1:K)))/sum(sum(S))>=0.99)
        break;
    end
end
parameters.K=K;
parameters.U=U;
input=input*U(:,1:K);
%}
% For example: Method not using any parameters.
%parameters = [];
    c=tabulate(label);
    parameters.nclass=size(c,1);
    parameters.pclass=c(:,3);
    %parameters.mu=zeros(nclass,)
    
    for i=1:parameters.nclass
        idx=find(label==c(i,1));
        parameters.mu(i,:)=mean(input(idx,:),1);
        parameters.sigma(i,:)=var(input(idx,:),1);
        %parameters.sigma(i,:)=std(inputs(idx,:),1);
    end
end