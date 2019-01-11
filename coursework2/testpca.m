clear ; close all; clc
load ('data.mat');

X=data(:,2:65);
Y=data(:,1);

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

[m, n] = size(X_norm);
Sigma=1/m*X_norm'*X_norm;
[U,S,V]=svd(Sigma);
for K=1:n
    %Z = projectData(X_norm, U, K);
    %X_rec  = recoverData(Z, U, K);
    if (sum(sum(S(1:K,1:K)))/sum(sum(S))>=0.99)
        break;
    end
end
Z=X_norm*U(:,1:K);