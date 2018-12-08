clear 
close all;
%clc
load cancer

n=size(cancer.inputs,1);
train_idx=1:floor(n/2);
test_idx=floor(n/2)+1:n;
input=cancer.inputs(test_idx,:);
output=cancer.outputs(test_idx,:);

%without pca
tic;
parameters = TrainsClassifierX_Gnb(cancer.inputs(train_idx,:),cancer.outputs(train_idx,:));
class = ClassifyX_Gnb(input, parameters);
toc
p1=length(find(class==output))/length(output);

tic;
parameters = TrainsClassifierX_knn(cancer.inputs(train_idx,:),cancer.outputs(train_idx,:));
%input=cancer.inputs(test_idx,:);
class = ClassifyX_knn(input, parameters);
toc
p2=length(find(class==output))/length(output);
return
%with pca
%knn timing much better performance a bit decrease
%gnd performace a bit worse
tic;
inputs=cancer.inputs(train_idx,:);
[m, m1] = size(inputs);
    Sigma=1/m*(inputs')*inputs;
    [U,S,V]=svd(Sigma);
    for K=1:m1
        %Z = projectData(X_norm, U, K);
        %X_rec  = recoverData(Z, U, K);
        if (sum(sum(S(1:K,1:K)))/sum(sum(S))>=0.999)
            break;
        end
    end
    inputs=inputs*U(:,1:K);
parameters = TrainsClassifierX_knn(inputs,cancer.outputs(train_idx,:));
input=input*U(:,1:K);
class = ClassifyX_knn(input, parameters);
toc
p=length(find(class==output))/length(output);

return