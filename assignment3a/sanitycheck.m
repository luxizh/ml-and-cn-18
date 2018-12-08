clear 
close all;

load cancer

n=size(cancer.inputs,1);
train_idx=1:floor(n/2);
test_idx=floor(n/2)+1:n;
input=cancer.inputs(test_idx,:);
output=cancer.outputs(test_idx,:);
parameters = TrainsClassifierX_Gnb(cancer.inputs(train_idx,:),cancer.outputs(train_idx,:));
class = ClassifyX_Gnb(input, parameters);
p1=length(find(class==output))/length(output);

parameters = TrainsClassifierX_knn(cancer.inputs(train_idx,:),cancer.outputs(train_idx,:));
%input=cancer.inputs(test_idx,:);
class = ClassifyX_knn(input, parameters);
p2=length(find(class==output))/length(output);