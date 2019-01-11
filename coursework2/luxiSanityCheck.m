function luxiSanityCheck()
tic;
assert(fopen('TrainClassifierX.m') > 0,'Could not find TrainClassifierX.m function file')
assert(fopen('ClassifyX.m') > 0,'Could not find ClassifyX.m function file')
load data
% Each datapoint is described by 3 distinct features and labelled with a
% single integer value.

n=size(data,1);
elems = randperm(n)';
train_idx=elems(1:floor(n/2));
test_idx=elems(floor(n/2)+1:n);
train_data=data(train_idx,2:65);
train_labels=data(train_idx,1);
test_data=data(test_idx,2:65);
test_labels=data(test_idx,1);
% TrainClassifierX should accept such input
% There are no requirements regardsing the format of the parameters
% variable.
parameters = TrainClassifierX(train_data, train_labels);

disp('TrainClassifierX has been implemented correctly.')


%predicted_labels = -1*ones(rand_test,1);

% Fuction ClassifyX should take 3 features of a single datapoint and return 
% the predicted class (a single integer) to which the particular point belongs.
for i = 1:size(test_labels,1)
   predicted_labels(i,1) = ClassifyX(test_data(i,:), parameters);
end

assert(max(predicted_labels) < 6 && min(predicted_labels) > 0, 'Classifier output label in invalid range.')
p=length(find(predicted_labels==test_labels))/length(test_labels);
disp('ClassifyX has been implemented correctly.')
disp('Sanity check passed!')
toc;
disp(p);
disp(toc);
end