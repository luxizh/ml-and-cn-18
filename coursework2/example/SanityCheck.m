function SanityCheck()

assert(fopen('TrainClassifierX.m') > 0,'Could not find TrainClassifierX.m function file')
assert(fopen('ClassifyX.m') > 0,'Could not find ClassifyX.m function file')

% Each datapoint is described by 3 distinct features and labelled with a
% single integer value.
rand_train = randi([100,10000]);
train_data = rand(rand_train,64);
train_labels = randi([1,5],rand_train,1);
disp(train_labels)
% TrainClassifierX should accept such input
% There are no requirements regardsing the format of the parameters
% variable.
parameters = TrainClassifierX(train_data, train_labels);

disp('TrainClassifierX has been implemented correctly.')

rand_test = randi([50,5000]);
test_data = rand(rand_test,64);
predicted_labels = -1*ones(rand_test,1);

% Fuction ClassifyX should take 3 features of a single datapoint and return 
% the predicted class (a single integer) to which the particular point belongs.
for i = 1:rand_test
   predicted_labels(i,1) = ClassifyX(test_data(i,:), parameters);
end

assert(max(predicted_labels) < 6 && min(predicted_labels) > 0, 'Classifier output label in invalid range.')

disp('ClassifyX has been implemented correctly.')
disp('Sanity check passed!')

end