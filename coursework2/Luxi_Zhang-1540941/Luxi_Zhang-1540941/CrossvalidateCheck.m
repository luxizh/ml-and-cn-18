% crossvalidate to choose parameters and compute accuracy and confusion matrix
clear;

assert(fopen('TrainClassifierX.m') > 0,'Could not find TrainClassifierX.m function file')
assert(fopen('ClassifyX.m') > 0,'Could not find ClassifyX.m function file')
load data
% Each datapoint is described by 3 distinct features and labelled with a
% single integer value.

n=size(data,1);
elems = randperm(n)';
%p=zeros(1,5);
%t=zeros(1,5);
for lambda=[0 0.5 0.7 1 3 5] %set different lambda
    for iter=10:10:100 %set different iterations
        for units=20:5:40 %set different units
    for i=1:5
    % set n fold cross validate set
    test_idx=elems(1:floor(n/5));
    train_idx=elems(floor(n/5)+1:n);
    elems=[elems(floor(n/5)+1:n);elems(1:floor(n/5))];
    train_data=data(train_idx,2:65);
    train_labels=data(train_idx,1);
    test_data=data(test_idx,2:65);
    test_labels=data(test_idx,1);
    tic;
    %start compute time cost
    %train data
    parameters = TrainClassifierX(train_data, train_labels,lambda,units,iter);
    %test data
    predicted_labels = ClassifyX(test_data, parameters);
    %check output
    assert(max(predicted_labels) < 6 && min(predicted_labels) > 0, 'Classifier output label in invalid range.')
    %compute accuracy
    p(i)=length(find(predicted_labels==test_labels))/length(test_labels);
    %disp('ClassifyX has been implemented correctly.')
    %disp('Sanity check passed!')
    t(i)=toc;
    %end compute time cost
    %disp(toc);
    %disp('TrainClassifierX has been implemented correctly.')
    %compute confusion matrix
    confu=zeros(5,5);
    %initialization
    for classi=1:5
        for classj=1:5
            %compute
            confu(classi,classj)=length(find(((test_labels==classi).*predicted_labels)==classj));
        end
    end
    end
    %show performance
    %disp(lambda);
    %disp(layers);
    fprintf('%2.0f %2.0f %f %f %f\n',iter,units,lambda,mean(p),mean(t));
    %disp(p);
    %disp(mean(p));
    %disp(mean(t));
        end
    end
end
return
