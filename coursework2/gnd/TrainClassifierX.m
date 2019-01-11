function parameters = TrainClassifierX(input, label)

%% All implementation should be inside the function.

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