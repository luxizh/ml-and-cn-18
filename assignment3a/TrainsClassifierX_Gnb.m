function parameters = TrainsClassifierX_Gnb(inputs, output)
    c=tabulate(output);
    parameters.nclass=size(c,1);
    parameters.pclass=c(:,3);
    %parameters.mu=zeros(nclass,)
    for i=1:parameters.nclass
        idx=find(output==c(i,1));
        parameters.mu(i,:)=mean(inputs(idx,:),1);
        parameters.sigmma(i,:)=var(inputs(idx,:),1);
    end
    %parameters.mu=mean(inputs,2);
    
end