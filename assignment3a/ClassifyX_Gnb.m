function class = ClassifyX_Gnb(input, parameters)
    for i=1:parameters.nclass
        compute_p(:,i)=prod(bsxfun(@times,1./sqrt(2*pi*parameters.sigmma(i,:)),...
            exp(bsxfun(@rdivide,-((bsxfun(@minus,input,parameters.mu(i,:)).^2)),...
            (2*parameters.sigmma(i,:))))),2).*parameters.pclass(i);
        %class.a=1./sqrt(2*pi*parameters.sigmma(i,:));
        %class.b=-((bsxfun(@minus,input,parameters.mu(i,:)).^2));
        %class.c=(2*parameters.sigmma(i,:));
    end
    [a b]=max(compute_p');
    class=b'-1;
end