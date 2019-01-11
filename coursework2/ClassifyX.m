function label = ClassifyX(input, parameters)

%% All implementation should be inside the function.

% For example: Random binary label (returns 0 or 1 randomly).
%label = randi([1,5],1);

    for i=1:parameters.nclass
        compute_p(:,i)=prod(bsxfun(@times,1./sqrt(2*pi*parameters.sigma(i,:)),...
            exp(bsxfun(@rdivide,-((bsxfun(@minus,input,parameters.mu(i,:)).^2)),...
            (2*parameters.sigma(i,:))))),2).*parameters.pclass(i);
        %class.a=1./sqrt(2*pi*parameters.sigma(i,:));
        %class.b=-((bsxfun(@minus,input,parameters.mu(i,:)).^2));
        %class.c=(2*parameters.sigma(i,:));
    end
%{
    for i=1:parameters.nclass
        compute_p(:,i)=prod(...
            exp(bsxfun(@rdivide,-((bsxfun(@minus,input,parameters.mu(i,:)).^2)),...
            (2*parameters.sigma(i,:)))),2).*parameters.pclass(i);
        %class.a=1./sqrt(2*pi*parameters.sigma(i,:));
        %class.b=-((bsxfun(@minus,input,parameters.mu(i,:)).^2));
        %class.c=(2*parameters.sigma(i,:));
    end   
%}
    [~, b]=max(compute_p,[],2);
    label=b;
end


