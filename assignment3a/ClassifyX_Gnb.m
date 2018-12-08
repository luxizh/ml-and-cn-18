function class = ClassifyX_Gnb(input, parameters)
%{
%incorrect format do not know why but not saving time
    for i=1:parameters.nclass
        x = bsxfun(@minus, input, parameters.mu(i,:));
        x = bsxfun(@rdivide, x, parameters.sigma(i,:));
        compute_p(:,i)=prod(1./sqrt(2*pi).*exp(-(x.^2)./2),2)*parameters.pclass(i);
    end
    %}
    %normalize performance no obviouse change timing a bit longer
    %input = bsxfun(@minus, input, parameters.nmu);
    %input = bsxfun(@rdivide, input, parameters.nsigma);
    %pca worse
    %input=input*parameters.U(:,1:parameters.k);
    for i=1:parameters.nclass
        compute_p(:,i)=prod(bsxfun(@times,1./sqrt(2*pi*parameters.sigma(i,:)),...
            exp(bsxfun(@rdivide,-((bsxfun(@minus,input,parameters.mu(i,:)).^2)),...
            (2*parameters.sigma(i,:))))),2).*parameters.pclass(i);
        %class.a=1./sqrt(2*pi*parameters.sigma(i,:));
        %class.b=-((bsxfun(@minus,input,parameters.mu(i,:)).^2));
        %class.c=(2*parameters.sigma(i,:));
    end
    
    [~, b]=max(compute_p,[],2);
    class=b-1;
end