clear;
load data
%n=size(data,1);
%elems = randperm(n)';
inputs=data(:,2:65);
labels=data(:,1);

%% each feature
[m, m1] = size(inputs);
idx1=labels==1;
idx2=labels==2;
idx3=labels==3;
idx4=labels==4;
idx5=labels==5;

for i=1:64
        %5 classes againest each feature
        subplot(8,8,i),h=histogram(inputs(idx1,i));h.EdgeColor=h.FaceColor;hold on;
        subplot(8,8,i),h=histogram(inputs(idx2,i));h.EdgeColor=h.FaceColor;hold on;
        subplot(8,8,i),h=histogram(inputs(idx3,i));h.EdgeColor=h.FaceColor;hold on;
        subplot(8,8,i),h=histogram(inputs(idx4,i));h.EdgeColor=h.FaceColor;hold on;
        subplot(8,8,i),h=histogram(inputs(idx5,i));h.EdgeColor=h.FaceColor;
        
        %xlabel(['feature',num2str(i)]);
        %ylabel('frequency');
end

%% scatter matrix
figure;
[m, m1] = size(inputs);
idx1=find(labels==1);
idx2=find(labels==2);
idx3=find(labels==3);
idx4=find(labels==4);
idx5=find(labels==5);
f=4;%number of features
for i=1:f
    for j=1:f
        %5 classes in scatter matrix 
        subplot(f,f,(i-1)*f+j),plot(inputs(idx1,i),inputs(idx1,j),'.');hold on;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx2,i),inputs(idx2,j),'.');hold on;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx3,i),inputs(idx3,j),'.');hold on;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx4,i),inputs(idx4,j),'.');hold on;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx5,i),inputs(idx5,j),'.');
        %{
        % to plot part of the scatter matrix
        ii=i+57;
        jj=j+57;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx1,ii),inputs(idx1,jj),'.');hold on;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx2,ii),inputs(idx2,jj),'.');hold on;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx3,ii),inputs(idx3,jj),'.');hold on;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx4,ii),inputs(idx4,jj),'.');hold on;
        subplot(f,f,(i-1)*f+j),plot(inputs(idx5,ii),inputs(idx5,jj),'.');
        %}
    end
end

%% 3d
figure
 %implement normalization
    mu = mean(inputs,1);%normalization parameters
    inputs = bsxfun(@minus, inputs, mu);
    sigma = std(inputs,1);%normalization parameters
    inputs = bsxfun(@rdivide, inputs, sigma);
    
    %implement PCA
    [m, m1] = size(inputs);
    Sigma=1/m*(inputs')*inputs;
    [U,S,~]=svd(Sigma);%compute U S (V)
    %reduce to 3D
    inputs=inputs*U(:,1:3);
    %compute variance
    variance=sum(sum(S(1:3,1:30)))/sum(sum(S));
for i=1:5
    idx=find(labels==i);
    plot3(inputs(idx,1),inputs(idx,2),inputs(idx,3),'.');
    hold on;
end
legend('class1','class2','class3','class4','class5')
grid on