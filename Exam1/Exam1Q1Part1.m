% Expected risk minimization with 2 classes
clear all, close all,

n = 2;      % number of feature dimensions
N = 10000;   % number of iid samples

% Class 0 parameters 
mu(:,1) = [-0.1;0]; 
Sigma(:,:,1) = [1 -0.9;-0.9 1]; 

% Class 1 parameters 
mu(:,2) = [0.1;0]; 
Sigma(:,:,2) = [1 0.9;0.9 1]; 

p = [0.8,0.2]; % class priors for labels 0 and 1 respectively

label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space

for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
save('Exam1.mat','n','N','label','Nc','x');

y=evalGaussian(x,mu(:,2),Sigma(:,:,2))./evalGaussian(x,mu(:,1),Sigma(:,:,1));
ysort=sort(y);
epsilon=0.0000000000000000000000000000000000000000000000000000000001;

%calculate gamma
gamma=zeros(1,N+1);
gamma(1)= ysort(1)-epsilon;
gamma(N+1)=ysort(N)+epsilon;
for h=1:N-1
    gamma(h+1)=(ysort(h)+ysort(h+1))/2;
end

%calculate p_error
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
p_error=zeros(1,N+1);
for k=1:N+1
    decision = (discriminantScore >= log(gamma(k)));

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

p_error(k)= [p10,p01]*Nc'/N; % probability of error, empirically estimated
end

%draw picture to find min P(error) and gamma 
figure(1),clf,
plot(gamma,p_error,'.');
title('P(error) and gamma'),
xlabel('gamma'), ylabel('p_error'), 
[min_perror,min_perror_index]=min(p_error);
gamma_min_error=gamma(min_perror_index);

%draw ROC
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
for a=1:N+1
decision = (discriminantScore >= log(gamma(a)));
ind10 = find(decision==1 & label==0); p10(a) = length(ind10)/Nc(1); % probability of false positive
ind11 = find(decision==1 & label==1); p11(a) = length(ind11)/Nc(2); % probability of true positive

end
decision = (discriminantScore >= log(gamma_min_error));
ind10 = find(decision==1 & label==0); p102 = length(ind10)/Nc(1); % probability of false positive
ind11 = find(decision==1 & label==1); p112 = length(ind11)/Nc(2); % probability of true positive
figure(2),clf
plot(p10,p11);hold on,
plot(p102,p112,'og');
axis equal,
title('Minimum expected risk classification ROC'),
xlabel('P(D = 1|L = 0)False Positive Probability'), ylabel('P(D = 1|L = 1)True Positive Probability'), 
