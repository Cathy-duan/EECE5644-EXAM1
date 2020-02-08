load('Exam1.mat','n','N','label','Nc','x');
Sigma2(:,:,1) = [1 0;0 1];Sigma2(:,:,2) = [1 0;0 1]; 
mu(:,1)=[-0.1;0];mu(:,2)=[0.1;0];
y=evalGaussian(x,mu(:,2),Sigma2(:,:,2))./evalGaussian(x,mu(:,1),Sigma2(:,:,1));
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
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma2(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma2(:,:,1)));% - log(gamma);
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
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma2(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma2(:,:,1)));% - log(gamma);
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
title('naive-Bayesian classifier ROC'),
xlabel('P(D = 1|L = 0)False Positive Probability'), ylabel('P(D = 1|L = 1)True Positive Probability'), 
