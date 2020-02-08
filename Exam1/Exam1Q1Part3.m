mu(:,1) = [-0.1;0]; 
Sigma(:,:,1) = [1 -0.9;-0.9 1];
mu(:,2) = [0.1;0]; 
Sigma(:,:,2) = [1 0.9;0.9 1]; 

load('Exam1.mat','n','N','label','Nc','x');
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly

ysort=sort(yLDA);
epsilon=0.0000000000000000000000000000000000000000000000000000000001;

%calculate tau
tau=zeros(1,N+1);
tau(1)= ysort(1)-epsilon;
tau(N+1)=ysort(N)+epsilon;
for h=1:N-1
    tau(h+1)=(ysort(h)+ysort(h+1))/2;
end
%calculate p_error
p_error=zeros(1,N+1);
for k=1:N+1
    decisionLDA = (yLDA >= tau(k));

ind00 = find(decisionLDA==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decisionLDA==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decisionLDA==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decisionLDA==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive

p_error(k)= [p10,p01]*Nc'/N; % probability of error, empirically estimated
end
%draw picture to find min P(error) and tau 
figure(1),clf,
plot(tau,p_error,'.');
title('P(error) and tau'),
xlabel('tau'), ylabel('p_error'), 
[min_perror,min_perror_index]=min(p_error);
tau_min_error=tau(min_perror_index);

%draw ROC
for a=1:N+1
    decisionLDA = (yLDA >= tau(a));

ind10 = find(decisionLDA==1 & label==0); p10(a) = length(ind10)/Nc(1); % probability of false positive
ind11 = find(decisionLDA==1 & label==1); p11(a) = length(ind11)/Nc(2); % probability of true positive

end
decisionLDA = (yLDA >= tau_min_error);

ind10 = find(decisionLDA==1 & label==0); p102 = length(ind10)/Nc(1); % probability of false positive
ind11 = find(decisionLDA==1 & label==1); p112 = length(ind11)/Nc(2); % probability of true positive
figure(2),clf
plot(p10,p11);hold on,
plot(p102,p112,'og');
axis equal,
title('Fisher LDA ROC'),
xlabel('P(D = 1|L = 0)False Positive Probability'), ylabel('P(D = 1|L = 1)True Positive Probability'), 