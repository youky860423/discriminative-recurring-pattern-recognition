function [ TPR,FPR,AUC ] = ROCandAUC( score,y )
%ROCANDAUC Summary of this function goes here
%   giving the test score, we provide the TPR and FPR, 
%   and the corresponding auc and roc.
B=size(y,2);
n=size(y,1);
FPR=zeros(n,B);
TPR=zeros(n,B);
AUC=zeros(B,1);
AUC2=zeros(B,1);
for b=1:B
    [scoresorted, idx]=sort(score(:,b));
    ysorted=y(idx,b);
    nmius=sum(ysorted==2);
    nplus=sum(ysorted==1);
    for i=1:n
        FPR(i,b)=sum(ysorted(i:n)==2)/nmius;
        TPR(i,b)=sum(ysorted(i:n)==1)/nplus;
    end
    %%%%%%%%ranking method for auc%%%%%%
    list=unique(scoresorted);
    rank=(1:n);
    for i=1:length(list)
        num=sum(scoresorted==list(i));
        if num>1
            rank(scoresorted==list(i))=sum(rank(scoresorted==list(i)))/num;
        end
    end
    R1=sum(rank(ysorted==1));
    U1=R1-nplus*(nplus+1)/2;
    AUC(b)=U1/(nplus*nmius);   
    %%%%%%%%%%%area method for auc%%%%%%%
    for i=1:size(TPR,1)-1
        AUC2(b)=AUC2(b)+(FPR(i,b)-FPR(i+1,b))*(TPR(i+1,b)+TPR(i,b))/2;
    end
end
AUC=mean(AUC);
%%%%%%%%%%%area method for average auc%%%%%%%
FPR=mean(FPR,2);
TPR=mean(TPR,2);
AUCavg=0;
for i=1:length(TPR)-1
    AUCavg=AUCavg+(FPR(i)-FPR(i+1))*(TPR(i+1)+TPR(i))/2;
end



end

