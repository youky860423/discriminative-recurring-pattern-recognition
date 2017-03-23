function [TPR,FPR,AUC,bagacc] = testing( w,X,X2,Y,y,option,winsize)
%TESTING bag level prediction vs. instance level prediction
B=size(X,3);
if strcmp(option.priorType,'conv')
    wtx=WTX(X,w,option.addone,option.conv);
else
    for b=1:B
        wtx(:,b)=X2{b}'*w;
    end
end
%%%%%%%%%%%fixing delay%%%%%%%%
cnt1=0;
truelab=2*ones(size(wtx));
for b=1:B
    truelab(ceil(winsize/2):end-floor(winsize/2),b)=(1:2)*y{b};
    idx=find(truelab(:,b)==1,1);
    if ~isempty(idx)
        [~,idx2]=max(wtx(:,b));
        cnt1=cnt1+1;
        delay1(cnt1)=idx-idx2;
    end
end
wtx=circshift(wtx,mode(delay1));
wtx_vec=wtx(:);
truelab_vec=truelab(:);
[TPR,FPR,AUC] = ROCandAUC(wtx_vec,truelab_vec);
% figure(5)
% plot(FPR,TPR)
% title('ROC');
% pause()
%%%%%union rule to compute the bag accuracy%%%%%%
prior=zeros(size(wtx));
prior(wtx<0)=exp(wtx(wtx<0))./(1+exp(wtx(wtx<0)));
prior(wtx>=0)=1./(1+exp(-wtx(wtx>=0))); 
baghit=(Y'==any(prior>=0.5,1));
bagacc=sum(baghit)/B;

end

