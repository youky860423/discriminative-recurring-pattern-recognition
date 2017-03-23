function [ wnew,rllharr,wnorm] = LocOptimization( w,X,X2,Y,maxiter,option)
%LOCOPTIMIZATION Summary of this function goes here
%   Detailed explanation goes here

options2 = struct('TolX',1e-10,'TolFun',1e-10,'GradObj','on','Display','off','LargeScale','off','HessUpdate','lbfgs','GoalsExactAchieve',0,'MaxIter',maxiter);
% %%%%%%%%%%%checking gradient%%%%%%%%%%
% [fval,grad] = fcal(w,X,X2,Y,option);
% tmp7=randn(size(w));
% w2=w+1e-7*tmp7;
% [fval2,~] = fcal(w2,X,X2,Y,option);
% ((fval2-fval)/1e-7 - sum(sum(grad.*tmp7)))/(sum(sum(grad.*tmp7)))

[wnew,rllharr,enough,output,grad] = fminlbfgs(@(w)fcal(w,X,X2,Y,option),w,options2);
wnorm=norm(wnew(1:end-1,:));
% % %%%%%%%%%%%%display dictionary words after minimization%%%%%%
% %     w=wnew;
% %     wtx=WconvX(X,w,option.addone,option.conv);
% %     F=size(X,1);
% %     if option.addone
% %         width=(size(w,1)-1)/F;
% %     else
% %         width=size(w,1)/F;
% %     end
% %     figure(1)
% %     for c=1:size(w,2)
% %         if option.addone
% %             words=reshape(w(1:end-1,c),width,[]);
% %         else 
% %             words=reshape(w(:,c),width,[]);
% %         end
% %         if F>1
% %             subplot(size(w,2),1,c);imagesc(words');colormap gray; colorbar;
% %         else
% %             subplot(size(w,2),1,c);plot(words')
% %         end
% %     end
% %         title(['pattern words with lambda',num2str(option.lambda)]);
% %         prior=Prior(wtx);
% %                
% %         figure(2)
% %         for aa=1:5
% %             subplot(5,1,aa);imagesc(prior(:,:,aa)');
% %             title('prior prob');
% %         end
% %         figure(3)
% %         post=Post(wtx,Y);
% %         for aa=1:5
% % %             posterior(:,1:end-1,aa)=post(:,1:end-1,aa)*Y(aa);
% %             subplot(5,1,aa);imagesc(post(:,:,aa)');
% %             title('posterior prob');
% %         end
% %         figure(4)
% %         for aa=1:5
% %             subplot(5,1,aa);imagesc(prior(:,:,aa)'-post(:,:,aa)',[-1 1]);
% %             title('difference');
% %         end
% %         figure(5)
% %         for aa=1:5
% %             subplot(5,1,aa);
% %             if F>1
% %                 imagesc(X(:,:,aa));colormap gray;
% %             else
% %                 plot(X(1,:,aa));
% %             end
% %             title(num2str(Y(aa,:)));
% %         end
% %         pause()
end

function [fval,grad] = fcal(w,X,X2,Y,option)
B=size(X,3);
if strcmp(option.priorType,'conv')
    wtx=WTX(X,w,option.addone,option.conv);
else
    for b=1:B
        wtx(:,b)=X2{b}'*w;
    end
end
wtx_max=max(wtx,[],1);
wtx_minus_max=wtx-ones(size(wtx,1),1)*wtx_max;
post=exp(wtx_minus_max)./(ones(size(wtx,1),1)*sum(exp(wtx_minus_max),1));
tmp=zeros(size(wtx));
tmp(wtx<0)=log(1+exp(wtx(wtx<0)));
tmp(wtx>=0)=log(1+exp(-wtx(wtx>=0)))+wtx(wtx>=0);
rll_part2=log(sum(exp(wtx_minus_max),1))+wtx_max;
fval=mean(sum(tmp,1)-rll_part2.*Y);
if option.addone
    tmpw=w(1:end-1,:);
else
    tmpw=w;
      
end
fval=fval+0.5*option.lambda*sum(sum(tmpw.^2));
if nargout > 1
    grad=gradient(wtx,post,w,X,X2,Y,option);
%     grad=grad(:);
end

% if nargout > 2
%     hess=Hessian(wtx,X2);
% end

end

function grad=gradient(wtx,post_weight,w,X,X2,Y,option)
prior=zeros(size(wtx));
prior(wtx<0)=exp(wtx(wtx<0))./(1+exp(wtx(wtx<0)));
prior(wtx>=0)=1./(1+exp(-wtx(wtx>=0)));
Activations=prior-post_weight.*(ones(size(wtx,1),1)*Y);
% Activations=prior;%post_weight.*(ones(size(wtx,1),1)*Y);
F=size(X,1);
B=size(X,3);
if (option.addone)
    width=(size(w,1)-1)/F;
else
    width=size(w,1)/F;
end
tempgrad=zeros(size(w,1),B);
if strcmp(option.priorType,'conv')
    tempgrad = PTW(X,Activations,width,option.addone,option.conv);%conv 1, fft 0
else
    for b=1:B
        tempgrad(:,b) = X2{b}*Activations(:,b);
    end
end
grad = mean(tempgrad,2);
if option.addone
    grad(1:end-1,:) = option.lambda*w(1:end-1,:) + grad(1:end-1,:);
    %%%%%%%adding dc constraint%%%%%%%%%
    if option.dc
        grad(1:end-1,:) = grad(1:end-1,:) - mean(grad(1:end-1,:));
    end
else
    grad = option.lambda*w + grad;
    %%%%%%%adding dc constraint%%%%%%%%%
    if option.dc
        grad = grad - mean(grad);
    end
end

end

function hessian=Hessian(wtx,X2)
prior=zeros(size(wtx));
prior(wtx<0)=exp(wtx(wtx<0))./(1+exp(wtx(wtx<0)));
prior(wtx>=0)=1./(1+exp(-wtx(wtx>=0)));
temphess=zeros(size(w,1),size(w,1),B);
for b=1:B
    temphess(:,:,b) = (X2{b}.*(ones(size(X2{b},1),1)*prior(:,b)'))*...
                (((1-prior(:,b))*ones(1,size(X2{b},1))).*X2{b});
end
hessian = mean(temphess,3);

end

