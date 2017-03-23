clear;
close all
B=200;
T=500;
pos_index=randi(T-1,B,1);
A=10;
pos_signal=zeros(T,B);
win=300;
%%%%%%%spike noise%%%%%%%
% spike_index=randi(T-1,B/2,1);
% sA_pos=100+10*randn(B/2,1);
for b=1:B
    pos_signal(1:pos_index(b),b)=A+randn(1,pos_index(b));
    pos_signal(pos_index(b)+1:end,b)=-A+randn(1,T-pos_index(b));
    X(1,:,b)=pos_signal(:,b)-mean(pos_signal(:,b));
%     if b<=B/2
%         X(1,spike_index(b),b)=sA_pos(b);
%     end
    y{b}=[zeros(1,T);ones(1,T)];
    y{b}(1,pos_index(b))=1;
    y{b}(2,pos_index(b))=0;
    X_new=[X(:,1,b)*ones(1,ceil(win/2)-1) ,X(:,:,b), X(:,end,b)*ones(1,floor(win/2))];
        for j=1:T
            temp=X_new(:,j:j+win-1)';
            X2{b}(:,j)=[temp; 1];
        end
    Y(b,1)=1;
end
for b=1:B/2
    X(:,:,B+b)=randn(1,T);
    X_new=[zeros(1,ceil(win/2)-1) ,X(:,:,B+b), zeros(1,floor(win/2))];
        for j=1:T
            temp=X_new(:,j:j+win-1)';
            X2{B+b}(:,j)=[temp; 1];
        end
    y{B+b}=[zeros(1,T);ones(1,T)];
    Y(B+b,1)=0;
end
% %%%%%%making spike noise in the negative data%%%%%%%%
% neg_index=randi(T-1,B/2,1);
% sA=100+10*randn(B/2,1);
% for b=1:B/2
%     X(:,:,1.5*B+b)=randn(1,T);
%     X(:,neg_index(b),1.5*B+b)=sA(b);
%     X_new=[zeros(1,ceil(win/2)-1) ,X(:,:,1.5*B+b),zeros(1,floor(win/2))];
%         for j=1:T
%             temp=X_new(:,j:j+win-1)';
%             X2{1.5*B+b}(:,j)=[temp; 1];
%         end
%     y{1.5*B+b}=[zeros(1,T);ones(1,T)];
%     Y(1.5*B+b,1)=0;
% end

for b=1:size(X,3)
    figure(1)
    plot(X(:,:,b));
    title(num2str(Y(b)))
    pause()
end
save('synsignal_1d.mat','X','X2','y','Y','win');