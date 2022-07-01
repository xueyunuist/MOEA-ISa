
%%
I=15;
Pro=zeros(1000,I); %保存1000次 的十个分段（1000次相似性的值的和）
D = 1000;
for n=1:1000
    PopDec = zeros(D,D);
    for i=1:D
        a = randperm(D);%生成1到D的随机序列
        PopDec(i,a(1:i)) = 1;
    end
    popNum = sum(PopDec,2);
    for k=1:I
        Jaccard =1 - pdist(PopDec(floor(D/I*(k-1)+1):floor(D/I*k),:), 'jaccard'); %杰卡德相似系数=1-杰卡德距离
        Jaccard = sum(Jaccard)/length(Jaccard);
        Pro(n,k) = Jaccard;
    end
end
p = sum(Pro,1)/1000;
s=1./p;
s=s/sum(s);












