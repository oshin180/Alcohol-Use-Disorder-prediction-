function [sample_entrop] = samp_entrop(signal)

Z=length(signal);
mc=2;
ss=1;

rr=0.2*std(signal);

%Split time series and save all templates of length m
for ij=1:Z-mc
    xmi(ij,1:mc)=signal(ij:ij+mc-1);
end
for ij=1:Z-mc+1
    xmj(ij,1:mc)=signal(ij:ij+mc-1);
end

%Save all matches minus the self-match, compute B
for ij=1:length(xmi)
    for jk=1:length(xmj)
        var1(jk,:)=abs(xmi(ij,:)-xmj(jk,:));
        var2(jk,1)=max(var1(jk,1),var1(jk,mc));
        for im=1:length(var2)
            if var2(im)<=rr
                var3(im)=1;
            else
                var3(im)=0;
            end
        end
        var4(ij,1)=sum(var3)-1;    
    end
end
B=sum(var4);

%similar for computing A
mc=mc+1;
for ij=1:Z-mc+1
    xm(ij,1:mc)=signal(ij:ij+mc-1);
end

for ij=1:length(xmi)
    for jk=1:length(xm)
        Var1(jk,1)=abs(xmi(ij,1)-xm(jk,1));
        Var1(jk,2)=abs(xmi(ij,2)-xm(jk,2));
        Var1(jk,3)=abs(xm(ij,3)-xm(jk,3));
        
        Var2(jk,1)=max(Var1(jk,1),Var1(jk,2));
        Varr2(jk,1)=max(Var2(jk,1),Var1(jk,mc));
        for im=1:length(Varr2)
            if Varr2(im)<=rr
                Var3(im)=1;
            else
                Var3(im)=0;
            end
        end
        Var4(ij,1)=sum(Var3)-1;    
    end
end
A=sum(Var4);

sample_entrop=-log(A/B);

return 
end


