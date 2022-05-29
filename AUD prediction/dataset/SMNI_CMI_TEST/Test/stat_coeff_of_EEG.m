function [wave] = stat_coeff_of_EEG(s)
S=s;
yy=1;
dave=[];
dave(1,yy)=abs(sum(S.^2));  %power
yy=yy+1;
dave(1,yy)=mean(S); %mean
yy=yy+1;
dave(1,yy)=var(S); %variance
yy=yy+1;
dave(1,yy)=std(S); %standard deviation
yy=yy+1;
dave(1,yy)=max(S); %max amp
yy=yy+1;
dave(1,yy)=min(S); %min amp
yy=yy+1;
dave(1,yy)=kurtosis(S); %kurtosis
yy=yy+1;
dave(1,yy)=skewness(S); %skewness
yy=yy+1;
dave(1,yy)=mode(S); %mode
yy=yy+1;
dave(1,yy)=median(S); %median
wave=dave;
return
end

