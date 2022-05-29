k=256;
l=0;
Ns=256;
L=xlsread('Data3.csv');
avg_raw=zeros(Ns,1);
sensor=L(:,5);
m=length(sensor)/256;
raw=zeros(1,66);
raw=[];
n=1;
ss=1;
tau=0;
p=0;

% Averagea all electrode at time 0ms, 1ms...
%  while k<=16384
%           for j=l:k-1                  
%               n=mod(j,Ns)+1;
%               avg_raw(n,1)=sensor(j+1);         
%           end
%       l=k;
%       k=k+Ns;       
%  end
%  avg_raw1=avg_raw/m;
%  S=avg_raw;
%  Wave=stat_coeff_of_EEG(S);
%  sample_entrop=samp_entrop(S);
%  [LLE lambda]=lyaprosen(S*100,tau,p);  
%  wave=horzcat(Wave,sample_entrop,LLE);
 
 
 
 selected_channels=[2,6,7,21,22,24,25,28,29,30,31,38,39,44,45,46,48,49,51,52,57,59,61]; %the Sf is result of GA.
    for j=1:23
        num=selected_channels(1,j);
        k=Ns*(num-1)+1;
        p=k+255;
        new_data(:,j)=sensor(k:p,1);
        k=0;
        p=0;
    end
    
 S=new_data(:,23);
 
 
 
 %wavelet function
     waveletFunction = 'db8';
     [C,L] = wavedec(S,5,waveletFunction);
     %Calculation the Details Vectors of every Band :
     D1 = wrcoef('d',C,L,waveletFunction,1); %NOISY
     
     D2 = wrcoef('d',C,L,waveletFunction,2); %gamma
%      Wave=stat_coeff_of_EEG(D2);
%      sample_entrop=samp_entrop(D2);
%      [LLE lambda]=lyaprosen(D2*100,tau,p);  
%      wave=horzcat(wave,Wave,sample_entrop,LLE);
     
     D3 = wrcoef('d',C,L,waveletFunction,3); %Beta
%      Wave=stat_coeff_of_EEG(D3);
%      sample_entrop=samp_entrop(D3);
%      [LLE lambda]=lyaprosen(D3*100,tau,p);  
%      wave=horzcat(wave,Wave,sample_entrop,LLE);
     
     D4 = wrcoef('d',C,L,waveletFunction,4); %alpha
%      Wave=stat_coeff_of_EEG(D4);
%      sample_entrop=samp_entrop(D4);
%      [LLE lambda]=lyaprosen(D4*100,tau,p);  
%      wave=horzcat(wave,Wave,sample_entrop,LLE);
     
     D5 = wrcoef('d',C,L,waveletFunction,5); %theta
%      Wave=stat_coeff_of_EEG(D5);
%      sample_entrop=samp_entrop(D5);
%      [LLE lambda]=lyaprosen(D2*100,tau,p);  
%      wave=horzcat(wave,Wave,sample_entrop,LLE);
     
     A5=wrcoef('a',C,L,waveletFunction,5);   %delta
%      Wave=stat_coeff_of_EEG(A5);
%      sample_entrop=samp_entrop(A5);
     [LLE lambda]=lyaprosen(A5*100,0,0);  
%      wave=horzcat(wave,Wave,sample_entrop,LLE);
 
 

% %multichannel eeg
% for j=1:m
%     raw(:,j)=sensor(n:k,1); 
%     n=k+1;
%     k=k+256;
%     signal=raw(:,j);
%     sample_entrop=samp_entrop(signal);
%     wave(1,ss)=sample_entrop;
%     ss=ss+1;
% end







