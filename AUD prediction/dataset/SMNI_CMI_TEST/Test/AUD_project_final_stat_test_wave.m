 Ns=256;
 k=Ns;
 b=0;
 l=0;
 avg_raw=zeros(Ns,1);
 sensor=[];
 wave=[];
 final_wave=[];
 numexcelfiles=480;  %file number of the last excel sheet is Data480.csv
 yy=0;
 gg=0;
 x2Range='2';
 x3Range='A';
 sheet=1;
 base='Data_';
 Name= 'F:\';
 h=0;
 currentfolder=cd;
 for i=1:numexcelfiles    
     baseFileName='Data';
     extension=num2str(i);
     extension1=str2double(extension);
     filename=strcat(baseFileName,extension,'.csv');
     name='currentfolder/filename';
     avg_raw=zeros(Ns,1);   
     [y,TXT,RAW]=xlsread(filename);
     sensor=y(:,5);
     m=length(sensor)/256; %64
     boody_boo(i,1)=TXT(2,5);   
     while k<=16384
         for j=l:k-1                  
             n=mod(j,Ns)+1;
             avg_raw(n,1)=avg_raw(n,1)+sensor(j+1);         
         end
     l=k;
     k=k+Ns;       
     end
     avg_raw=avg_raw/m;
     S=avg_raw;
     Wave=stat_coeff_of_EEG(S);
     sample_entrop=samp_entrop(S);
     wave=horzcat(Wave,sample_entrop);
     %wavelet function
     waveletFunction = 'db8';
     [C,L] = wavedec(S,5,waveletFunction);
     %Calculation the Details Vectors of every Band :
     D1 = wrcoef('d',C,L,waveletFunction,1); %NOISY
     for iii=2:5   %2 3 4 5
         signal = wrcoef('d',C,L,waveletFunction,iii); %D2--gamma, %D3--beta, %D4--alpha, %D5--theta
         Wave=stat_coeff_of_EEG(signal);
         wave=horzcat(wave,Wave);
         sample_entrop=samp_entrop(signal);
         wave=horzcat(wave,sample_entrop);
         
     end
     signal=wrcoef('a',C,L,waveletFunction,5); %A5--delta
     Wave=stat_coeff_of_EEG(signal);
     wave=horzcat(wave,Wave);
     sample_entrop=samp_entrop(signal);
     wave=horzcat(wave,sample_entrop);
     
     final_wave=vertcat(final_wave,wave);
   
     
     avg_raw=zeros(Ns,1);
     sensor=[];
     k=Ns;
     l=0;
     n=0;
     yy=0;
     gg=0;
     wave=[];
     Wave=[];
     sample_entrop=0;
     
 end
 
Filename='F:\AUD files for LSTM\MTech project final susma\statistical coeff\Test wave.xlsx';
xlswrite(Filename,final_wave,sheet,x2Range);
  
filename='F:\AUD files for LSTM\MTech project final susma\statistical coeff\labels test wave.xlsx';
xlswrite(filename,boody_boo,sheet,x3Range);






