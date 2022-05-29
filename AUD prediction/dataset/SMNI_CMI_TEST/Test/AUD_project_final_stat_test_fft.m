 Ns=256;
 k=Ns;
 b=0;
 l=0;
 avg_raw=zeros(1,Ns);
 sensor=[];
 numexcelfiles=480;  %file number of the last excel sheet is Data468.csv
 
 yy=1;
 x=1;
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
     sensor=y(1:end,5); 
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
        
     %do fft
     S=fft(avg_raw);
     Pxx=1/(length(avg_raw)*Ns)*abs(S(1:length(avg_raw)/2+1)).^2;
     Pxx(2:end-1)=2*Pxx(2:end-1);
     freq=0:Ns/length(avg_raw):Ns/2;
        
     wave(:,yy)=bandpower(Pxx,freq,[0.5,3.5],'psd'); %delta
     yy=yy+1;
     wave(:,yy)=bandpower(Pxx,freq,[3.5,7.5],'psd');%theta
     yy=yy+1;
     wave(:,yy)=bandpower(Pxx,freq,[7.5,12.5],'psd');%alpha
     yy=yy+1;
     wave(:,yy)=bandpower(Pxx,freq,[12.5,30],'psd');%beta
     yy=yy+1;
     wave(:,yy)=bandpower(Pxx,freq,[30,100],'psd');%gamma
        
     for ii=1:5
         signal(i,x)=abs(sum(wave(:,ii).^2));
         x=x+1;
         signal(i,x)=mean(wave(:,ii));
         x=x+1;
%          signal(i,x)=var(wave(:,ii));
%          x=x+1;
%          signal(i,x)=std(wave(:,ii));
%          x=x+1;
         signal(i,x)=max(wave(:,ii));
         x=x+1;
         signal(i,x)=min(wave(:,ii));
         x=x+1;
%          signal(i,x)=kurtosis(wave(:,ii));
%          x=x+1;
%          signal(i,x)=skewness(wave(:,ii));
     end 
     yy=1;
     x=1;
     sensor=[];
     wave=[];
     k=Ns;
     l=0;
     n=0;
 end
  
  Filename='F:\AUD files for LSTM\MTech project final susma\statistical coeff\fft\Test fft.xlsx';
  xlswrite(Filename,signal,sheet,x2Range);
 
  filename='F:\AUD files for LSTM\MTech project final susma\statistical coeff\fft\labels test fft.xlsx';
  xlswrite(filename,boody_boo,sheet,x3Range);






