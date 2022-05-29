 Ns=256;
 p=1;
 k=Ns;
 n=1;
 numexcelfiles=480;  %file number of the last excel sheet is Data468.csv
 currentfolder=cd;
 final_dataset=[];
 q=1;
 labels={'gamma';'beta';'alpha';'theta';'delta'};
 fin_labels=[];
 
 for i=1:numexcelfiles
     fin_labels=vertcat(fin_labels,labels);
 end
 
 
 for i=1:numexcelfiles    
     baseFileName='Data';
     extension=num2str(i);
     extension1=str2double(extension);
     filename=strcat(baseFileName,extension,'.csv');
     name='currentfolder/filename';
     [y,TXT,RAW]=xlsread(filename);
     sensor=y(:,5);
     p=1;
     k=Ns;
     value=[];
     for j=1:64
         if k<=16384
         value(1:Ns,j)=sensor(p:k,1);
         p=k+1;
         k=k+256;
         end
     end
     
     for j=1:64
         waveletFunction = 'db8';
         S=value(:,j);
         [C,L] = wavedec(S,5,waveletFunction);
         D1 = wrcoef('d',C,L,waveletFunction,1); %NOISY
         for k=2:5
             signal=wrcoef('d',C,L,waveletFunction,k);
             m=mean(signal);
             dataset(n,j)=m;
             n=n+1;
         end
         signal=wrcoef('a',C,L,waveletFunction,5);
         m=mean(signal);
         dataset(n,j)=m;
         n=1;         
     end
     
     final_dataset=vertcat(final_dataset,dataset);
     dataset=[];     
 end
 
 %the result you run in GA
 







