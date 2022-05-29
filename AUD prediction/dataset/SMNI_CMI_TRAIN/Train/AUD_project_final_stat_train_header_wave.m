
wave{1,1}='Raw';
wave{1,2}='Gamma';
wave{1,3}='Beta';
wave{1,4}='Alpha';
wave{1,5}='Theta';
wave{1,6}='Delta';

param{1,1}='_power';
param{1,2}='_mean';
param{1,3}='_variance';
param{1,4}='_std-deviation';
param{1,5}='_max-amp';
param{1,6}='_min-amp';
param{1,7}='_kurtosis';
param{1,8}='_skewness';
param{1,9}='_mode';
param{1,10}='_median';
param{1,11}='_sample_entropy';


m=1;

    for j=1:6
        for k=1:11
            name{1,m}=strcat(wave{1,j},param{1,k});
            m=m+1;
        end
    end

x3Range='A1';
sheet=1;

Filename='F:\AUD files for LSTM\MTech project final susma\statistical coeff\Train wave.xlsx';
xlswrite(Filename,name,sheet,x3Range);