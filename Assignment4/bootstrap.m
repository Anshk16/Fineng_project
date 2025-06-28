function [dates, discounts]=bootstrap(datesSet, ratesSet)
dates=datesSet;
rates=ratesSet;

DayForm='long';
Settlement_date = dates.settlement;
calendar=zeros(51,1);%per tutti gli anni dal 2023 al 2073
calendar(1)=Settlement_date;
cont=2;
temp=calendar(1);

for i=2:51
    cont=cont+1;
    if mod(cont,4)==0
        calendar(i)=temp+366;
    else
        calendar(i)=temp+365;
    end

    temp=calendar(i);

    if weekday(calendar(i),DayForm)==7
        calendar(i)=calendar(i)+2;
    elseif weekday(calendar(i),DayForm)==1
        calendar(i)=calendar(i)+1;
    end
end

%per controcheck
matlab.datetime.compatibility.convertDatenum(calendar);

%short rates by depos with Act 360
mean_depos=(0.5*(rates.depos(:,1)+rates.depos(:,2)));
short_dt_DCF=yearfrac(calendar(1),dates.depos,2);
short_dt_ZR=yearfrac(calendar(1),dates.depos,3);
short_DCF=1./(1+short_dt_DCF.*mean_depos);
short_zRates=-log(short_DCF)./short_dt_ZR;

%plot(short_DCF(1:4))

%mid rates by futures with Act 360 
fwd_rates=0.5*(rates.futures(:,1)+rates.futures(:,2));%non più percentuale
mid_dt=yearfrac(dates.futures(:,1),dates.futures(:,2),2);
fwd_DCF=1./(1+mid_dt.*fwd_rates);

mid_ZR=[];
mid_DCF=[];
mid_dates=[];
prova=[];

%interpolate first future_DCF
X=yearfrac(calendar(1),dates.depos(3:4),3);
XQ=yearfrac(calendar(1),dates.futures(1,1),3);
YQ=interp1(X,short_zRates(3:4),XQ);

%15-mar-23
mid_ZR=[mid_ZR;YQ];
mid_DCF=[mid_DCF;exp(-YQ*XQ)];
mid_dates=[mid_dates;dates.futures(1,1)];
%15-giu-23
mid_DCF=[mid_DCF;exp(-YQ*XQ)*fwd_DCF(1)];
mid_ZR=[mid_ZR;-log(mid_DCF(end))./yearfrac(calendar(1),dates.futures(1,2),3)];
mid_dates=[mid_dates;dates.futures(1,2)];
prova=[prova;mid_DCF(end)];

%interpolate second future_DCF
X=yearfrac(calendar(1),dates.depos(5:6),3);
XQ=yearfrac(calendar(1),dates.futures(2,1),3);
YQ=interp1(X,short_zRates(5:6),XQ);

%21_giu-23
mid_ZR=[mid_ZR;YQ];
mid_DCF=[mid_DCF;exp(-YQ*XQ)];
mid_dates=[mid_dates;dates.futures(2,1)];
%21-set-23
mid_DCF=[mid_DCF;exp(-YQ*XQ)*fwd_DCF(2)];
mid_ZR=[mid_ZR;-log(mid_DCF(end))./yearfrac(calendar(1),dates.futures(2,2),3)];
mid_dates=[mid_dates;dates.futures(2,2)];
prova=[prova;mid_DCF(end)];

%interpolate third future_DCF
X=yearfrac(calendar(1),dates.futures(2,:),3);
XQ=yearfrac(calendar(1),dates.futures(3,1),3);
YQ=interp1(X,mid_ZR(end-1:end),XQ);

%20-sett-23
mid_ZR=[mid_ZR;YQ];
mid_DCF=[mid_DCF;exp(-YQ*XQ)];
mid_dates=[mid_dates;dates.futures(3,1)];
%20-dic-23
mid_DCF=[mid_DCF;exp(-YQ*XQ)*fwd_DCF(3)];
mid_ZR=[mid_ZR;-log(mid_DCF(end))./yearfrac(calendar(1),dates.futures(3,2),3)];
mid_dates=[mid_dates;dates.futures(3,2)];
prova=[prova;mid_DCF(end)];

% fourth future DCF

XQ=yearfrac(calendar(1),dates.futures(4,1),3);
YQ=mid_ZR(end);


%20-mar-24
mid_DCF=[mid_DCF;exp(-YQ*XQ)*fwd_DCF(4)];
mid_ZR=[mid_ZR;-log(mid_DCF(end))./yearfrac(calendar(1),dates.futures(4,2),3)];
mid_dates=[mid_dates;dates.futures(4,2)];
prova=[prova;mid_DCF(end)];

%we interpolate zRate for 1year 
X=yearfrac(calendar(1),dates.futures(4,:),3);
XQ=yearfrac(calendar(1),calendar(2),3);
zRate_1yr=interp1(X,mid_ZR(end-1:end),XQ);


% fifth future DCF

XQ=yearfrac(calendar(1),dates.futures(5,1),3);
YQ=mid_ZR(end);


%20-giu-24
mid_DCF=[mid_DCF;exp(-YQ*XQ)*fwd_DCF(5)];
mid_ZR=[mid_ZR;-log(mid_DCF(end))./yearfrac(calendar(1),dates.futures(5,2),3)];
mid_dates=[mid_dates;dates.futures(5,2)];
prova=[prova;mid_DCF(end)];

%interpolate sixth future_DCF
X=yearfrac(calendar(1),dates.futures(5,:),3);
XQ=yearfrac(calendar(1),dates.futures(6,1),3);
YQ=interp1(X,mid_ZR(end-1:end),XQ);

%19-giu-24
mid_ZR=[mid_ZR;YQ];
mid_DCF=[mid_DCF;exp(-YQ*XQ)];
mid_dates=[mid_dates;dates.futures(6,1)];
%19-set-24
mid_DCF=[mid_DCF;exp(-YQ*XQ)*fwd_DCF(6)];
mid_ZR=[mid_ZR;-log(mid_DCF(end))./yearfrac(calendar(1),dates.futures(6,2),3)];
mid_dates=[mid_dates;dates.futures(6,2)];
prova=[prova;mid_DCF(end)];

%interpolate seventh future_DCF
X=yearfrac(calendar(1),dates.futures(6,:),3);
XQ=yearfrac(calendar(1),dates.futures(7,1),3);
YQ=interp1(X,mid_ZR(end-1:end),XQ);

%19-giu-24
mid_ZR=[mid_ZR;YQ];
mid_DCF=[mid_DCF;exp(-YQ*XQ)];
mid_dates=[mid_dates;dates.futures(7,1)];
%19-set-24
mid_DCF=[mid_DCF;exp(-YQ*XQ)*fwd_DCF(7)];
mid_ZR=[mid_ZR;-log(mid_DCF(end))./yearfrac(calendar(1),dates.futures(7,2),3)];
mid_dates=[mid_dates;dates.futures(7,2)];
prova=[prova;mid_DCF(end)];

% SWAP
calendar_temp=calendar(2:end);
X=yearfrac(Settlement_date,dates.swaps,3);
Y=0.5*(rates.swaps(:,1)+rates.swaps(:,2));
XQ=yearfrac(Settlement_date,calendar_temp,3);
interpolated_swaps=spline(X,Y,XQ);

DCF=zeros(length(calendar_temp),1);
DCF(1)=exp(-zRate_1yr*yearfrac(calendar(1),calendar(2),3));

for i=2:50
    SIR=interpolated_swaps(i);
    DCF(i)=(1-SIR*sum(yearfrac(calendar_temp(1:i-1),calendar_temp(2:i),6).*DCF(1:i-1)))./(1+yearfrac(calendar_temp(i-1),calendar_temp(i),6)*SIR);

end

long_DCF=DCF;

%PROVA1=sort([1;short_DCF(1:4);prova;long_DCF],'descend');
%PROVA2=sort([calendar(1);dates.depos(1:4);dates.futures(1:7,2);calendar(2:end)],'ascend');

discounts=sort([1;short_DCF(1:4);prova;long_DCF],'descend');
dates=sort([calendar(1);dates.depos(1:4);dates.futures(1:7,2);calendar(2:end)],'ascend');
end

