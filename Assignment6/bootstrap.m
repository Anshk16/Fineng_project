function [dates, discounts]=bootstrap(datesSet, ratesSet)
DayForm='long';

%we import settlement date
Settlement_date = datesSet.settlement;

%we compute the number of years between settlement date and last swap
n=year(datetime(datesSet.swaps(end), 'ConvertFrom', 'datenum'))-year(datetime(Settlement_date, 'ConvertFrom', 'datenum')); %number of years

%we initialize a vector of n+1 zeros for all the years bewtween 2023 and
%2073
calendar=zeros(n+1,1);%per tutti gli anni dal 2023 al 2073

%we initialize a vector with the usual holidays in format datenum
holidays=[738885 738943 738886 738905 738944];
weekend=[1 0 0 0 0 0 1];

%we initialize first value of calendar as the settlement date
calendar(1)=Settlement_date;
temp=calendar(1);

%we do a cycle for all the years in the calendar
for i=2:n+1

    %we add 1 year on top of the last value of the calendar
    calendar(i)=addtodate(temp,1,'year');
    
    %we save the value of calendar because we don't want to build on top of
    %changed last year values(see next if)
    temp=calendar(i);
    
    %we check if the day is a holiday or weekend and if yes we change it to
    %the first buisness day after
    if ~isbusday(calendar(i),holidays,weekend)

        calendar(i)=busdate(calendar(i),1,holidays,weekend);

    end
end
clear temp

%to check what we found
matlab.datetime.compatibility.convertDatenum(calendar);

%we initialize the dates and discounts outputs
dates=Settlement_date;
discounts=1; %first DCF is 1

%% DEPOS

%now we focus on the depos, we want to consider only depos whose expiry is
%smaller than the first settle of the futures

cont_depos=0;
while (datesSet.depos(cont_depos+1)<=datesSet.futures(1,1))
    cont_depos=cont_depos+1;
end

%we add those dates to our output vector dates
dates=[dates;datesSet.depos(1:cont_depos)];

%mid prices of depos
mid_depos=mean([ratesSet.depos(1:cont_depos,1) ratesSet.depos(1:cont_depos,2)],2);

%we update also discounts output vector with act/360 (2)
discounts=[discounts;1./(1+yearfrac(Settlement_date,datesSet.depos(1:cont_depos),2).*mid_depos)];

%% FUTURES

%now we focus on the futures, again we consider only the futures whose
%expiry is smaller than the first settle of the swaps
cont_futures=0;
while (datesSet.futures(cont_futures+1,2)<=datesSet.swaps(1,1))
    cont_futures=cont_futures+1;
end

%we compute the mid prices of futures L(t0,ti,ti+1)
mid_futures=mean([ratesSet.futures(1:cont_futures,1) ratesSet.futures(1:cont_futures,2)],2);

%we also compute the forward disount factors B(t0,ti,ti+1), again with
%act/360 (2) 
fwd_DCF=1./(1+yearfrac(datesSet.futures(1:cont_futures,1),datesSet.futures(1:cont_futures,2),2).*mid_futures);

%first future: we check if its settle date is the same date as the last
%depos, if yes they have the same DCF (B(t0,ti)) and we can calculate
%B(t0,ti+1)=B(t0,ti)*B(t0,ti,ti+1), otherwise we obtain the B(t0,ti) by
%interpolating with the depos we didn't consider

if datesSet.futures(1,1)==datesSet.depos(cont_depos)

    DCF_t0_ti=discounts(dates==datesSet.futures(1,1));

else 
    
    %we initialize the interp date
    interp_date=datesSet.futures(1,1);

    %we create a new set of dates including the first depos we didn't
    %include before and all those we already have
    dates=[dates;datesSet.depos(cont_depos+1)];

    %we compute the mid price of the above cited depos
    mid_depos_prime=mean([ratesSet.depos(cont_depos+1,1) ratesSet.depos(cont_depos+1,2)],2);

    %we compute the discount of the above cited depos
    discount_to_add=1/(1+yearfrac(dates(1),datesSet.depos(cont_depos+1),2)*mid_depos_prime);

    %we create a new set of discounts including the above cited depos and
    %all those we already have
    discounts=[discounts;discount_to_add];

    %lastly, we interpolate the DCF_t0_ti by interpolating the
    %corresponding zero rates
    DCF_t0_ti=zRatesInterp(dates,discounts,interp_date);


end

%we update dates and discounts with the expiry for the first future and the
%B(t0,ti+1)=B(t0,ti)*B(t0,ti,ti+1)
 
dates=[dates; datesSet.futures(1,2)];
discounts=[discounts;DCF_t0_ti*fwd_DCF(1)];

%we preallocate the remaining dates and discounts
len=length(discounts);

discounts=[discounts;zeros(cont_futures-1,1)];
dates=[dates;zeros(cont_futures-1,1)];

for i=2:cont_futures
    
    %we initialize settle and expiry for each future
    t_i=datesSet.futures(i,1); %ti
    t_i_plus=datesSet.futures(i,2); %ti+1

    %we compute DCF_t0_ti for each i-th future, i.e for the settle
    DCF_t0_ti=zRatesInterp(dates(1:len+i-2),discounts(1:len+i-2),t_i);
    
    %we compute DCF_t0_ti_plus for each i-th future, i.e for the expiry
    DCF_t0_ti_plus=DCF_t0_ti*fwd_DCF(i);

    %we update dates and discounts
    discounts(len+i-1)=DCF_t0_ti_plus;
    dates(len+i-1)=t_i_plus;
end

%% SWAPS

%we build a complete set of swaps

%we compute the number of years between settlement date and last swap
n=year(datetime(datesSet.swaps(end), 'ConvertFrom', 'datenum'))-year(datetime(Settlement_date, 'ConvertFrom', 'datenum')); %number of years

%we initialize a vector of n+1 zeros for all the years bewtween 2023 and
%2073
calendar=zeros(n+1,1);%per tutti gli anni dal 2023 al 2073

%we initialize a vector with the usual holidays in format datenum
holidays=[738885 738943 738886 738905 738944];
weekend=[1 0 0 0 0 0 1];

%we initialize first value of calendar as the settlement date
calendar(1)=Settlement_date;
temp=calendar(1);

%we do a cycle for all the years in the calendar
for i=2:n+1

    %we add 1 year on top of the last value of the calendar
    calendar(i)=addtodate(temp,1,'year');
    
    %we save the value of calendar because we don't want to build on top of
    %changed last year values(see next if)
    temp=calendar(i);
    
    %we check if the day is a holiday or weekend and if yes we change it to
    %the first buisness day after
    if ~isbusday(calendar(i),holidays,weekend)

        calendar(i)=busdate(calendar(i),1,holidays,weekend);

    end
end
clear temp

%to check what we found
matlab.datetime.compatibility.convertDatenum(calendar);

%we only consider swaps from the first given swaps, the rest we are not
%gonna add them to the discounts
cont_swaps=0;
while (calendar(cont_swaps+1)<datesSet.swaps(1,1))
    cont_swaps=cont_swaps+1;
end

%since we considered the settlement date 
%cont_swaps=cont_swaps-1;

%we initialize the BPV value
BPV=0;
for j=1:cont_swaps-1
    Bj=zRatesInterp(dates,discounts,calendar(j+1));
    BPV=BPV+Bj*yearfrac(calendar(j),calendar(j+1),6);
end

%now we spline interpole the full set of swaps

avg_swaps=0.5*(ratesSet.swaps(:,1)+ratesSet.swaps(:,2));
avg_y=-log(avg_swaps)./yearfrac(Settlement_date,datesSet.swaps,3);
%the i+2 is because we want it to start from the first swap
interpolated_swaps=spline(datesSet.swaps,avg_swaps,calendar(cont_swaps+1:end));

%we preallocate the memory for dates and discounts
len=length(dates);
discounts=[discounts;zeros(length(calendar)-cont_swaps,1)];
dates=[dates;zeros(length(calendar)-cont_swaps,1)];

%now we compute the rest of the DCF's
for i=cont_swaps+1:length(calendar)
    
    %time interval between ti-1 and ti
    delta=yearfrac(calendar(i-1),calendar(i),6);

    dates(len-cont_swaps+i)=calendar(i);
    Bi=(1-interpolated_swaps(i-cont_swaps)*BPV)/(1+interpolated_swaps(i-cont_swaps)*delta);
    discounts(len-cont_swaps+i)=Bi;

    BPV=BPV+Bi*delta;
end
end

