function Dates = BusinessDates(numberPayments, FirstPaymentDate, typeStep, intervalSteps, flag)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%numberPayments: is the number of payments we receive or pay in the
%interval we consider
%
%FirstPaymentDate: is the first day we receive or pay
%
%typeSteps: is the type of period we are considering, daily, monthly, ecc.
%
%intervalSteps: is how many typeSteps there are in the interval we want
%
%flag: is useful for the function busdate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%we initialize a vector of numberPayments zeros 
Dates =zeros(numberPayments,1);

%we initialize a vector with the usual holidays in format datenum
holidays=[738885 738943 738886 738905 738944 739340];
weekend=[1 0 0 0 0 0 1];

%we initialize first value of calendar as the first date we must pay the
%coupon
Dates(1)=FirstPaymentDate;

%we do a cycle for all the trimester in the calendar
for i=2:numberPayments

    %we add 3 months on top of the last value of the calendar
    Dates(i)=addtodate(Dates(1),intervalSteps * (i-1),typeStep);
    
    %we save the value of calendar because we don't want to build on top of
    %changed last typeSteps values(see next if)
    
    %we check if the day is a holiday or weekend and if yes we change it to
    %the first business day...
    % flag == 1 ...after
    % flag == -1 ...before
    if ~isbusday(Dates(i),holidays,weekend)

        Dates(i)=busdate(Dates(i),flag,holidays,weekend);

    end
end