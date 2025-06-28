function [DCF_t0_ti] = zRatesInterp(dates,discounts,interp_dates)

% Interpolate discount factors for given dates.
% 
%     Parameters:
%     - dates: List of Dates for which discounts are known.
%     - discounts: List of discount factors corresponding to the dates.
%     - interp_dates: List of Dates for which discount factors need to be interpolated.
% 
%     Returns:
%     - DCF_t0_ti: Array of interpolated discount factors.
 m=length(interp_dates);
 DCF_t0_ti=zeros(m,1);

for i=1:m
    if ismember(interp_dates(i),dates)

        %interp_dates is already in dates, we already have the DCF_t0_ti
        DCF_t0_ti(i)=discounts(dates==interp_dates(i));

    elseif interp_dates(i)<max(dates) && interp_dates(i)>min(dates)

        %interp_dates is between two dates of the dates for which we know
        %DCF_t0_ti
    
        %respective zero rates
        y=-log(discounts(2:end))./(yearfrac(dates(1),dates(2:end),3));     
        
        %we now linear interpolate 
        interp_y=interp1(dates(2:end),y,interp_dates(i));

        DCF_t0_ti(i)=exp(-interp_y*yearfrac(dates(1),interp_dates(i),3));
    else

        %interp_dates not in dates, hence is after, we extrapolate based on
        %all dates that we have
        y=-log(discounts(2:end))./(yearfrac(dates(1),dates(2:end),3));
        interp_y=interp1(dates(2:end),y,interp_dates,'previous','extrap');

        DCF_t0_ti(i)=exp(-interp_y*yearfrac(dates(1),interp_dates,3));

    end
end

end
