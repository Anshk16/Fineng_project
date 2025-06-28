function Payoff=SwaptionPayoff(IndexStartSwap,AlgorithmDates,X,AlgorithmDiscounts,alpha,sigma,Strike, dt, today, term1, term2, integral, sigma_function)
Payoff=[];
IBDayCount = 3;

for j=1:length(X)
    % Calculating B(T_alpha,T_i)
    B=[];
    for i=IndexStartSwap+1:length(AlgorithmDates)
        tau=yearfrac(AlgorithmDates(IndexStartSwap),AlgorithmDates(i),IBDayCount);
        B=[B;(AlgorithmDiscounts(i)/AlgorithmDiscounts(IndexStartSwap))*integral(AlgorithmDates(i), tau, X(j))];
    end
    % Calculating the payoff
    BPV=sum(B)*dt;
    Payoff=[Payoff;max((1-B(end)-BPV*Strike),0)];
end
