function Vpi = Comp_Vpi(s,Absorbing,gamma,action,R)
    if action(s)==1
        s1=s-1;
    else
        s1=s+1;
    end
    if Absorbing(s)==1
        Vpi=0;
    else
        Vpi=transition_function(s1,action(s),s)*(R(s1,s,action(s))+gamma*Comp_Vpi(s1,Absorbing,gamma,action,R));
    end

end