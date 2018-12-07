function Vpi = DP_Vpi(S,A,T,gamma,Pac,R,tol)
    Vpi=zeros(1,S);
    delta=tol*2;
    while(delta>tol)
        delta=0;
        for s=1:S
            v=Vpi(s);
            Vpi(s)=0;
            for a=1:A
                Vpi(s)=Vpi(s)+Pac(s,a)*T(:,s,a)'*R(:,s,a)+Vpi*Pac(s,a)*T(:,s,a)*gamma;
            end
            %[s,delta,abs(v-Vpi(s))]
            delta=max(delta,abs(v-Vpi(s)));
        end
    end
end