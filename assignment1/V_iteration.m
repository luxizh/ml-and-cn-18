function [OPolicy,Vpi] = V_iteration(S,A,T,gamma,R,tol,Absorbing)
    OPolicy = zeros(1,S);
    Vpi=-100*ones(1,S);
    delta=2*tol;
    while(delta>tol)
        delta=0;
        for s=1:S
            if Absorbing(s)
                Vpi(s)=0;
                continue
            end
            tv=Vpi(s);
            for a=1:A
                if (Vpi(s)<T(:,s,a)'*R(:,s,a)+Vpi*T(:,s,a)*gamma)
                    Vpi(s)=T(:,s,a)'*R(:,s,a)+Vpi*T(:,s,a)*gamma;
                    OPolicy(s)=a;
                end
            end
            delta=max(delta,abs(tv-Vpi(s)));
        end
    end
end
