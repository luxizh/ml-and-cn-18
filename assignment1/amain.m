clc;
clear;
close all;

[S, A, T, R, StateNames, ActionNames, Absorbing]=StairClimbingMDP1();
Pac = policy(S,Absorbing);
Vpi=zeros(1,S);
tol=0.001;
%% ============Q4 varing gama=================
gamma=0.5;
Vpi = DP_Vpi(S,A,T,gamma,Pac,R,tol);

%{
for s=1:7
    Vpi(s) = Comp_Vpi(s,Absorbing,gamma,action,R);
end
%}

%% ============Q5 plot=================
tVpi=zeros(1,S);
Vpi_up4=zeros(11,1);
Vpi_down4=zeros(11,1);
for i=1:11
    gamma=(i-1)*0.1;
    tVpi = DP_Vpi(S,A,T,gamma,bsxfun(@times,~Absorbing',[ones(1,S);zeros(1,S)]'),R,tol);
    Vpi_up4(i)=tVpi(4);
    tVpi = DP_Vpi(S,A,T,gamma,bsxfun(@times,~Absorbing',[zeros(1,S);ones(1,S)]'),R,tol);
    Vpi_down4(i)=tVpi(4);
    %Vpi_up4(i) = Comp_Vpi(4,Absorbing,gamma,ones(1,S)*2,R);
    %Vpi_down4(i) = Comp_Vpi(4,Absorbing,gamma,ones(1,S),R);
end
figure(1);
plot(([1:11]-1)*0.1,Vpi_up4),hold on;
plot(([1:11]-1)*0.1,Vpi_down4);
legend('up','down')
xlabel('gama'),ylabel('value function of s4');
%% =============Q6==================
gamma=0.55;
[O,V] = V_iteration(S,A,T,gamma,R,tol,Absorbing);
%{
gama=0.5;
action=ones(1,S);
action_try=ones(1,S);
Vpif=zeros(1,S);
for i=1:7
    action_try=action;
    Vpif(i) = Comp_Vpi(i,Absorbing,gama,action,R);
    action_try(i)=3-action(i);
    if(Comp_Vpi(i,Absorbing,gama,action_try,R)>Vpif(i))
        action=action_try;
    end
end
%}
