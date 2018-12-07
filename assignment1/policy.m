function Paction = policy(S,Absorbing)
% always up
%Paction=bsxfun(@times,~Absorbing',[ones(1,S);zeros(1,S)]');%A*S
Paction=bsxfun(@times,~Absorbing',[zeros(1,S);ones(1,S)]');
%always down
%action=ones(1,S)*2;

end