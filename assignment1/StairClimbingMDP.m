%%% Basic code to specify an MDP
%%% MLNC Lab Assignment MDP
%%% Aldo Faisal (2015), Imperial College London
function [S, A, T, R, StateNames, ActionNames, Absorbing] = StairClimbingMDP()
% States are:  { s1 <-- s2 <=> s3 <=> s4 <=> s5 <=> s6 --> s7 ];
S = 7; 
StateNames =  ['s1'; 's2'; 's3'; 's4'; 's5'; 's6'; 's7'];

% Actions are: {L,R} --> {1, 2 }
A = 2; 
ActionNames =  ['L'; 'R'];

% Matrix indicating absorbing states
Absorbing = [
%P  1   2   3   4   5  6  7  G   <-- STATES 
       1   0    0   0  0  0  1
];

% load transition
T = transition_matrix()

% load reward matrix
R = reward_matrix(S,A)

%--------------------------------------------------------------------------

% the transition subfunction
function prob = transition_function(priorState, action, postState) 
% reward function (defined locally)
T = transition_matrix()
prob = T(postState,priorState,action)

% get the transition matrix
function T = transition_matrix()
TR = [
% MODIFY HERE
% 1 ...7    <-- FROM STATE
0   0   0   0   0   0   0 ; % 1 TO STATE
0   0   1   0   0   0   0 ; % .
0   0   0   1   0   0   0 ; % .    
0   0   0   0   1   0   0 ; % .
0   0   0   0   0   1   0 ; % .
0   0   0   0   0   0   1 ; % .
0   0   0   0   0   0   0 ; % 7
];
TL = [
% MODIFY HERE
% 1 ...7    <-- FROM STATE
0   0   0   0   0   0   0 ; % 1 TO STATE
1   0   0   0   0   0   0 ; % .
0   1   0   0   0   0   0 ; %  .  
0   0   1   0   0   0   0 ; % .
0   0   0   1   0   0   0 ; % .
0   0   0   0   1   0   0 ; % .
0   0   0   0   0   0   0 ; % 7
];
T = cat(3, TL, TR); %transition probabilities for each action 


%--------------------------------------------------------------------------

% the locally defined reward function
function rew = reward_function(priorState, action, postState) 
% reward function (defined locally)
% MODIFY HERE
if ((priorState == 2) && (action == 1) && (postState == 1))
    rew = -1;
elseif ((priorState == 6) && (action == 2) && (postState == 7))
    rew = 1;
elseif (action == 1)
    rew = 0;
else
    rew = 0;
end

% get the reward matrix
function R = reward_matrix(S, A)
% i.e. 11x11 matrix of rewards for being in state s, 
%performing action a and ending in state s'
R = zeros(S, S, A); 
for i = 1:S
   for j = 1:A
      for k = 1:S
         R(k, i, j) = reward_function(i, j, k);
      end
   end    
end
