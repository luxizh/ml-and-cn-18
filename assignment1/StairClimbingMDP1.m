function [S, A, T, R, StateNames, ActionNames, Absorbing] = StairClimbingMDP1()
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
T = transition_matrix();

% load reward matrix
R = reward_matrix(S,A);
end