function T = transition_matrix()
TL = [
% MODIFY HERE
% 1 ...7    <-- FROM STATE
0   1   0   0   0   0   0 ; % 1 TO STATE
0   0   1   0   0   0   0 ; % .
0   0   0   1   0   0   0 ; % .    
0   0   0   0   1   0   0 ; % .
0   0   0   0   0   1   0 ; % .
0   0   0   0   0   0   0 ; % .
0   0   0   0   0   0   0 ; % 7
];
TR = [
% MODIFY HERE
% 1 ...7    <-- FROM STATE
0   0   0   0   0   0   0 ; % 1 TO STATE
0   0   0   0   0   0   0 ; % .
0   1   0   0   0   0   0 ; %  .  
0   0   1   0   0   0   0 ; % .
0   0   0   1   0   0   0 ; % .
0   0   0   0   1   0   0 ; % .
0   0   0   0   0   1   0 ; % 7
];
T = cat(3, TL, TR);