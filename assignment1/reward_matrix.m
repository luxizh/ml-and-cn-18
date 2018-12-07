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
end