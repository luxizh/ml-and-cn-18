function prob = transition_function(priorState, action, postState) 
% reward function (defined locally)
T = transition_matrix();
prob = T(postState,priorState,action);
end