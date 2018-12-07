function rew = reward_function(priorState, action, postState) 
% reward function (defined locally)
% MODIFY HERE
if ((priorState == 2) && (action == 1) && (postState == 1))
    rew = -100;
elseif ((priorState == 6) && (action == 2) && (postState == 7))
    rew = 100;
elseif (action == 1)
    rew = 10;
else
    rew = -10;
end