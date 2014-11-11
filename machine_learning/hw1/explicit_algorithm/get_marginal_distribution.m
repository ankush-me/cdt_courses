function dist = get_marginal_distribution(node);
% This function returns the marginal distribution of the variable node
% argument.
%
%@param     node :  variable node
%
%@return    dist :  marginal distribution of variable node argument


% must be a variable node
assert(strfind(node.name,'vn_') == 1)

% create logic to calculate column vector dist which is the marginal
% distribution in this variable node

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% AG : 
%% It is assumed that the message-passing has been done before 
%% calling this function. i.e. : get_all_messages(node) has been
%% called beforehand.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if node.set
	dist = double([node.value==true; node.value==false]);
else
	d = length(node.m{1});
	dist = ones(d,1);
	for i = 1 : length(node.c)
	    dist = dist .* node.m{i};
	end
	dist = dist/sum(dist); %% normalize
end
