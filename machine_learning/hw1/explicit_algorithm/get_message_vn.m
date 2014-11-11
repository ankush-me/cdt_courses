function message = get_message_vn(to_node, this_node)
% Calculate the message in the variable node this_node to pass to the 
% factor node to_node.
%
% @param     to_node     :   factor node we are passing a message to
% @param     this_node   :   variable node we are passing a message from
%
% @return    message     :   message we are passing, MUST BE COLUMN VECTOR

%% AG :
% this_node : must be a variable node
assert(strfind(this_node.name,'vn_') == 1)
% to_node : must be a factor node
assert(strfind(to_node.name,'fn_') == 1)

% this function is recursive so it first gathers together the messages it
% needs in order to pass the correct message.
for i = 1 : length(this_node.c)
	if ~strcmp(this_node.c{i}.name,to_node.name)
    	cmnd = ['get_message_' this_node.c{i}.name '(this_node)'];
        this_node.m{i} = eval(cmnd);
    end
end

% set default value
d = size(this_node.m{1},1);
message = ones(d,1);

% You are responsible for implementing the logic below here to calculate
% the correct value of message.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% AG : 
mask = ones(d,1);
if this_node.set
	mask = double([this_node.value==true; this_node.value==false]);
end	

for i = 1 : length(this_node.c)
	if ~strcmp(this_node.c{i}.name,to_node.name)
    	message = message .* this_node.m{i};
	end
end
message = message .* mask;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
