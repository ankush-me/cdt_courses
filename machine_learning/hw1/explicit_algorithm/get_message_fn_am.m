function message = get_message_fn_am(to_node);
% Calculates a message to be sent to the argument node from the factor node
% between variable nodes a and m.
%
% @param     to_node :   variable node to which a message is being passed
% @return    message :   message to pass to to_node, MUST BE COLUMN VECTOR

global fn_am

node = fn_am;

% this for loops ensures that all the messages needed to pass the requested
% message are up to date
for i = 1 : length(node.c)
    if ~strcmp(node.c{i}.name,to_node.name)
    	node.m{i} = get_message_vn(node,node.c{i});
    end
end

% now all the child messages are up to date, calculate the message to be
% sent to to_node
%% AM has two neighbors : vn_m, vn_a
p_m_a = zeros(2,2);
p_m_a(1,1) = 0.7;
p_m_a(1,2) = 0.01;
p_m_a(2,:) = 1.0 - p_m_a(1,:);

to_dim  = strfind('ma', to_node.name(4:end));
d_order = [1,2];
d_order = [d_order(d_order~=to_dim) to_dim];
fac     = permute(p_m_a, d_order);
msg_prod = node.m{d_order(1)};
message  = fac'*msg_prod;
