function message = get_message_fn_bea(to_node)
% Calculates a message to be sent to the argument node from the factor
% between variable nodes b, e, and a.
%
% @param     to_node :   variable node to which a message is being passed
%
% @return    message :   message to pass to to_node, MUST BE COLUMN VECTOR

global fn_bea

node = fn_bea;

% this function is recursive so it first gathers all the messages necessary
% to send the message being asked for
for i = 1 : length(node.c)
    if ~strcmp(node.c{i}.name,to_node.name)
        node.m{i} = get_message_vn(node,node.c{i});
    end
end

% now all the child messages are up to date, calculate the message to be
% sent to to_node

%% AG:
%% First encode the given conditional probabilities:
p_a_be = zeros(2,2,2);
%% the order of arguments : (a,b,e) : 1==true, 2==false
p_a_be(1,1,1) = 0.95;
p_a_be(1,1,2) = 0.94;
p_a_be(1,2,1) = 0.29;
p_a_be(1,2,2) = 0.001;
p_a_be(2,:,:) = 1.0 - p_a_be(1,:,:);

%% find which of {a,b,e} is the to_node:
to_dim  = strfind('abe', to_node.name(4:end));
d_order = [1,2,3];
%% reorder the probabilities so that the output dimension is always 3:
d_order = [d_order(d_order~=to_dim) to_dim];
fac     = permute(p_a_be, d_order);

%% calculate the product of incoming messages:
msg_prod = node.m{d_order(1)} * node.m{d_order(2)}';

%% calculate the messages:
m_true   = sum(sum(squeeze(fac(:,:,1)) .* msg_prod));
m_false  = sum(sum(squeeze(fac(:,:,2)) .* msg_prod));
message  = [m_true; m_false];







