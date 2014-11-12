% factor_node : object used for factor nodes in a factor graph
% representation of a graphical model.
%
% properties
%
% factor : factor object associated with this factor node
%
classdef factor_node < node

    properties(GetAccess = 'public', SetAccess = 'private')
        factor;
    end
    
    methods
        
        % factor_node : constructor method to create this factor node
        %
        % @param factor : factor associated with this factor node
        %
        function obj = factor_node(unid, factor)
            obj = obj@node(unid);
            if nargin == 2;
                obj.factor = factor;
            end
        end
        
        % addNode : adds a node to the cell array of neighboring nodes
        %
        function addNode(obj,node)
            l = length(obj.nodes);
            obj.nodes{l + 1} = node;
            obj.messages{l + 1} = ones(node.dimension,1);
        end
        
        % getMessage : gets the message to be sent to the given node. THIS
        %              METHOD IS ONE WITH STUDENT NEEDS TO FILL OUT.
        %
        % @param to_unid : node to which message will be sent
        %
        function message = getMessage(obj, to_unid)

            %% base-case: if I am a leaf: return the factor as it is
            if length(obj.nodes)==1
                message = obj.factor.factor_matrix;
            else
                %% find out the neighbor index of to_unid:
                num_neigh  = length(obj.nodes);
                to_ind = -1;
                for i = 1 : num_neigh
                    if strcmp(obj.nodes{i}.unid, to_unid)
                        to_ind = i;
                        break;
                    end
                end
                assert(to_ind > -1 , 'factor.getMessage : must be connected.');

                %% exclude the message from the to_node:
                neigh_msg = {obj.messages{1:to_ind-1}, [1], obj.messages{to_ind+1:end}};
                msg_prod  = multilinear_product(neigh_msg{:});

                %% CREATE COPIES ALONG OUTPUT DIMENSION
                rep_num = ones(num_neigh,1);
                rep_num(to_ind) = obj.nodes{to_ind}.dimension;
                msg_tensor = repmat(msg_prod, rep_num');

                %% DO ELEMENT WISE PRODUCT
                fac_join = obj.factor.product(msg_tensor);

                %% SUM OVER THE INPUT DIMENSION
                for j=1:num_neigh
                    if j~=to_ind
                        fac_join = sum(fac_join, j);
                    end
                end
                message = fac_join(:);
            end
        end

        % display : overides the default display behavior
        %
        function display(obj)
            display@node(obj);
            disp(' ');
            disp('factor is = ');
            disp(obj.factor);
        end     
    end
    
end