function[result]=Forward(batch,parameter,isTraining)
    % Forward 
    result=Forward_Word(batch,parameter,isTraining);
    % Forward calculation for word level
    result=Forward_Sen(batch,parameter,isTraining,result);
    % Forward calculation for sub-sentence level
end

function[result]=Forward_Sen(batch,parameter,isTraining,result)
    N=size(batch.doc_sen_matrix,1);
    T=size(batch.doc_sen_matrix,2);
    zeroState=zeroMatrix([parameter.hidden,N],parameter.isGPU);
    all_h_t=cell(parameter.layer_num,T);
    result.all_c_t_sen=cell(parameter.layer_num,T);
    result.lstms_sen=cell(parameter.layer_num,T);
    for ll=1:parameter.sen_layer_num
        for tt=1:T
            all_h_t{ll,tt}=zeroState;
            result.all_c_t_sen{ll,tt}=zeroState;
        end
    end
    for t=1:T
        for ll=1:parameter.sen_layer_num
            W=parameter.sen_W{ll};
            % parameter at sub-sentence level
            if t==1
                h_t_1=zeroState;
                c_t_1 =zeroState;
            else
                c_t_1 = result.all_c_t_sen{ll, t-1};
                h_t_1 = all_h_t{ll, t-1};
            end
            if ll==1
                x_t=result.h(:,batch.doc_sen_matrix(:,t));
                % embedding at clause level
            else
                x_t=all_h_t{ll-1,t};
            end
            x_t(:,batch.doc_sen_delete{t})=0;
            h_t_1(:,batch.doc_sen_delete{t})=0;
            c_t_1(:,batch.doc_sen_delete{t})=0;
            % mask
            [result.lstms_sen{ll, t},all_h_t{ll, t},result.all_c_t_sen{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
            % lstm unit calculation
        end
    end
    result.sen_h=all_h_t{parameter.sen_layer_num,T};
    clear x_t;
    clear h_t_1;
    clear c_t_1;
    clear all_h_t;
end

function[result]=Forward_Word(batch,parameter,isTraining)
    % Forward calculation at word level, bidirectional
    N=size(batch.sen_word_matrix,1);
    T=size(batch.sen_word_matrix,2);
    zeroState=zeroMatrix([parameter.hidden,N],parameter.isGPU);
    all_h_t=cell(parameter.layer_num,T);
    result.all_c_t_word=cell(parameter.layer_num,T);
    all_h_t_r=cell(parameter.layer_num,T);
    result.all_c_t_word_r=cell(parameter.layer_num,T);
    result.lstms_word = cell(parameter.layer_num,T);
    result.lstms_word_r = cell(parameter.layer_num,T);

    for ll=1:parameter.layer_num
        for tt=1:T
            all_h_t{ll,tt}=zeroState;
            result.all_c_t_word{ll,tt}=zeroState;
            % left to right
            all_h_t_r{ll,tt}=zeroState;
            result.all_c_t_word_r{ll,tt}=zeroState;
            % right to left
        end
    end

    % left to right
    for t=1:T
        for ll=1:parameter.layer_num
            W=parameter.W{ll};
            if t==1
                h_t_1=zeroState;
                c_t_1 =zeroState;
            else
                c_t_1 = result.all_c_t_word{ll, t-1};
                h_t_1 = all_h_t{ll, t-1};
            end
            if ll==1
                x_t=parameter.vect(:,batch.sen_word_matrix(:,t));
            else
                x_t=all_h_t{ll-1,t};
            end
            x_t(:,batch.sen_word_delete{t})=0;
            h_t_1(:,batch.sen_word_delete{t})=0;
            c_t_1(:,batch.sen_word_delete{t})=0;
            [result.lstms_word{ll, t},all_h_t{ll, t},result.all_c_t_word{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
        end
    end
    result.h1=all_h_t{parameter.layer_num,T};
    clear x_t;
    clear h_t_1;
    clear c_t_1;
    clear all_h_t;
    
    % right to left
    for t=1:T
        for ll=1:parameter.layer_num
            W=parameter.V{ll};
            if t==1
                h_t_1=zeroState;
                c_t_1 =zeroState;
            else
                c_t_1 =result.all_c_t_word_r{ll, t-1};
                h_t_1 = all_h_t_r{ll, t-1};
            end
            if ll==1
                x_t=parameter.vect(:,batch.sen_word_matrix_r(:,t));
            else
                x_t=all_h_t_r{ll-1,t};
            end
            x_t(:,batch.sen_word_delete{t})=0;
            h_t_1(:,batch.sen_word_delete{t})=0;
            c_t_1(:,batch.sen_word_delete{t})=0;
            [result.lstms_word_r{ll, t},all_h_t_r{ll, t},result.all_c_t_word_r{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
        end
    end
    result.h2=all_h_t_r{parameter.layer_num,T};
    result.h=[result.h1;result.h2];
    clear all_h_t_r;
    clear h_t_1;
    clear c_t_1;
    clear x_t;
end

function[lstm, h_t, c_t]=lstmUnit(W,parameter,x_t,h_t_1, c_t_1, ll, t,isTraining)
    input=[x_t; h_t_1];
    if parameter.dropout~=0 && isTraining==1
        if parameter.CheckGrad==1
            drop_left=repmat(parameter.drop_left,1,size(x_t,2));
        else
            drop_left=randSimpleMatrix(size(input),parameter.isGPU)<1-parameter.dropout;
        end
        input=input.*drop_left;
        %x_t=x_t.*drop_left;
    else if parameter.dropout~=0 && isTraining==0
        x_t=x_t;
    end
    end
    ifoa_linear = W*input;
    ifo_gate=parameter.nonlinear_gate_f(ifoa_linear(1:3*parameter.hidden,:));
    i_gate = ifo_gate(1:parameter.hidden, :);
    f_gate = ifo_gate(parameter.hidden+1:2*parameter.hidden,:);
    o_gate =ifo_gate(parameter.hidden*2+1:3*parameter.hidden,:);
    lstm.store=ifoa_linear(3*parameter.hidden+1:4*parameter.hidden,:);
    if parameter.activation==1
        a_signal = parameter.nonlinear_f(lstm.store);
    else if parameter.activation==2
         a_signal = parameter.nonlinear_f(lstm.store+lstm.store.^3);
    end
    end
    c_t=f_gate.*c_t_1 + i_gate.*a_signal;
    h_t = o_gate.*c_t;
    lstm.input = input;
    lstm.i_gate = i_gate;
    lstm.f_gate = f_gate;
    lstm.o_gate = o_gate;
    lstm.a_signal = a_signal;
    lstm.c_t = c_t;
    if parameter.dropout~=0 && isTraining==1
        lstm.drop_left=drop_left;
    end
end
