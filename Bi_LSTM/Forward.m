function[lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,isTraining)
    % forward calculation for bi-directional LSTM
    N=size(batch.Word,1);
    T=batch.MaxLen;
    zeroState=zeroMatrix([parameter.hidden,N],parameter.isGPU);
    all_h_t=cell(parameter.layer_num,T);
    % h from left to right
    all_c_t=cell(parameter.layer_num,T);
    % c from left to right
    all_h_t_r=cell(parameter.layer_num,T);
    % h from right to left
    all_c_t_r=cell(parameter.layer_num,T);
    % c frim right to left
    lstms = cell(parameter.layer_num,T);
    lstms_r = cell(parameter.layer_num,T);
    for ll=1:parameter.layer_num
        for tt=1:T
            all_h_t{ll,tt}=zeroState;
            all_c_t{ll,tt}=zeroState;
            all_h_t_r{ll,tt}=zeroState;
            all_c_t_r{ll,tt}=zeroState;
        end
    end

    % lstm calculation from left to right
    for t=1:T
        for ll=1:parameter.layer_num
            W=parameter.W{ll};
            if t==1
                h_t_1=zeroState;
                c_t_1 =zeroState;
            else
                c_t_1 = all_c_t{ll, t-1};
                h_t_1 = all_h_t{ll, t-1};
            end
            if ll==1
                x_t=parameter.vect(:,batch.Word(:,t));
            else
                x_t=all_h_t{ll-1,t};
            end
            x_t(:,batch.Delete{t})=0;
            h_t_1(:,batch.Delete{t})=0;
            c_t_1(:,batch.Delete{t})=0;
            % set values for positions not being occupied
            [lstms{ll, t},all_h_t{ll, t},all_c_t{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
            % lstm unit calculation
        end
    end
    h=all_h_t{parameter.layer_num,T};
    clear all_h_t;


    % lstm calculation from right to left
    for t=1:T
        for ll=1:parameter.layer_num
            W=parameter.V{ll};
            if t==1
                h_t_1=zeroState;
                c_t_1 =zeroState;
            else
                c_t_1 = all_c_t_r{ll, t-1};
                h_t_1 = all_h_t_r{ll, t-1};
            end
            if ll==1
                x_t=parameter.vect(:,batch.Word_r(:,t));
            else
                x_t=all_h_t_r{ll-1,t};
            end
            x_t(:,batch.Delete{t})=0;
            h_t_1(:,batch.Delete{t})=0;
            c_t_1(:,batch.Delete{t})=0;
            [lstms_r{ll, t},all_h_t_r{ll, t},all_c_t_r{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
        end
    end
    h_r=all_h_t_r{parameter.layer_num,T};
    clear all_h_t_r;
end

function[lstm, h_t, c_t]=lstmUnit(W,parameter,x_t,h_t_1, c_t_1, ll, t,isTraining)
    % lstm unit calcualtions
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
