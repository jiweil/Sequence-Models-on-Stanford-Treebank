function[lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,isTraining)
    N=size(batch.Word,1);
    T=batch.MaxLen;
    zeroState=zeroMatrix([parameter.hidden,N]);
    all_h_t=cell(parameter.layer_num,T);
    all_c_t=cell(parameter.layer_num,T);
    all_h_t_r=cell(parameter.layer_num,T);
    all_c_t_r=cell(parameter.layer_num,T);
    lstms = cell(parameter.layer_num,T);
    lstms_r = cell(parameter.layer_num,T);
    for ll=1:parameter.layer_num
        for tt=1:T
            all_h_t{ll,tt}=zeroMatrix([parameter.hidden,N]);
            all_c_t{ll,tt}=zeroMatrix([parameter.hidden,N]);
            all_h_t_r{ll,tt}=zeroMatrix([parameter.hidden,N]);
            all_c_t_r{ll,tt}=zeroMatrix([parameter.hidden,N]);
        end
    end
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
            [lstms{ll, t},all_h_t{ll, t},all_c_t{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);
        end
    end
    h=all_h_t{parameter.layer_num,T};
    clear all_h_t;

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

