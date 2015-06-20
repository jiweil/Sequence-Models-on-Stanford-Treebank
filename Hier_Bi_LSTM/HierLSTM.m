function[]=HierLSTM()
clear;
n= gpuDeviceCount;
parameter.isGPU = 0;
if n>0 % GPU exists
    parameter.isGPU = 1;
    gpuDevice(1);
else
    print('no gpu ! ! ! ! !');
end
addpath('../misc');


parameter.dimension=30;
parameter.alpha=0.05; 
% learning rate
parameter.layer_num=2;  
% number of layer for word composition
parameter.sen_layer_num=1;
% number of layer for sub-sentence/clause composition
parameter.hidden=15;
parameter.lstm_out_tanh=0;
parameter.Initial=0.01;
parameter.class=5;
parameter.dropout=0.25;
% dropout rate
params.lstm_out_tanh=0;
parameter.isTraining=1;
parameter.CheckGrad=0;
parameter.PreTrainEmb=1;
if parameter.PreTrainEmb==1
    % read pre-trained embeddings
    parameter.dimension=300;
    parameter.hidden=150;
end
parameter.update_embedding=1;
% whether update word embeddings

parameter.batch_size=1000;
% batch size
parameter.C=0;
% L2 regulizer

if parameter.CheckGrad==1 && parameter.dropout~=0
    % for gradient check, use identical dropout
    parameter.drop_left_1=randSimpleMatrix([parameter.dimension,1])<1-parameter.dropout;
    parameter.drop_left_1=randSimpleMatrix([parameter.dimension,1])<1-parameter.dropout;
    parameter.drop_left=randSimpleMatrix([parameter.hidden,1])<1-parameter.dropout;
end

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;
parameter.activation=2;
% activation function

train_file='../data/sequence_train_segment.txt'
dev_file='../data/sequence_dev_root_segment.txt';
test_root_file='../data/sequence_test_root_segment.txt';

if 1==0
    train_file='../data/train_small.txt';
end

iter=0;

[parameter,ada]=Initial(parameter); %intial parameter

dev_acc=[];
% accruacy for different iterations on dev set
test_acc=[];
% accruacy for different iterations on test set
iter=0;
while 1 
    fd_train=fopen(train_file);
    iter=iter+1;
    while 1
        [batch,Stop]=ReadData(fd_train,parameter);
        if length(batch.doc_sen_matrix)==0
            break;
        end
        [result]=Forward(batch,parameter,1);
        % Forward
        [batch_cost,grad,prediction]=softmax(result,batch,parameter,1);
        % softmax
        if parameter.isTraining==1
            grad=Backward(batch,grad,result,parameter);
            % backward propagation
            clear result;
            if 1==0
                check(batch,grad,parameter);
                % whether check gradient
            end
            [parameter,ada]=update_parameter(parameter,ada,grad);
            % update parameter
            clear grad;
            clear prediction;
            clear batch;
        end
        if Stop==1
            % end of document
            break;
        end
    end
    dev_a=Testing(dev_file,parameter);
    % accuracy on dev set for current iteration
    test_a=Testing(test_root_file,parameter)
    % accuracy on test set for current iteration
    dev_acc=[dev_acc;dev_a];
    test_acc=[test_acc,test_a];
    if iter==15
        [m1,m2]=max(dev_acc);
        % find best performance on dev set
        disp('accuracy is ');
        test_acc(m2)
        break;
    end
end
end

function[accuracy]=Testing(test_file,parameter)
% testing 
    fd_test=fopen(test_file);
    correct=0;
    total=0;
    while 1
        [batch,Stop]=ReadData(fd_test,parameter);
        if length(batch.doc_sen_matrix)==0
            break;
        end
        [result]=Forward(batch,parameter,0);
        [batch_cost,grad,prediction]=softmax(result,batch,parameter,0);
        correct=correct+sum(prediction'==batch.Tag);
        total=total+length(prediction);
        if Stop==1
            break;
        end
        clear result;
        clear batch_cost;
        clear grad;
        clear prediction;
    end
    accuracy=1.0*correct/total;
    fclose(fd_test);
end


function[]=check(batch,grad,parameter)
    %check_U(grad.U(1,1),1,1,batch,parameter);
    check_sen_W(grad.sen_W{1}(1,1),1,1,1,batch,parameter);
    check_W(grad.W{1}(1,1),1,1,1,batch,parameter);
    check_W(grad.W{2}(1,1),2,1,1,batch,parameter);
    %check_W(grad.W{3}(1,1),3,1,1,batch,parameter);

    check_V(grad.V{1}(1,1),1,1,1,batch,parameter);
    check_V(grad.V{2}(1,1),2,1,1,batch,parameter);
    %check_V(grad.V{3}(1,1),3,1,1,batch,parameter);
    if 1==0
    for i=1:20
        word_index=grad.indices(i);
        check_vect(grad.W_emb(1,i),1,word_index,batch,parameter)
    end
    end
end

function[parameter,ada]=update_parameter(parameter,ada,grad)
    % update parameter using ada-grad
    for i=1:parameter.layer_num
        grad.W{i}=grad.W{i}+parameter.C*parameter.W{i};
        ada.W{i}=ada.W{i}+grad.W{i}.^2;
        L=find(ada.W{i}~=0);
        % non-zero value
        parameter.W{i}(L)=parameter.W{i}(L)-parameter.alpha*grad.W{i}(L)./sqrt(ada.W{i}(L));

        grad.V{i}=grad.V{i}+parameter.C*parameter.V{i};
        ada.V{i}=ada.V{i}+grad.V{i}.^2;
        L=find(ada.V{i}~=0);
        parameter.V{i}(L)=parameter.V{i}(L)-parameter.alpha*grad.V{i}(L)./sqrt(ada.V{i}(L));
    end
    for i=1:parameter.sen_layer_num
        grad.sen_W{i}=grad.sen_W{i}+parameter.C*parameter.sen_W{i};
        ada.sen_W{i}=ada.sen_W{i}+grad.sen_W{i}.^2;
        L=find(ada.sen_W{i}~=0);
        parameter.sen_W{i}(L)=parameter.sen_W{i}(L)-parameter.alpha*grad.sen_W{i}(L)./sqrt(ada.sen_W{i}(L));
    end

    ada.U=ada.U+grad.U.^2;
    L=find(ada.U~=0);
    parameter.U(L)=parameter.U(L)-parameter.alpha*grad.U(L)./sqrt(ada.U(L));

    if parameter.update_embedding==1
        % if update embedding
        ada.vect(:,grad.indices)=ada.vect(:,grad.indices)+grad.W_emb.^2;
        L=find(grad.W_emb~=0);
        ada_part=ada.vect(:,grad.indices);
        update_vect=zeroMatrix(size(ada_part),parameter.isGPU);
        update_vect(L)=grad.W_emb(L)./sqrt(ada_part(L));
        parameter.vect(:,grad.indices)=parameter.vect(:,grad.indices)-parameter.alpha*update_vect;
    end
    clear grad;
end

function[parameter,ada]=Initial(parameter)
    %random initialization
    m=parameter.Initial;
    for i=1:parameter.layer_num
        if i==1
            parameter.W{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden],parameter.isGPU);
            ada.W{i}=zeroMatrix([4*parameter.hidden,parameter.dimension+parameter.hidden],parameter.isGPU);
            parameter.V{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden],parameter.isGPU);
            ada.V{i}=zeroMatrix([4*parameter.hidden,parameter.dimension+parameter.hidden],parameter.isGPU);
        else
            parameter.W{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden],parameter.isGPU);
            ada.W{i}=zeroMatrix([4*parameter.hidden,2*parameter.hidden],parameter.isGPU);
            parameter.V{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden],parameter.isGPU);
            ada.V{i}=zeroMatrix([4*parameter.hidden,2*parameter.hidden],parameter.isGPU);
        end
    end
    for i=1:parameter.sen_layer_num
        if i==1
            parameter.sen_W{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,3*parameter.hidden],parameter.isGPU);
            ada.sen_W{i}=zeroMatrix([4*parameter.hidden,3*parameter.hidden],parameter.isGPU);
        else
            parameter.sen_W{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden],parameter.isGPU);
            ada.sen_W{i}=zeroMatrix([4*parameter.hidden,2*parameter.hidden],parameter.isGPU);
        end
    end

    parameter.U=randomMatrix(parameter.Initial,[parameter.class,parameter.hidden],parameter.isGPU);
    ada.U=zeroMatrix([parameter.class,parameter.hidden],parameter.isGPU);
    if parameter.PreTrainEmb==1
        %parameter.vect=gpuArray(load('../SG_300_sentiment.txt')');
        if parameter.isGPU==1
            parameter.vect=gpuArray(load('../data/sentiment_glove_300.txt')');
        else
            parameter.vect=load('../data/sentiment_glove_300.txt')';
        end
            
    else
        parameter.vect=randomMatrix(parameter.Initial,[parameter.dimension,19539],parameter.isGPU);
    end
    if parameter.update_embedding==1
        ada.vect=zeroMatrix([parameter.dimension,19539],parameter.isGPU);
    end
end


% check gradient
function check_vect(value1,i,j,batch,parameter)
    e=0.001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    [result]=Forward(batch,parameter,0);
    [cost1,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    [result]=Forward(batch,parameter,0);
    [cost2,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end


function check_V(value1,ll,i,j,batch,parameter)
    e=0.0001;
    parameter.V{ll}(i,j)=parameter.V{ll}(i,j)+e;
    [result]=Forward(batch,parameter,0);
    [cost1,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.V{ll}(i,j)=parameter.V{ll}(i,j)-2*e;
    [result]=Forward(batch,parameter,0);
    [cost2,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.V{ll}(i,j)=parameter.V{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end


function check_sen_W(value1,ll,i,j,batch,parameter)
    e=0.0001;
    parameter.sen_W{ll}(i,j)=parameter.sen_W{ll}(i,j)+e;
    [result]=Forward(batch,parameter,0);
    [cost1,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.sen_W{ll}(i,j)=parameter.sen_W{ll}(i,j)-2*e;
    [result]=Forward(batch,parameter,0);
    [cost2,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.sen_W{ll}(i,j)=parameter.sen_W{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_W(value1,ll,i,j,batch,parameter)
    e=0.0001;
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)+e;
    [result]=Forward(batch,parameter,0);
    [cost1,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)-2*e;
    [result]=Forward(batch,parameter,0);
    [cost2,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_U(value1,i,j,batch,parameter)
    e=0.0001;
    parameter.U(i,j)=parameter.U(i,j)+e;
    [result]=Forward(batch,parameter,0);
    [cost1,grad,prediction]=softmax(result,batch,parameter,1);

    parameter.U(i,j)=parameter.U(i,j)-2*e;
    [result]=Forward(batch,parameter,0);
    [cost2,grad,prediction]=softmax(result,batch,parameter,1);
    parameter.U(i,j)=parameter.U(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end
