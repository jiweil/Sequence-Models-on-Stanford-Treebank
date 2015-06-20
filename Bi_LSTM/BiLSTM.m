function[]=BiLSTM()
clear;
%matlabpool open 16

n= gpuDeviceCount;
parameter.isGPU = 0;
if n>0 % GPU exists
    parameter.isGPU = 1;
    gpuDevice(1);
else
    print('no gpu ! ! ! ! !');
end
addpath('../misc');

parameter.alpha=0.05;
%learning rate
parameter.hidden=30;
parameter.lstm_out_tanh=0;
parameter.layer_num=2;
%number of layer
parameter.Initial=0.01;
%initialization
parameter.class=5;
parameter.dropout=0.25;
%dropout rate
params.lstm_out_tanh=0;
parameter.isTraining=1;
parameter.CheckGrad=0;
parameter.PreTrainEmb=1;
%read from pre-trained embeddings
if parameter.PreTrainEmb==1
    parameter.dimension=300;
    parameter.hidden=150;
    %number of hidden unites
end
parameter.update_embedding=1;
% whether update embeddings

parameter.mini_batch_size=1000;
% minibatch size

if parameter.CheckGrad==1 && parameter.dropout~=0
    parameter.drop_left_1=randSimpleMatrix([parameter.dimension,1])<1-parameter.dropout;
    parameter.drop_left_1=randSimpleMatrix([parameter.dimension,1])<1-parameter.dropout;
    parameter.drop_left=randSimpleMatrix([parameter.hidden,1])<1-parameter.dropout;
end
% if do gradient checking, using pre-ordained dropout
parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;
parameter.activation=1;

train_file='../data/sequence_train.txt';
dev_file='../data/sequence_dev_root.txt';
test_file='../data/sequence_test_root.txt';

parameter.C=0;
[devTag,devNode]=ReadData(dev_file);
devBatches=GetBatch(devTag,devNode,parameter.mini_batch_size,parameter);
% read development batches

[TrainTag,TrainNode]=ReadData(train_file);
TrainBatches=GetBatch(TrainTag,TrainNode,parameter.mini_batch_size,parameter);
% read training batches
[TestTag,TestNode]=ReadData(test_file);
TestBatches=GetBatch(TestTag,TestNode,parameter.mini_batch_size,parameter);

% read testing baatches

[parameter,ada]=Initial(parameter); 
%intial parameter
test_acc=[];
dev_acc=[];

iter=0;
while 1 
    iter=iter+1;
    for j=1:length(TrainBatches)
        batch=TrainBatches{j};
        [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
        % forward
        [batch_cost,grad,prediction]=softmax([h;h_r],batch,parameter,1);
        % softmax
        if parameter.isTraining==1
            grad=Backward(batch,grad,parameter,lstms,all_c_t,lstms_r,all_c_t_r);
            % backward propagation
            clear all_c_t;
            clear all_c_t_r;
            clear lstms;
            clear lstms_r;
            [parameter,ada]=update_parameter(parameter,ada,grad);
            % update parameters
        end
    end
    test_a=Testing(TestBatches,parameter);
    % testing accuracy for current iteration
    test_acc=[test_acc,test_a];
    dev_acc=[dev_acc,Testing(devBatches,parameter)];
    % development accuracy for current iteration
    
    if iter==15
        [a1,a2]=max(dev_acc);
        % select best performance on development set
        acc=test_acc(a2(1));
        disp('fine-grained accuracy at root level is');
        disp(acc);
        break;
    end
end
end

function[accuracy]=Testing(TestBatches,parameter)
% testing with gold standards
    correct=0;
    total=0;
    for j=1:length(TestBatches)
        batch=TestBatches{j};
        %batch.Label
        [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,0);
        [cost1,grad,prediction]=softmax([h;h_r],batch,parameter,0);

        correct=correct+sum(prediction==batch.Label);
        total=total+length(prediction);
    end
    accuracy=correct/total;
end

function[parameter,ada]=update_parameter(parameter,ada,grad)
% adagrad for parameter update
    for i=1:parameter.layer_num
        grad.W{i}=grad.W{i}+parameter.C*parameter.W{i};
        ada.W{i}=ada.W{i}+grad.W{i}.^2;
        L=find(ada.W{i}~=0);
        parameter.W{i}(L)=parameter.W{i}(L)-parameter.alpha*grad.W{i}(L)./sqrt(ada.W{i}(L));

        grad.V{i}=grad.V{i}+parameter.C*parameter.V{i};
        ada.V{i}=ada.V{i}+grad.V{i}.^2;
        L=find(ada.V{i}~=0);
        parameter.V{i}(L)=parameter.V{i}(L)-parameter.alpha*grad.V{i}(L)./sqrt(ada.V{i}(L));
    end
    ada.U=ada.U+grad.U.^2;
    L=find(ada.U~=0);
    parameter.U(L)=parameter.U(L)-parameter.alpha*grad.U(L)./sqrt(ada.U(L));
    if parameter.update_embedding==1
        %parameter.vect(:,grad.indices)=parameter.vect(:,grad.indices)-0.05*grad.W_emb;
        if 1==1
        ada.vect(:,grad.indices)=ada.vect(:,grad.indices)+grad.W_emb.^2;
        L=find(grad.W_emb~=0);
        ada_part=ada.vect(:,grad.indices);
        update_vect=zeroMatrix(size(ada_part),parameter.isGPU);
        update_vect(L)=grad.W_emb(L)./sqrt(ada_part(L));
        parameter.vect(:,grad.indices)=parameter.vect(:,grad.indices)-parameter.alpha*update_vect;
        end
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
    parameter.U=randomMatrix(parameter.Initial,[parameter.class,2*parameter.hidden],parameter.isGPU);
    ada.U=zeroMatrix([parameter.class,2*parameter.hidden],parameter.isGPU);
    if parameter.PreTrainEmb==1
        %parameter.vect=gpuArray(load('../data/neg_300_sentiment.txt')');
        %disp('neg_300_sentiment.txt')
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

function[Batches]=GetBatch(Tag,Word,batch_size,parameter)
% get batches
    N_batch=ceil(length(Word)/batch_size);
    Batches={};
    for i=1:N_batch
        Begin=batch_size*(i-1)+1;
        End=batch_size*i;
        if End>length(Word)
            End=length(Word);
        end
        current_batch=Batch();
        current_batch.Label=zeros(1,End-Begin+1);
        for j=Begin:End
            leng=length(Word{j});
            if current_batch.MaxLen<leng
                current_batch.MaxLen=leng;
            end
        end
        current_batch.Word=ones(End-Begin+1,current_batch.MaxLen);
        current_batch.Word_r=ones(End-Begin+1,current_batch.MaxLen);
        Delete=zeros(End-Begin+1,current_batch.MaxLen);
        for j=Begin:End
            leng=length(Word{j});
            current_batch.Word(j-Begin+1,current_batch.MaxLen-leng+1:current_batch.MaxLen)=Word{j};
            current_batch.Word_r(j-Begin+1,current_batch.MaxLen-leng+1:current_batch.MaxLen)=wrev(Word{j});
            Delete(j-Begin+1,1:current_batch.MaxLen-leng)=1;
            current_batch.Label(j-Begin+1)=Tag{j};
        end
        for j=1:current_batch.MaxLen
            current_batch.Delete{j}=find(Delete(:,j)==1);
            current_batch.Left{j}=find(Delete(:,j)==0);
        end
        Batches{i}=current_batch;
    end
end

function[Tag,Word]=ReadData(filename)
% read data
    fd=fopen(filename);
    tline = fgets(fd);
    i=0;
    Tag={};Word={};
    while ischar(tline)
        i=i+1;
        text=deblank(tline);
        Tag{i}=str2num(text(1));
        text=text(3:length(text));
        Word{i}=str2num(text);
        tline = fgets(fd);
    end
end

% check gradient
function check_vect(value1,i,j,batch,parameter)
    e=0.001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost1,grad,prediction]=softmax([h;h_r],batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost2,grad,prediction]=softmax([h;h_r],batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end


function check_V(value1,ll,i,j,batch,parameter)
    e=0.0001;
    parameter.V{ll}(i,j)=parameter.V{ll}(i,j)+e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost1,grad,prediction]=softmax([h;h_r],batch,parameter,0);
    parameter.V{ll}(i,j)=parameter.V{ll}(i,j)-2*e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost2,grad,prediction]=softmax([h;h_r],batch,parameter,0);
    parameter.V{ll}(i,j)=parameter.V{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end

function check_W(value1,ll,i,j,batch,parameter)
    e=0.0001;
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)+e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost1,grad,prediction]=softmax([h;h_r],batch,parameter,0);
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)-2*e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost2,grad,prediction]=softmax([h;h_r],batch,parameter,0);
    parameter.W{ll}(i,j)=parameter.W{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    [value1,value2]
    value1-value2
end

function check_U(value1,i,j,batch,parameter)
    e=0.0001;
    parameter.U(i,j)=parameter.U(i,j)+e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost1,grad,prediction]=softmax([h;h_r],batch,parameter);
    parameter.U(i,j)=parameter.U(i,j)-2*e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost2,grad,prediction]=softmax([h;h_r],batch,parameter);
    parameter.U(i,j)=parameter.U(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end
