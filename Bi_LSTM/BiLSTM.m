function[]=BiLSTM()
clear;
%matlabpool open 16

n= gpuDeviceCount;
parameter.isGPU = 0;
gpuDevice(3);
addpath('../misc');


parameter.dimension=300;
parameter.alpha=0.01;
filename='dr0'
parameter.hidden=parameter.dimension;
parameter.lstm_out_tanh=0;
parameter.layer_num=1;
parameter.Initial=0.1;
parameter.class=5;
parameter.dropout=0.25;
params.lstm_out_tanh=0;
parameter.isTraining=1;
parameter.CheckGrad=0;
parameter.PreTrainEmb=1;
if parameter.PreTrainEmb==1
    parameter.dimension=300;
    parameter.hidden=300;
end
parameter.update_embedding=1;

parameter.mini_batch_size=1000;

if parameter.CheckGrad==1 && parameter.dropout~=0
    parameter.drop_left=randSimpleMatrix([2*parameter.hidden,1])<1-parameter.dropout;
end
%alpha: learning rate for minibatch

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;

train_file='../data/sequence_train.txt';
dev_file='../data/sequence_dev_root.txt';
test_file='../data/sequence_test_root.txt';


parameter.C=0;
disp('reading data')
[devTag,devNode]=ReadData(dev_file);
devBatches=GetBatch(devTag,devNode,parameter.mini_batch_size,parameter);

[TrainTag,TrainNode]=ReadData(train_file);
TrainBatches=GetBatch(TrainTag,TrainNode,parameter.mini_batch_size,parameter);

[TestTag,TestNode]=ReadData(test_file);
TestBatches=GetBatch(TestTag,TestNode,parameter.mini_batch_size,parameter);

disp('reading done')

for i=1:10

[parameter,ada]=Initial(parameter); %intial parameter
test_acc=[];
dev_acc=[];

iter=0;

while 1 
    iter=iter+1;
    for j=1:length(TrainBatches)
        batch=TrainBatches{j};
        [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
        [batch_cost,grad,prediction]=softmax([h;h_r],batch,parameter,1);
        if parameter.isTraining==1
            grad=Backward(batch,grad,parameter,lstms,all_c_t,lstms_r,all_c_t_r);
            if parameter.CheckGrad==1
                check_V(grad.V{1}(1,1),1,1,1,batch,parameter)
                check_W(grad.W{1}(100,100),1,100,100,batch,parameter)
                check_vect(grad.W_emb(1,1),1,grad.indices(1),batch,parameter);
            end

            clear all_c_t;
            clear all_c_t_r;
            clear lstms;
            clear lstms_r;
            [parameter,ada]=update_parameter(parameter,ada,grad);
        end
    end
    test_a=Testing(TestBatches,parameter)
    test_acc=[test_acc,test_a];
    dev_acc=[dev_acc,Testing(devBatches,parameter)];
    
    if iter==15
        [a1,a2]=max(dev_acc);
        acc=test_acc(a2(1));
        disp('fine-grained accuracy at root level is');
        disp(acc);
        dlmwrite(filename,acc,'-append')
        break;
    end
end

end

while 1==1
    a=1;
end
end

function[accuracy]=Testing(TestBatches,parameter)
    correct=0;
    total=0;
    pre=[];
    true=[];
    for j=1:length(TestBatches)
        batch=TestBatches{j};
        %batch.Label
        [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,0);
        [cost1,grad,prediction]=softmax([h;h_r],batch,parameter,0);

        correct=correct+sum(prediction==batch.Label);
        pre=[pre;prediction'];
        true=[true;batch.Label'];
        total=total+length(prediction);
    end
    accuracy=correct/total;
    if accuracy>0.49
        dlmwrite('store',[pre,true]);
    end
end

function[parameter,ada]=update_parameter(parameter,ada,grad)
    for i=1:parameter.layer_num
        ada.W{i}=ada.W{i}+grad.W{i}.^2;
        parameter.W{i}=parameter.W{i}-parameter.alpha*grad.W{i}./sqrt(ada.W{i});
        ada.V{i}=ada.V{i}+grad.V{i}.^2;
        parameter.V{i}=parameter.V{i}-parameter.alpha*grad.V{i}./sqrt(ada.V{i});
    end

    ada.U=ada.U+grad.U.^2;
    parameter.U=parameter.U-parameter.alpha*grad.U./sqrt(ada.U);

    ada.vect(:,grad.indices)=ada.vect(:,grad.indices)+grad.W_emb.^2;
    parameter.vect(:,grad.indices)=parameter.vect(:,grad.indices)-parameter.alpha*grad.W_emb./sqrt(ada.vect(:,grad.indices));

    clear grad;
end

function[parameter,ada]=Initial(parameter)
    %random initialization
    m=parameter.Initial;
    for i=1:parameter.layer_num
        parameter.W{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
        ada.W{i}=smallMatrix(size(parameter.W{i}));

        parameter.V{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,parameter.dimension+parameter.hidden]);
        ada.V{i}=smallMatrix(size(parameter.V{i}));
    end
    parameter.U=randomMatrix(parameter.Initial,[parameter.class,2*parameter.hidden]);
    ada.U=smallMatrix(size(parameter.U));
    parameter.vect=gpuArray(load('../data/sentiment_glove_300.txt')');
    ada.vect=smallMatrix(size(parameter.vect));
end

function[Batches]=GetBatch(Tag,Word,batch_size,parameter)
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


function check_vect(value1,i,j,batch,parameter)
    e=0.001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost1,grad,prediction]=softmax([h;h_r],batch,parameter,0);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    [lstms,h,all_c_t,lstms_r,h_r,all_c_t_r]=Forward(batch,parameter,1);
    [cost2,grad,prediction]=softmax([h;h_r],batch,parameter,0);
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
