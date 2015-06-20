function[grad]=Backward(batch,grad,result,parameter)
    % backward propagation to clause level
    grad=Backward_Sen(batch,grad,result,parameter);
    % backward propagation to word level
    grad=Backward_Word(batch,grad,result,parameter);
    grad=Normalize(grad,batch,parameter);
end

function[grad]=Normalize(grad,batch,parameter)
    % normaile gradients, divided by number of instances
    N=size(batch.doc_sen_matrix,1);
    for ll=1:parameter.layer_num
        grad.W{ll}=grad.W{ll}/N;
        grad.V{ll}=grad.V{ll}/N;
    end
    for ll=1:parameter.sen_layer_num
        grad.sen_W{ll}=grad.sen_W{ll}/N;
    end
    grad.U=grad.U/N;
    grad.W_emb=grad.W_emb/N;
end

function[grad]=Backward_Sen(batch,grad,result,parameter)
    % backward propagation to clause/sub-sentence level
    N=size(batch.doc_sen_matrix,1);
    T=size(batch.doc_sen_matrix,2);
    zeroState=zeroMatrix([parameter.hidden,N],parameter.isGPU);
    for ll=1:parameter.sen_layer_num
        dh{ll}=zeroState;
        dc{ll}=zeroState;
        grad.sen_W{ll}=zeroMatrix(size(parameter.sen_W{ll}),parameter.isGPU);
    end
    grad.sen=zeroMatrix(size(result.h),parameter.isGPU);
    
    for sen_tt=T:-1:1
        for ll=parameter.sen_layer_num:-1:1
            if sen_tt==T&&ll==parameter.sen_layer_num
                dh{ll}=grad.grad_ht(1:parameter.hidden,:);;
            end
            if sen_tt==1
                c_t_1 =zeroState;
            else
                c_t_1=result.all_c_t_sen{ll,sen_tt-1};
            end
            c_t=result.all_c_t_sen{ll, sen_tt};
            lstm=result.lstms_sen{ll, sen_tt};
            [lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1, dc{ll}, dh{ll},ll,sen_tt,zeroState,parameter,parameter.sen_W{ll});
            dc{ll} = lstm_grad.dc;
            dh{ll} = lstm_grad.input(end-parameter.hidden+1:end, :);
            grad.sen_W{ll}=grad.sen_W{ll}+lstm_grad.W;
            if ll==1
                sen_index=batch.doc_sen_matrix(:,sen_tt);
                grad.sen(:,sen_index(batch.doc_sen_left{sen_tt}))=grad.sen(:,sen_index(batch.doc_sen_left{sen_tt}))+lstm_grad.input(1:end-parameter.hidden,batch.doc_sen_left{sen_tt});
                % gradient with regard to sub-sentences
            else
                dh{ll-1}=dh{ll-1}+lstm_grad.input(1:end-parameter.hidden,:);
            end
        end
    end
    for ll=1:parameter.sen_layer_num
        dh{ll}=zeroState;
        dc{ll}=zeroState;
    end
end
function[grad]=Backward_Word(batch,grad,result,parameter)
    % backward propagation to word level
    N=size(batch.sen_word_matrix,1);
    T=size(batch.sen_word_matrix,2);

    zeroState=zeroMatrix([parameter.hidden,N],parameter.isGPU);
    for ll=1:parameter.layer_num
        dh{ll}=zeroState;
        dc{ll}=zeroState;
        grad.W{ll}=zeroMatrix(size(parameter.W{ll}),parameter.isGPU);
        grad.V{ll}=zeroMatrix(size(parameter.V{ll}),parameter.isGPU);
    end

    wordCount = 0;
    numInputWords=size(batch.sen_word_matrix,1)*size(batch.sen_word_matrix,2);
    allEmbGrads=zeroMatrix([parameter.dimension,numInputWords],parameter.isGPU);
    % backpropagate from right to left
    for word_tt=T:-1:1
        unmaskedIds=batch.sen_word_left{word_tt};
        for ll=parameter.layer_num:-1:1
            if word_tt==T&&ll==parameter.layer_num
                dh{ll}=grad.sen(1:parameter.hidden,:);
            end
            if word_tt==1
                c_t_1 =zeroState;
            else
                c_t_1=result.all_c_t_word{ll,word_tt-1};
            end
            c_t=result.all_c_t_word{ll,word_tt};
            lstm=result.lstms_word{ll,word_tt};
            [lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1, dc{ll}, dh{ll},ll,word_tt,zeroState,parameter,parameter.W{ll});
            dc{ll} = lstm_grad.dc;
            dh{ll} = lstm_grad.input(end-parameter.hidden+1:end, :);
            grad.W{ll}=grad.W{ll}+lstm_grad.W;
            if ll==1
                embIndices=batch.sen_word_matrix(unmaskedIds,word_tt)';
                embGrad = lstm_grad.input(1:parameter.dimension,unmaskedIds);
                numWords = length(embIndices);
                allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
                allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
                wordCount = wordCount + numWords;
            else
                dh{ll-1}=dh{ll-1}+lstm_grad.input(1:parameter.hidden,:);
            end
        end
    end
    allEmbGrads(:, wordCount+1:end) = [];
    allEmbIndices(wordCount+1:end) = [];
    [grad.W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices);

    for ll=1:parameter.layer_num
        dh{ll}=zeroState;
        dc{ll}=zeroState;
    end
    % backpropagate from left to right

    wordCount = 0;
    allEmbGrads=zeroMatrix([parameter.dimension,numInputWords],parameter.isGPU);
    allEmbIndices=[];
    for word_tt=T:-1:1
        unmaskedIds=batch.sen_word_left{word_tt};
        for ll=parameter.layer_num:-1:1
            if word_tt==T&&ll==parameter.layer_num
                dh{ll}=grad.sen(1+parameter.hidden:2*parameter.hidden,:);;
            end
            if word_tt==1
                c_t_1 =zeroState;
            else
                c_t_1=result.all_c_t_word_r{ll,word_tt-1};
            end
            c_t=result.all_c_t_word_r{ll,word_tt};
            lstm=result.lstms_word_r{ll,word_tt};
            [lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1, dc{ll}, dh{ll},ll,word_tt,zeroState,parameter,parameter.V{ll});
            dc{ll} = lstm_grad.dc;
            dh{ll} = lstm_grad.input(end-parameter.hidden+1:end, :);
            grad.V{ll}=grad.V{ll}+lstm_grad.W;
            if ll==1
                embIndices=batch.sen_word_matrix_r(unmaskedIds,word_tt)';
                embGrad = lstm_grad.input(1:parameter.dimension,unmaskedIds);
                numWords = length(embIndices);
                allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
                allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
                wordCount = wordCount + numWords;
            else
                dh{ll-1}=dh{ll-1}+lstm_grad.input(1:parameter.hidden,:);
            end
        end
    end
    allEmbGrads(:, wordCount+1:end) = [];
    allEmbIndices(wordCount+1:end) = [];
    [W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices);
    grad.W_emb=grad.W_emb+W_emb;
    clear result
    clear allEmbGrads;
    clear allEmbIndices;
end

function[lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1, dc, dh, ll, t, zero_state,parameter,W)
    dc = arrayfun(@plusMult, dc, lstm.o_gate, dh);
    do = arrayfun(@sigmoidPrimeTriple, lstm.o_gate, c_t, dh);
    di = arrayfun(@sigmoidPrimeTriple, lstm.i_gate, lstm.a_signal, dc);

    if t>1 
        df = arrayfun(@sigmoidPrimeTriple, lstm.f_gate, c_t_1, dc);
    else 
        df = zero_state;
    end
    lstm_grad.dc = lstm.f_gate.*dc;
    if parameter.activation==1
        dl = arrayfun(@tanhPrimeTriple, lstm.a_signal, lstm.i_gate, dc);
    else if parameter.activation==2
        dl= arrayfun(@tanh_cube,lstm.a_signal,lstm.store,lstm.i_gate, dc);
    end
    end
    d_ifoa = [di; df; do; dl];
    lstm_grad.W = d_ifoa*lstm.input'; %dw
    lstm_grad.input =W'*d_ifoa;% dx dh
    if parameter.dropout~=0 
        lstm_grad.input=lstm_grad.input.*lstm.drop_left;
    end
end

function[value]=tanh_cube(x,y,z,m)
    value=(1-x*x)*(3*y^2+1)*z*m;
end

function [value] = plusTanhPrimeTriple(t, x, y, z)
    value = t + (1-x*x)*y*z;
end
function [value] = tanhPrimeDouble(x, y)
    value = (1-x*x)*y;
end
function [value] = tanhPrimeTriple(x, y, z)
    value = (1-x*x)*y*z;
end
function [value] = plusMult(x, y, z)
    value = x + y*z;
end
function [value] = sigmoidPrimeTriple(x, y, z)
    value = x*(1-x)*y*z;
end

