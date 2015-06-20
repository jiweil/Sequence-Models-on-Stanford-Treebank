function[cost,grad,prediciton]=softmax(soft_max_h,batch,parameter,isTraining)
    N=size(batch.Word,1);
    if isTraining==1 && parameter.dropout~=0
        if parameter.CheckGrad==1
            drop_left=repmat(parameter.drop_left,1,size(x_t,2));
        else
            drop_left=randSimpleMatrix(size(soft_max_h),parameter.isGPU)<1-parameter.dropout;
        end
        soft_max_h=soft_max_h.*drop_left;
    end

    Score=exp(parameter.U*soft_max_h);
    norms = sum(Score, 1);
    probs=bsxfun(@rdivide,Score, norms);
    scoreIndices = sub2ind(size(probs),batch.Label,1:N);
    cost=sum(-log(probs(scoreIndices)))/N;
    [asdfasf,prediciton]=max(Score);
    grad=[];
    if isTraining==1
        probs(scoreIndices)=probs(scoreIndices)-1;
        if parameter.dropout~=0
            grad.grad_ht=parameter.U'*probs.*drop_left ;
        else
            grad.grad_ht=parameter.U'*probs; % dimension=(hidden*class)*(class*N)=hidden*Num_of_example
        end
        grad.U=probs*soft_max_h'; %dimension=(class*N)*(N*hidden)=class*hidden;
    end
    clear Score;
    clear norms;
    clear probs;
    clear scoreIndices;
end

