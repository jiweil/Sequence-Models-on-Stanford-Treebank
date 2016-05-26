function[cost,grad,prediciton]=softmax(soft_max_h,batch,parameter,isTraining)
    N=size(batch.Word,1);

    Score=exp(parameter.U*soft_max_h);
    norms = sum(Score, 1);
    probs=bsxfun(@rdivide,Score, norms);
    scoreIndices = sub2ind(size(probs),batch.Label,1:N);
    probs(scoreIndices);
    cost=sum(-log(probs(scoreIndices)))/N;
    [asdfasf,prediciton]=max(Score);
    grad=[];
    if isTraining==1
        probs(scoreIndices)=probs(scoreIndices)-1;
        grad.grad_ht=parameter.U'*probs; % dimension=(hidden*class)*(class*N)=hidden*Num_of_example
        grad.U=probs*soft_max_h'; %dimension=(class*N)*(N*hidden)=class*hidden;
    end
end

