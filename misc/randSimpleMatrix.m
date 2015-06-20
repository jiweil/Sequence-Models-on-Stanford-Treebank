function [result]=randSimpleMatrix(size,isGPU)
    if isGPU==1
        result=rand(size,'double', 'gpuArray');
    else
        result=rand(size);
    end
end
