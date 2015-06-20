function [result] = oneMatrix(size,isGPU)
    if isGPU==1
        result = ones(size, 'double', 'gpuArray');
    else 
        result = ones(size);
    end
end
