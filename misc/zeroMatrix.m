function [result] = zeroMatrix(size,isGPU)
    if isGPU==1
        result = zeros(size, 'double', 'gpuArray');
    else
        result = zeros(size);
    end
end
