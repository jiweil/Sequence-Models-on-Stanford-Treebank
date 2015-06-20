function [result] = randomMatrix(rangeSize, size,isGPU)
    if isGPU==1
        result = 2*rangeSize * (rand(size,'double', 'gpuArray') - 0.5);
    else 
        result = 2*rangeSize * (rand(size)-0.5);
    end
end
