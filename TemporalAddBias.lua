-- Input: B, M, C
--   Bias: C
-- Output: B, M, C
local TemporalAddBias, parent = torch.class('nn.TemporalAddBias', 'nn.Module')

local function checkInputSize(input)
    assert(input:dim() == 3)
    return input:size(1), input:size(2), input:size(3)
end

function TemporalAddBias:__init(inputSize)
    parent.__init(self)

    assert(type(inputSize) == 'number')
    local C = inputSize
    self.bias = torch.Tensor(1,1,C)
    self.gradBias = torch.Tensor(1,1,C)

    self:reset(0)
end

function TemporalAddBias:reset(v)
    v = v or 0
    self.bias:fill(v)
end

function TemporalAddBias:updateOutput(input)
    local B, M, C = checkInputSize(input)

    self.output:resizeAs(input):copy(input)
    self.output:add( self.bias:expand(B,M,C) )

    return self.output
end

function TemporalAddBias:updateGradInput(input, gradOutput)
    if self.gradInput then
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
        return self.gradInput
    end
end

function TemporalAddBias:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    local B, M, C = checkInputSize(input)
    assert(input:isSameSizeAs(gradOutput)) -- B, M, C

    self.gradBias:add(scale,
        gradOutput:view(B*M, C):sum(1):view(1,1,C)
    )
end