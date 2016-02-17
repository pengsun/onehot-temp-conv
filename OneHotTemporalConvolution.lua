--- One-hot Temporal Convolution
-- Tensor size flow:
-- B, M (,V)
--     V, C, kW
-- B, M-kW+1, C
-- where
--   B = batch size
--   M = nInputFrame = sequence length
--   V = inputFrameSize = vocabulary size
--   C = outputFrameSize = "embedding" size

require'torch'
require'nn'

--- classdef: One-Hot Temporal Convolution
local OneHotTemporalConvolution, parent = torch.class('nn.OneHotTemporalConvolution', 'nn.Sequential')

function OneHotTemporalConvolution:__init(V, C, kW)
    parent.__init(self)

    local function check_arg()
        assert(V>0 and C>0 and kW>0)
        self.V = V
        self.C = C
        self.kW = kW
    end
    check_arg()

    -- submodules: lookuptable + narrow
    local ltna = {}
    for i = 1, kW do
        local offset = i
        local length = 1 -- set it as (M - kW + 1) at runtime
        ltna[i] = nn.Sequential()
            -- B, M (,V)
            :add(nn.NarrowNoBP(2,offset,length))
            -- B, M-kW+1 (,V)
            :add(nn.LookupTable(V,C))
            -- B, M-kW+1, C
    end

    -- send input to each submodule
    local ct = nn.ConcatTable()
    for i = 1, kW do
        ct:add(ltna[i])
    end

    -- the container to be returned
    -- B, M (,V)
    self:add(ct)
    -- {B, M-kW+1, C}, {B, M-kW+1, C}, ...
    self:add(nn.CAddTable())
    -- B, M-kW+1, C
end

function OneHotTemporalConvolution:updateOutput(input)
    assert(input:dim()==2, "input size must be dim 2: B, M")
    local M = input:size(2)
    assert(M >= self.kW,
        ("kernel size %d > seq length %d, failed"):format(self.kW, M)
    )
    self:reset_seq_length(M)

    return parent.updateOutput(self, input)
end

--- Okay with default OnehotTemporalConv:backward

function OneHotTemporalConvolution:__tostring__()
    local s = string.format('%s(%d -> %d, %d',
        torch.type(self),  self.V, self.C, self.kW)
    return s .. ')'
end

function OneHotTemporalConvolution:index_copy_weight(vocabIdxThis, convThat, vocabIdxThat)
    assert(torch.type(convThat) == torch.type(self),
        "arg convThat is an unexpected type " .. type(convThat) .. ", expected " .. type(self)
    )
    assert(torch.type(vocabIdxThis) == 'torch.LongTensor')
    assert(torch.type(vocabIdxThat) == 'torch.LongTensor')

    -- p = kernel size = region size = #LookupTable
    local function check_kernelsize()
        local p = self:get(1):size() -- #Sub modules = #LookupTable
        local pp = convThat:get(1):size()
        assert(p == pp,
            "inconsistent region size: this = " .. p .. ", that = " .. pp
        )
        return p
    end
    local p = check_kernelsize()

    for i = 1, p do
        -- ConcatTable, Sequential, LookupTable
        local weigthThis = self:get(1):get(i):get(2).weight
        local weightThat = convThat:get(1):get(i):get(2).weight

        -- weight sizes:
        --   V1, C
        --   V2, C
        local CThis, CThat = weigthThis:size(2), weightThat:size(2)
        assert(CThis == CThat,
            "inconsisten outputFrameSize/featureMaps: this " .. CThis, ", that ", CThat
        )

        -- do the copying: this(idx1, :) = that(idx2, :)
        local thisType = weigthThis:type()
        weigthThis:indexCopy(1, vocabIdxThis,
            weightThat:type(thisType):index(1, vocabIdxThat)
        )
    end

end

--- helpers
function OneHotTemporalConvolution:reset_seq_length(M)
    local contable = self.modules[1]
    local kW = #contable.modules
    for i = 1, kW do
        -- reset nn.Narrow length
        local length = M -kW + 1
        contable.modules[i].modules[1].length = length
    end
end