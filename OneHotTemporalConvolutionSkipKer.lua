--- One-hot Temporal Convolution
-- Tensor size flow:
-- B, M (,V)
--     V, C, kW, kS
-- B, M-kWW+1, C
-- where
--   B = batch size
--   M = nInputFrame = sequence length
--   V = inputFrameSize = vocabulary size
--   C = outputFrameSize = "embedding" size
--   KS = kernel Skip
--   kWW = kernel virtual width

require'torch'
require'nn'

-- classdef: One-Hot Temporal Convolution
local OneHotTemporalConvolutionSkipKer, parent = torch.class('nn.OneHotTemporalConvolutionSkipKer', 'nn.Sequential')

function OneHotTemporalConvolutionSkipKer:__init(V, C, kW, kS)
    parent.__init(self)

    local function check_arg()
        assert(V>0 and C>0 and kW>0)
        self.V = V
        self.C = C
        self.kW = kW

        local function get_dft_kS()
            local kS = {}
            for i = 1, #self.kW - 1 do kS[i] = 0 end
            return kS
        end
        local kS = kS or get_dft_kS()
        table.foreach(kS, function(x) assert(x>=0, "each kS elem must >= 0") end)
        assert(#kS == kW-1, "kS and kW dont match")
        self.kS = kS
    end
    check_arg()

    -- submodules: narrow + lookuptable
    local submds = {}
    local offset = {[1] = 1}
    for i = 1, kW do
        if i > 1 then
            offset[i] = offset[i-1] + self.kS[i-1] + 1
        end

        local length = 1 -- set it as (M - kWW + 1) at runtime
        submds[i] = nn.Sequential()
            -- B, M (,V)
            :add(nn.OneHotNarrowExt(V, 2,offset[i],length))
            -- B, M-kWW+1 (,V)
            :add(nn.LookupTableExt(V,C))
            -- B, M-kWW+1, C
    end
    self.offset = offset
    self.kWW = offset[kW] -- last one is the virtual kernel width

    -- multiplexer: send input to each submodule
    local ct = nn.ConcatTable()
    for i = 1, kW do
        ct:add(submds[i])
    end

    -- the container to be returned
    -- B, M (,V)
    self:add(ct)
    -- {B, M-kWW+1, C}, {B, M-kWW+1, C}, ...
    self:add(nn.CAddTable())
    -- B, M-kWW+1, C
end

function OneHotTemporalConvolutionSkipKer:updateOutput(input)
    assert(input:dim()==2, "input size must be dim 2: B, M")

    -- need to the seq length for current input batch
    local M = input:size(2)
    assert(M >= self.kWW,
        ("(virtual) kernel size %d > seq length %d, failed"):format(self.kWW, M)
    )
    self:_reset_seq_length(M)

    return parent.updateOutput(self, input)
end

-- Okay with default backward(), which calls each module's backward()

function OneHotTemporalConvolutionSkipKer:__tostring__()
    local s = string.format('%s(%d -> %d, %d (virtual %d)',
        torch.type(self),  self.V, self.C, self.kW, self.kWW)
    return s .. ')'
end

-- additional methods
function OneHotTemporalConvolutionSkipKer:should_updateGradInput(flag)
    assert(flag==true or flag==false, "flag must be boolean!")

    -- set each submoule
    local function set_each_flag(mods)
        for _, md in ipairs(mods) do
            md:should_updateGradInput(flag)
        end
    end
    local ms = self:findModules('nn.OneHotNarrowExt')
    local mms = self:findModules('nn.LookupTableExt')
    set_each_flag(ms)
    set_each_flag(mms)
end

function OneHotTemporalConvolutionSkipKer:index_copy_weight(vocabIdxThis, convThat, vocabIdxThat)
    assert(torch.type(convThat) == torch.type(self),
        "arg convThat is an unexpected type " .. type(convThat) .. ", expected " .. type(self)
    )
    assert(torch.type(vocabIdxThis) == 'torch.LongTensor')
    assert(torch.type(vocabIdxThat) == 'torch.LongTensor')

    -- ref vars of this, that wieghts
    local weightConvThis = self:parameters()
    local weightConvThat = convThat:parameters()

    -- p = kernel size = region size = #LookupTable
    local function check_kernelsize()
        local p, pp = #weightConvThis, #weightConvThat
        assert(p == pp,
            "inconsistent region size: this = " .. p .. ", that = " .. pp
        )
        return p
    end
    local p = check_kernelsize()

    for i = 1, p do
        local weigthThis = weightConvThis[i]
        local weightThat = weightConvThat[i]

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

-- helpers
function OneHotTemporalConvolutionSkipKer:_reset_seq_length(M)
    local contable = self.modules[1]
    local length = M - self.kWW + 1
    for i = 1, #contable.modules do
        -- reset nn.OneHotNarrowExt length
        contable.modules[i].modules[1].length = length
    end
end

--- preliminary doc
--[[
### OneHotTemporalConvolutionSkipKer

#### Constructor:
```lua
module = nn.OneHotTemporalConvolutionSkipKer(inputFrameSize, outputFrameSize, kW, kS)
```

Example:
```Lua
require'cunn'
require'onehot-temp-conv'

B = 200 -- batch size
M = 45 -- sequence length (#words)
V = 12333 -- inputFrameSize (vocabulary size)
C = 300 -- outputFrameSize (#output feature maps, or embedding size)
kW = 3 -- convolution kernel size (width)
kS = {2, 3}

-- inputs: the one-hot vector as index in set {1,2,...,V}. size: B, M
inputs = torch.LongTensor(B, M):apply(
    function (e) return math.random(1,V) end
):cuda()

-- the 1d conv module
tf = nn.OneHotTemporalConvolutionSkipKer(V, C, kW, kS):cuda()

-- outputs: the dense tensor. size: B, M-kWW+1, C
outputs = tf:forward(inputs)
MM = outputs:size(2)
kWW = kW + tablex.reduce('+', kS)
assert(MM == M-kWW+1)

-- back prop: the gradients w.r.t. parameters
gradOutputs = outputs:clone():normal()
tf:backward(inputs, gradOutputs)
```
]]--