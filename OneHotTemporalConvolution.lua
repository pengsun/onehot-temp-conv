--- One-hot Temporal Convolution classdef
-- Tensor size flow:
-- B, M (,V)
--     V, C, kW
-- B, M-kW+1, C
-- where
--   B = batch size
--   M = nInputFrame = sequence length
--   V = inputFrameSize = vocabulary size
--   C = outputFrameSize = "embedding" size
--
-- TODO: shortcut for internal module getter

require'torch'
require'nn'

-- main methods
local OneHotTemporalConvolution, parent = torch.class('nn.OneHotTemporalConvolution', 'nn.Sequential')

function OneHotTemporalConvolution:__init(V, C, kW, opt)
    parent.__init(self)

    local function check_arg()
        assert(V>0 and C>0 and kW>0)
        self.V = V
        self.C = C
        self.kW = kW

        opt = opt or {}
        self.hasBias = opt.hasBias or false -- default no Bias
    end
    check_arg()

    -- submodules: narrow + lookuptable
    local submds = {}
    for i = 1, kW do
        local offset = i
        local length = 1 -- set it as (M - kW + 1) at runtime
        submds[i] = nn.Sequential()
            -- B, M (,V)
            :add(nn.OneHotNarrowExt(V, 2,offset,length))
            -- B, M-kW+1 (,V)
            :add(nn.LookupTableExt(V,C))
            -- B, M-kW+1, C
    end

    -- multiplexer: send input to each submodule
    local ct = nn.ConcatTable()
    for i = 1, kW do
        ct:add(submds[i])
    end

    -- the container to be returned
    local inplace = true
    -- B, M (,V)
    self:add(ct)
    -- {B, M-kW+1, C}, {B, M-kW+1, C}, ...
    self:add(nn.CAddTable(inplace))
    -- B, M-kW+1, C
    if self.hasBias == true then
        self:add(nn.TemporalAddBias(C, inplace))
    end
    -- B, M-kW+1, C
end

function OneHotTemporalConvolution:setPadding(pv)
    local ms = self:findModules('nn.LookupTableExt')
    for _, m in ipairs(ms) do
        m:setPadding(pv)
    end
    return self
end

function OneHotTemporalConvolution:zeroPaddingWeight()
    local ms = self:findModules('nn.LookupTableExt')
    for _, m in ipairs(ms) do
        local paddingInd = m.paddingValue
        if paddingInd > 0 then
            m.weight:select(1, paddingInd):fill(0)
        end
    end
    return self
end

function OneHotTemporalConvolution:updateOutput(input)
    assert(input:dim()==2, "input size must be dim 2: B, M")

    -- need to the seq length for current input batch
    local M = input:size(2)
    assert(M >= self.kW,
        ("kernel size %d > seq length %d, failed"):format(self.kW, M)
    )
    self:_reset_seq_length(M)

    return parent.updateOutput(self, input)
end

--[[ Okay with default backward(), which calls each module's backward() ]]--

function OneHotTemporalConvolution:__tostring__()
    local s = string.format('%s(%d -> %d, %d',
        torch.type(self),  self.V, self.C, self.kW)
    return s .. ')'
end

-- additional methods
function OneHotTemporalConvolution:should_updateGradInput(flag)
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

function OneHotTemporalConvolution:index_copy_weight(vocabIdxThis, convThat, vocabIdxThat)
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

-- static functions
function OneHotTemporalConvolution.share_weights(tOhConv, tKerPos)
    -- e.g.,
    -- share_weights( {ohconv1, ohconv2}, {1,1} ) -- share kernel at position 1
    -- share_weights( {ohconv1, ohconv2}, {1,2} ) -- share ohconv1 kernel at position 1 with ohconv2 kernel at postition 2
    -- share_weights( {ohconv1, ohconv2, ohconv3}, {1,1,1} ) --

    assert(type(tOhConv) == 'table' and type(tKerPos) == 'table')
    assert(#tOhConv == #tKerPos)

    -- this: the base one
    local kerPosThis = tKerPos[1]
    local ohConvThis = tOhConv[1]
    local lookupThis = ohConvThis:get(1):get(kerPosThis):get(2)

    -- that: the others
    for i = 2, #tOhConv do
        local kerPosThat =tKerPos[i]
        local ohConvThat = tOhConv[i]
        local lookupThat = ohConvThat:get(1):get(kerPosThat):get(2)

        lookupThat:share(lookupThis, 'weight', 'gradWeight')
    end

end

-- helpers
function OneHotTemporalConvolution:_reset_seq_length(M)
    local contable = self.modules[1]
    local kW = #contable.modules
    for i = 1, kW do
        -- reset nn.Narrow length
        local length = M -kW + 1
        contable.modules[i].modules[1].length = length
    end
end