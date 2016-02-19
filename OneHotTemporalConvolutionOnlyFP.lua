--- One-hot Temporal Convolution, Only do Forward-Propagation
-- used as a feature extractor

require'torch'
require'nn'

--- classdef
local OneHotTemporalConvolutionOnlyFP, parent = torch.class('nn.OneHotTemporalConvolutionOnlyFP', 'nn.Module')

function OneHotTemporalConvolutionOnlyFP:__init(ohConv)
    assert(torch.type(ohConv) == 'nn.OneHotTemporalConvolution',
        "arg ohConv is an unexpected type " .. torch.type(ohConv) .. ", expected nn.OneHotTemporalConvolution"
    )

    parent.__init(self)

    self.md = ohConv:clone()

    collectgarbage()
end

function OneHotTemporalConvolutionOnlyFP:updateOutput(input)
    return self.md:updateOutput(input)
end

--- Okay with default OnehotTemporalConv:backward, which does nothing

function OneHotTemporalConvolutionOnlyFP:__tostring__()
    local s = string.format('%s(%d -> %d, %d',
        torch.type(self),  self.md.V, self.md.C, self.md.kW)
    return s .. ')'
end
