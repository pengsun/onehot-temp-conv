--- nn.Narrow without Back Propagation
local NarrowNoBP, parent = torch.class('nn.NarrowNoBP', 'nn.Narrow')

function NarrowNoBP:__init(...)
    parent.__init(self, ...)
end

function NarrowNoBP:updateGradInput(input, gradOutput)
    -- simply return the default empty gradInput
    return self.gradInput
end

