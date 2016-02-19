--- nn.NarrowExt
-- extend nn.Narrow in that
--   * the updateGradInput() can be turned off during bp()

local NarrowExt, parent = torch.class('nn.NarrowExt', 'nn.Narrow')

-- class def
function NarrowExt:__init(...)
    parent.__init(self, ...)
    self.flagUpdateGradInput = false
end

function NarrowExt:updateGradInput(input, gradOutput)
    if false == self.flagUpdateGradInput then
        -- simply return the default empty gradInput
        return self.gradInput
    else
        -- use the original method
        return parent.updateGradInput(self, input, gradOutput)
    end
end

-- additional methods
function NarrowExt:should_updateGradInput(flag)
    assert(flag==true or flag==false, "flag must be boolean!")
    self.flagUpdateGradInput = flag
end