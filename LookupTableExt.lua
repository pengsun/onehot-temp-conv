--- nn.LookupTableExt
-- extend nn.LookupTable in that
--   * can do the updateGradInput(), which is not implemented in nn.LookupTable
--   * the updateGradInput() can be turned off during bp()

local LookupTableExt, parent = torch.class('nn.LookupTableExt', 'nn.LookupTable')

-- class def
function LookupTableExt:__init(...)
    --TODO: check arg, only support (V, C)
    parent.__init(self, ...)
    self.flagUpdateGradInput = false
end

function LookupTableExt:updateGradInput(input, gradOutput)
    if false == self.flagUpdateGradInput then
        -- simply returns the default null gradInput
        return self.gradInput
    else
        -- do it TODO
    end
end

-- additional methods
function LookupTableExt:should_updateGradInput(flag)
    assert(flag==true or flag==false, "flag must be boolean!")
    self.flagUpdateGradInput = flag
end
