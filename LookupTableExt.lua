--- nn.LookupTableExt
-- extend nn.LookupTable in that
--   * can do the updateGradInput(), which is not implemented in nn.LookupTable
--   * the updateGradInput() can be turned off during bp()

local LookupTableExt, parent = torch.class('nn.LookupTableExt', 'nn.LookupTable')

-- class def
function LookupTableExt:__init(...)
    assert(table.pack(...).n == 2, "currently only supports initialization nn.LookuptTableExt(V, C)")
    parent.__init(self, ...)
    self.flagUpdateGradInput = false
end

function LookupTableExt:updateGradInput(input, gradOutput)
    local function check_size()
        assert(input:dim()==2, "input must be sized (B, M)")
        assert(gradOutput:dim()==3, "gradOutput must be sized (B, M, C)")

        local B, M, C = gradOutput:size(1), gradOutput:size(2), gradOutput:size(3)
        assert(B == input:size(1), "input, gradOutput size not match")
        assert(M == input:size(2), "input, gradOutput size not match")
        assert(C == self.weight:size(2))

        local V = self.weight:size(1)

        return B, M, V, C
    end

    if false == self.flagUpdateGradInput then -- make it null
        self.gradInput = torch.Tensor():type(gradOutput:type())
    else -- do it really
        -- gradInput: B, M, V
        -- weight: V, C
        -- gradOutput: B, M, C
        local B, M, V, C = check_size()
        -- make them 2D matrix
        local dydy = gradOutput:reshape(B*M, C)
        local ww = self.weight:transpose(1,2)
        -- matrix matrix multiplication
        local dxdx = torch.mm(dydy, ww)
        -- restore the size
        self.gradInput = torch.reshape(dxdx, B, M, V):contiguous()
    end

    return self.gradInput
end

-- additional methods
function LookupTableExt:should_updateGradInput(flag)
    assert(flag==true or flag==false, "flag must be boolean!")
    self.flagUpdateGradInput = flag
end