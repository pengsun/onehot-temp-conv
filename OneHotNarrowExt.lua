--- nn.OneHotNarrowExt
-- extend nn.Narrow in that
--   * the size interpretation
--        input: B, M (,V)
--        gradInput: B, M, V
--   * the updateGradInput() can be turned off during bp()

local OneHotNarrowExt, parent = torch.class('nn.OneHotNarrowExt', 'nn.Narrow')

-- class def
function OneHotNarrowExt:__init(V, dim, length, off)
    self.V = V or error('no V')
    parent.__init(self, dim, length, off)
    self.flagUpdateGradInput = false
end

-- Okay with the parent updateOutput()

function OneHotNarrowExt:updateGradInput(input, gradOutput)
    if false == self.flagUpdateGradInput then -- make it null
        self.gradInput = torch.Tensor():typeAs(gradOutput)
    else -- do it really, as a dense high dimensional tensor
        -- input: B, M1,...,Mk (, V)
        -- gradInput: B, M1,...,Mk, V
        self.gradInput = self.gradInput:typeAs(gradOutput)

        local inputSize = input:size():totable()
        table.insert(inputSize, self.V)
        self.gradInput:resize(table.unpack(inputSize)):zero()

        self.gradInput:narrow(self.dimension,self.index,self.length):copy(gradOutput)
    end
    return self.gradInput
end

-- additional methods
function OneHotNarrowExt:should_updateGradInput(flag)
    assert(flag==true or flag==false, "flag must be boolean!")
    self.flagUpdateGradInput = flag
end