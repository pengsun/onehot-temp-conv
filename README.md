# One-Hot Temporal Convolution as Torch's nn module

Implement the sparse (one-hot input) 1-dimensional (temporal) convolution defined in [1, 2]. 
For NLP task, it applies the convolution over the one-hot word vector directly that the word embedding can be ommited, as shwon in [1, 2].
Only support the "Narrow Convolution", i.e., the sequence size is reduced after "forward".
Convolution stride must be 1.
The exposed module is essentially a wrapper depending on `nn.LookupTable`, so that:

* Input must be a `Tensor` of word index pointing to the vocabulary
* Gradient of the inputs are unavailable (`gradInput` is a dummy variable) during `backward()` by default, since it saves a lot of training time while `gradInput` is not involved. Call the method `should_updateGradInput(flag)` to explicitly enable/disable it if `gradInput` is indeed desired/undesired. See explanations below.
* Both CPU and GPU are supported, depending on `require'nn'` or `require'cunn'`

Interfaces and tensor size layout are consistent with `nn.TemporalConvolution`.

The terms of `nn.TemporalConvolution` are borrowed here and are aliased as the following:
```
  B = batch size
  M = sequence length = nInputFrame = #words
  V = inputFrameSize = vocabulary size
  C = outputFrameSize = #output feature maps = #hidden units = embedding size
  kW = convolution kernel size = kernel width
```

## Prerequisites
* Torch 7


## Installation
* run command ```git clone https://github.com/pengsun/onehot-temp-conv```
* cd to the directory, run command ```luarocks make```

Then the lib will ba installed to your torch 7 directory. Delete the git-cloned source directory `onehot-temp-conv` if you like.


## Usage
After installation, running `require'onehot-temp-conv'` will add to the `nn` namespace the following classes:

### OneHotTemporalConvolution

#### Constructor:
```lua
module = nn.OneHotTemporalConvolution(inputFrameSize, outputFrameSize, kW)
```

Applies a 1D convolution over an input sequence composed of `nInputFrame` frames. The `input` tensor in
`forward(input)` must be a 2D tensor in size
```
BatchSize x nInputFrame
```
where each element is an index ranging from `1` to `InputFrameSize`.

The output will be sized
``` 
BatchSize x nOutputFrame x outputFrameSize
```
where `nOutputFrame = nInputFrame - kW + 1`.

The parameters are the following:
  * `inputFrameSize`: The input frame size expected in sequences given into `forward()`.
  * `outputFrameSize`: The output frame size the convolution layer will produce.
  * `kW`: The kernel width of the convolution, `kW <= nInputFrame` required.
See the example below for the NLP alias of these terms. 

Example 1:
```Lua
  require'onehot-temp-conv'
  
  B = 200 -- batch size
  M = 45 -- sequence length (#words)
  V = 12333 -- inputFrameSize (vocabulary size)
  C = 300 -- outputFrameSize (#output feature maps, or embedding size)
  kW = 5 -- convolution kernel size (width)
  
  -- inputs: the one-hot vector as index in set {1,2,...,V}. size: B, M
  inputs = torch.LongTensor(B, M):apply(
    function (e) return math.random(1,V) end
  )
  
  -- the 1d conv module
  tf = nn.OneHotTemporalConvolution(V, C, kW)
  
  -- outputs: the dense tensor. size: B, M-kW+1, C
  outputs = tf:forward(inputs)

  -- back prop: the gradients w.r.t. parameters
  gradOutputs = outputs:clone():normal()
  tf:backward(inputs, gradOutputs)
```

Example 2 (gpu):
```Lua
  require'cunn'
  require'onehot-temp-conv'
  
  B = 200 -- batch size
  M = 45 -- sequence length (#words)
  V = 12333 -- inputFrameSize (vocabulary size)
  C = 300 -- outputFrameSize (#output feature maps, or embedding size)
  kW = 5 -- convolution kernel size (width)
  
  -- inputs: the one-hot vector as index in set {1,2,...,V}. size: B, M
  inputs = torch.LongTensor(B, M):apply(
    function (e) return math.random(1,V) end
  ):cuda()
  
  -- the 1d conv module
  tf = nn.OneHotTemporalConvolution(V, C, kW):cuda()
  
  -- outputs: the dense tensor. size: B, M-kW+1, C
  outputs = tf:forward(inputs)

  -- back prop: the gradients w.r.t. parameters
  gradOutputs = outputs:clone():normal()
  tf:backward(inputs, gradOutputs)
```

#### Method:
```lua
OneHotTemporalConvolution:should_updateGradInput(flag)
```
Set if it should do updateGradInput (default to false while class construction). `flag` must be `true` or `false`

#### Method:
```lua
OneHotTemporalConvolution:index_copy_weight(vocabIdxThis, convThat, vocabIdxThat)
```
Copy the weight from another `OneHotTemporalConvolution` with the respective vocabulary index. 
`vocabIdxThis`, `vocabIdxThat` must be `torch.LongTensor`.
`convThat` must be also a `OneHotTemporalConvolution`.
Suppose the weight size
```
this: V1, C, p
that: V2, C, p
```
where the vocabulary size `V1` and `V2` can be different (but the outputFeatureMap `C` and region size `p` must be the same).
Then calling the method would in effect do the copying
```Matlab
this(vocabIdxThis, :, :) = that(vocabIdxThat, :, :)
```

#### A note
on `gradInput` and `updateGradInput()`. When `OneHotTemporalConvolution` is usually used as the first layer, the `gradInput` is usually unnecessary since it does not contribute to parameters updating during training. That's why it's default to dummy. When you do need it, just call `should_updateGradInput(true)`, and `gradInput` will be available after calling `updateGradInput()` or `backward()`. Use it with caution as `gradInput` is usually very large and demands much memory.

Example:
```Lua
  require'cunn'
  require'onehot-temp-conv'
  
  B = 1 -- batch size
  M = 225 -- sequence length (#words)
  V = 30*1000 -- inputFrameSize (vocabulary size)
  C = 300 -- outputFrameSize (#output feature maps, or embedding size)
  kW = 5 -- convolution kernel size (width)
  
  -- inputs: the one-hot vector as index in set {1,2,...,V}. size: B, M
  inputs = torch.LongTensor(B, M):apply(
    function (e) return math.random(1,V) end
  ):cuda()
  
  -- the 1d conv module
  tf = nn.OneHotTemporalConvolution(V, C, kW):cuda()
  
  -- outputs: the dense tensor. size: B, M-kW+1, C
  outputs = tf:forward(inputs)

  -- enable backprop for input
  tf:should_updateGradInput(true)

  -- back prop: the gradients w.r.t. parameters and inputs
  gradOutputs = outputs:clone():normal()
  gradInputs = tf:backward(inputs, gradOutputs)
  
  -- size should be B x M x V
  print(gradInputs:size())
```

#### A note
on parameters. When you need the kernel weight and its gradient, call `self:parameters()` or `self:getParameters()` - note that `OneHotTemporalConvolution` is derived from the container `nn.Sequential`.

### OneHotTemporalConvolutionDummyBP

#### Constructor:
```lua
module = nn.OneHotTemporalConvolutionDummyBP(ohConv)
```

Derived from `nn.Module`.
Initialize from a (pre-trained) `nn.OneHotTemporalConvolution`.
Only do forward propagation.
Back propagation is dummy in that 1) the parameters are invincible to outer module. 2) both the `gradInput` and `gradParameters` are not updated.
This module should work as a "feature extractor".

Example:
```lua
  require'nn'
  
  V, C, p = 33, 10, 2
  
  ohConv = nn.OneHotTemporalConvolution(V, C, p)
  fetext = nn.OneHotTemporalConvolutionNoBP(ohConv:float())
  
  -- fprop
  B, M = 12, 5
  inputs = torch.LongTensor(B, M):apply(
      function (e) return math.random(1,V) end
  ):float()
  outputs = fetext:forward(inputs)
  
  -- cannot get the parameters
  params, grad = fetext:getParameters()
  assert(params:numel()==0 and grad:numel()==0)
```

### OneHotNarrowExt
An auxiliary class. Extend `nn.Narrow` in that 

* interpret input as index of one-hot tensor
* the updateGradInput() can be turned off during bp()

### LookupTableExt
An auxiliary class. Extend `nn.LookupTable` in that

* can do the updateGradInput(), which is not implemented in `nn.LookupTable`
* the updateGradInput() can be turned off during backward()

##Reference
[1] Rie Johnson and Tong Zhang. Effective use of word order for text categorization with convolutional neural networks. NAACL-HLT 2015. 

[2] Rie Johnson and Tong Zhang. Semi-supervised convolutional neural networks for text categorization via region embedding. NIPS 2015.
