## One-Hot Temporal Convolution as Torch's nn module

Implement the sparse(one-hot input) 1-dimensional (temporal) convolution defined in [1, 2]. 
For NLP task, it directly takes as input the one-hot word vector and do the convolution.
Only support the "Narrow Convolution", i.e., the sequence size is reduced after "forward".
Stride is not allowed.

Interfaces, terms and tensor size layout are consistent with `nn.TemporalConvolution`.

The exposed module is essentially a wrapper depending on `nn.LookupTable`, so that:
* Input must be a `Tensor` of word index pointing to the vocabulary
* Gradient of the inputs are unavailable
* Both CPU and GPU are supported, depending on `require'nn'` or `require'cunn'`


### Prerequisites
* Torch 7


### Installation
* run command ```git clone https://github.com/pengsun/onehot-temp-conv```
* cd to the directory, run command ```luarocks make```

Then the lib will ba installed to your torch 7 directory. Delete the git-cloned source directory `onehot-temp-conv` if you like.


### Usage
After installation, running `require'onehot-temp-conv'` will add to the `nn` namespace the following classes:

## NarrowNoBP ##
Derived from `nn.Narrow`, but it doesn't change `gradInput` during back-propagation. An auxiliary class.

## OneHotTemporalConvolution ##
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

Example:

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

###Reference
[1] Rie Johnson and Tong Zhang. Effective use of word order for text categorization with convolutional neural networks. NAACL-HLT 2015. 

[2] Rie Johnson and Tong Zhang. Semi-supervised convolutional neural networks for text categorization via region embedding. NIPS 2015.
