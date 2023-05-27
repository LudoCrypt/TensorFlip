# TensorFlip
Flip Tensors in Stable Diffusion
# Asymmetric Tiling Version
The regular script is incompatible with the [Asymmetric Tiling](https://github.com/tjm35/asymmetric-tiling-sd-webui/tree/main) script, so I've included one that merges the two scripts code

If you're gonna use the merged version of the script, make sure the normal Asymmetric Tiling script isn't in the scripts folder, make sure that the regular version of THIS script is also not in the scripts folder.

# Prompt Version
If you prefer, you can use the tensor_flip_prompt_set.py which is a variant that allows you to control the settings within the prompt, allowing for use in x/y/z grids.

## Operations
There are two operations, flipping, and shuffling. Flipping is the basic one, modifies the image slightly per step. Shuffling randomly shuffles the tensor based on the seed, can majorly affect the image.

## Syntax
The basic syntax is `<flip:A:B:C:D>` or `<shuffle:A:B:C:D>`.

`<flip:A>` starts flipping from step A.

`<flip:A:B>` flips in the range A (inclusive) -> B (exclusive).

`<flip:A:B:C>` flips every Cth step within the range.

`<flip:A:B:C:D>` flips in blocks of steps, where the size of each block is D, and the space between blocks is C
