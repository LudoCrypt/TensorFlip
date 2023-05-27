# TensorFlip
Flip Tensors in Stable Diffusion
# Asymmetric Tiling
The regular script is incompatible with the [Asymmetric Tiling](https://github.com/tjm35/asymmetric-tiling-sd-webui/tree/main) script, so I've included one that merges the two scripts code

If you're gonna use the merged version of the script, make sure the normal Asymmetric Tiling script isn't in the scripts folder, make sure that the regular version of THIS script is also not in the scripts folder.

# Prompt Settings
If you prefer, you can use the tensor_flip_prompt_set.py which is a variant that allows you to control the settings within the prompt, allowing for use in x/y/z grids. The syntax is `<flip:0:1>` where 0 and 1 are your start and stopping steps respectively. You may also use a new Shuffle operation, `<shuffle:0:1>` where 0 and 1 are the start and stop steps respectively. This randomly shuffles the tensor based on the seed, helps create wildly different, yet similar resilts!
