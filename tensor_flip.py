import modules.scripts
import modules.sd_hijack
import modules.shared
import gradio
import torch

from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from typing import Optional

class Script(modules.scripts.Script):

    def title(self):
        return "Tensor Flip"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        with gradio.Accordion("Tensor Flip", open=False):
            flip = gradio.Checkbox(False, label="Flip")
            startFlippingStep = gradio.Number(0, label="Start flipping from step N", precision=0)
            stopFlippingStep = gradio.Number(-1, label="Stop flipping after step N (-1: Don't stop)", precision=0)

        return [flip, startFlippingStep, stopFlippingStep]

    def process(self, p, flip, startFlippingStep, stopFlippingStep):
        if (flip):
            p.extra_generation_params = {
               "Flip": flip,
               "Start Flipping From Step": startFlippingStep,
               "Stop Flipping After Step": stopFlippingStep,
            }
            self.__hijackConv2DMethods(flip, startFlippingStep, stopFlippingStep)
        else:
            self.__restoreConv2DMethods()

    def postprocess(self, *args):
        self.__restoreConv2DMethods()

    def __hijackConv2DMethods(self, flip, startFlippingStep, stopFlippingStep):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer.flippingStartStep = startFlippingStep
                layer.flippingStopStep = stopFlippingStep
                layer._conv_forward = Script.__replacementConv2DConvForward.__get__(layer, Conv2d)

    def __restoreConv2DMethods(self):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer._conv_forward = Conv2d._conv_forward.__get__(layer, Conv2d)

    def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        step = modules.shared.state.sampling_step
        working = F.pad(input, self._reversed_padding_repeated_twice, mode='constant')
        if ((self.flippingStartStep < 0 or step >= self.flippingStartStep) and (self.flippingStopStep < 0 or step <= self.flippingStopStep)):
            working = working.flip(0)
        return F.conv2d(working, weight, bias, self.stride, (0, 0), self.dilation, self.groups)
