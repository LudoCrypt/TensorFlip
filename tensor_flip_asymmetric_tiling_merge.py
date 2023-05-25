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
        return "Tensor Flip + Asymmetric Tiling"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        with gradio.Accordion("Tensor Flip + Asymmetric Tiling", open=False):
            flip = gradio.Checkbox(False, label="Flip")
            startFlippingStep = gradio.Number(0, label="Start flipping from step N", precision=0)
            stopFlippingStep = gradio.Number(-1, label="Stop flipping after step N (-1: Don't stop)", precision=0)
            tileActive = gradio.Checkbox(False, label="Active")
            tileX = gradio.Checkbox(True, label="Tile X")
            tileY = gradio.Checkbox(False, label="Tile Y")
            startStep = gradio.Number(0, label="Start tiling from step N", precision=0)
            stopStep = gradio.Number(-1, label="Stop tiling after step N (-1: Don't stop)", precision=0)

        return [flip, startFlippingStep, stopFlippingStep, tileActive, tileX, tileY, startStep, stopStep]

    def process(self, p, flip, startFlippingStep, stopFlippingStep, tileActive, tileX, tileY, startStep, stopStep):
        if (flip or tileActive):
            if (tileActive and flip):
                p.extra_generation_params = {
                    "Flip": flip,
                    "Start Flipping From Step": startFlippingStep,
                    "Stop Flipping After Step": stopFlippingStep,
                    "Tile X": tileX,
                    "Tile Y": tileY,
                    "Start Tiling From Step": startStep,
                    "Stop Tiling After Step": stopStep,
                }
            else:
                if (flip):
                    p.extra_generation_params = {
                        "Flip": flip,
                        "Start Flipping From Step": startFlippingStep,
                        "Stop Flipping After Step": stopFlippingStep,
                    }
                else:
                    p.extra_generation_params = {
                        "Tile X": tileX,
                        "Tile Y": tileY,
                        "Start Tiling From Step": startStep,
                        "Stop Tiling After Step": stopStep,
                    }
            self.__hijackConv2DMethods(flip, startFlippingStep, stopFlippingStep, tileActive, tileX, tileY, startStep, stopStep)
        else:
            self.__restoreConv2DMethods()

    def postprocess(self, *args):
        self.__restoreConv2DMethods()

    def __hijackConv2DMethods(self, flip, startFlippingStep, stopFlippingStep, tileActive, tileX, tileY, startStep, stopStep):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer.flipping = flip
                layer.tileActive = tileActive
                if (flip):
                    layer.flippingStartStep = startFlippingStep
                    layer.flippingStopStep = stopFlippingStep
                if (tileActive):
                    layer.padding_modeX = 'circular' if tileX else 'constant'
                    layer.padding_modeY = 'circular' if tileY else 'constant'
                    layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
                    layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
                    layer.paddingStartStep = startStep
                    layer.paddingStopStep = stopStep
                layer._conv_forward = Script.__replacementConv2DConvForward.__get__(layer, Conv2d)

    def __restoreConv2DMethods(self):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer._conv_forward = Conv2d._conv_forward.__get__(layer, Conv2d)

    def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        step = modules.shared.state.sampling_step
        if (self.tileActive and ((self.paddingStartStep < 0 or step >= self.paddingStartStep) and (self.paddingStopStep < 0 or step <= self.paddingStopStep))):
            working = F.pad(input, self.paddingX, mode=self.padding_modeX)
            working = F.pad(working, self.paddingY, mode=self.padding_modeY)
        else:
            working = F.pad(input, self.paddingX, mode='constant')
            working = F.pad(working, self.paddingY, mode='constant')
        if (self.flipping and ((self.flippingStartStep < 0 or step >= self.flippingStartStep) and (self.flippingStopStep < 0 or step <= self.flippingStopStep))):
            working = working.flip(0)
        return F.conv2d(working, weight, bias, self.stride, (0, 0), self.dilation, self.groups)
