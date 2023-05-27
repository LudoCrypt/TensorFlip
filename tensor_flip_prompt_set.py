import modules.scripts
import modules.sd_hijack
import modules.shared
import torch
import re

from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from typing import Optional

import modules.scripts as scripts

class Script(scripts.Script):

    def title(self):
        return "Tensor Flip"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process(self, p):
        extracted = Script.extract(p.prompt)
        p.all_prompts = [Script.clean(string) for string in p.all_prompts]
        if (Script.check(extracted, 'flip') or Script.check(extracted, 'shuffle')):
            p.extra_generation_params = {
                "Tensor Operations": extracted
            }
            self.__hijackConv2DMethods(p, extracted)
        else:
            self.__restoreConv2DMethods()

    def postprocess(self, *args):
        self.__restoreConv2DMethods()
        
    def extract(prompt):
        pattern = r'<(\w+(?::\w+)*)>'
        matches = re.findall(pattern, prompt)
        extracted_options = [tuple(match.split(':')) for match in matches]
        return extracted_options
    
    def check(extracted, setting):
        for option in extracted:
            if option[0] == setting:
                return True
        return False

    def clean(prompt):
        pattern = r'<\w+(?::\w+)*>'
        cleaned_prompt = re.sub(pattern, '', prompt)
        return cleaned_prompt

    def __hijackConv2DMethods(self, p, settings):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer.settings = settings
                layer._conv_forward = Script.__replacementConv2DConvForward.__get__(layer, Conv2d)

    def __restoreConv2DMethods(self):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer._conv_forward = Conv2d._conv_forward.__get__(layer, Conv2d)

    def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        step = modules.shared.state.sampling_step
        working = F.pad(input, self._reversed_padding_repeated_twice, mode='constant')
        for option in self.settings:
                name = option[0]
                params = tuple(int(x) for x in option[1:])
                if len(params) == 1:
                    if step >= params[0]:
                        if name == 'flip':
                            working = working.flip(0)
                        if name == 'shuffle':
                            working = working[torch.randperm(working.size(0))]
                elif len(params) == 2:
                    if params[0] <= step < params[1]:
                        if name == 'flip':
                            working = working.flip(0)
                        if name == 'shuffle':
                            working = working[torch.randperm(working.size(0))]
                elif len(params) == 3:
                    if params[0] <= step < params[1] and (step - params[0]) % (params[2] + 1) == 0:
                        if name == 'flip':
                            working = working.flip(0)
                        if name == 'shuffle':
                            working = working[torch.randperm(working.size(0))]
                elif len(params) == 4:
                    start = params[0]
                    end = params[1]
                    num_steps = params[3]
                    block_size = params[2]
                    block_index = (step - start) // (num_steps + block_size)
                    block_start = start + block_index * (num_steps + block_size)
                    block_end = block_start + num_steps
                    if block_start <= step < block_end and step < end and step >= start:
                        if name == 'flip':
                            working = working.flip(0)
                        if name == 'shuffle':
                            working = working[torch.randperm(working.size(0))]

        return F.conv2d(working, weight, bias, self.stride, (0, 0), self.dilation, self.groups)
