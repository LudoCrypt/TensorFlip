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
            if option[0] == 'flip':
                if (int(option[1]) <= step <= int(option[2])) or (int(option[1]) < 0 or int(option[2]) < 0):
                    working = working.flip(0)
            if option[0] == 'shuffle':
                if (int(option[1]) <= step <= int(option[2])) or (int(option[1]) < 0 or int(option[2]) < 0):
                    working = working[torch.randperm(working.size(0))]

        return F.conv2d(working, weight, bias, self.stride, (0, 0), self.dilation, self.groups)
