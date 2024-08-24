"""
Copyright 2024 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import torch
import shap
from typing import Union, Dict, List, Tuple, Callable
from abc import ABC



class BaseExplainer(ABC):
    """
    Abstract class for all standard explainers computing feature importances for model predictions.
    """
    def __init__(self, 
                 model_func: Callable[[np.array], np.array],
                 abs = False):
        
        self.model_func = model_func
        self.abs = abs


    def explain(self,
                inputs: np.array,
                **kwargs) -> np.array:
        
        pass

    def build_explainer(self):
        pass

    
    def f_softmax(self, x, target):
        y=self.model_func(x)
        y= (np.exp(y - np.max(y, axis=-1, keepdims=True)) / np.exp(y- np.max(y, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True))[:,target]
        return y
    
    def f_target_logit(self, x, target):
        y=self.model(x)[:,target]
        return y



class KernelShap(BaseExplainer):
    """
    Class for Kernel SHAP explainer as wrapper around the one provided in the shap library.
    """

    def __init__(self, 
                 model_func: Callable[[np.array], np.array],
                 abs: bool = False,
                 baseline: np.array = None,
                 explainer_kwargs: Dict = {}):
        
        super().__init__(model_func, abs)
        self.explainer_kwargs = explainer_kwargs

        if baseline is not None:
            self.explainer = self.build_explainer(model_func, baseline = baseline)
        else:
            self.explainer = None

    def explain(self,
                inputs: np.array,
                **kwargs) -> Dict:
        

        explanation = self.explainer.shap_values(inputs, **kwargs)

        if self.abs:
            explanation = np.abs(explanation)

        return explanation
    
    def build_explainer(self, 
                        forward_fn: Callable[[np.array], np.array],
                        baseline: np.array) -> shap.KernelExplainer:
        
        return shap.KernelExplainer(forward_fn, baseline, **self.explainer_kwargs)

    


