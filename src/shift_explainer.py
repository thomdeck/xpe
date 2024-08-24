"""
Copyright 2024 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import torch
import ot
from scipy.stats import ks_2samp
from typing import Union, Dict, Tuple, Callable
from abc import ABC
from src.base_explainer import BaseExplainer, KernelShap

class ShiftExplainer(ABC):
    """
    Abstract class for all shift explainer classes explaining the model behaviour under distributtion shifts.
    """

    def __init__(self, 
                 model_func: Callable[[np.array], np.array],
                 explainer: Union[BaseExplainer, None] = None,
                 abs = False):
        
        self.model_func = model_func
        self.abs = abs

        if explainer is None:
            self.explainer = KernelShap(model_func, abs = abs)
        else:
            self.explainer = explainer
        
    
    def explain(self):
        pass
    
    def f_softmax(self, x, target):
        y=self.model_func(x)
        y= (np.exp(y - np.max(y)) / np.exp(y- np.max(y)).sum())[:,target]
        return y
    
    def f_target_logit(self, x, target):
        y=self.model_func(x)[:,target]
        return y

    def f_entropy(self, x):
        y=self.model_func(x)
        y= (np.exp(y - np.max(y, axis=-1, keepdims=True)) / np.exp(y- np.max(y, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True))
        return -np.sum(y * np.log(y+ 1e-12), axis=-1)[..., np.newaxis]
    
    def f_cross_entropy(self, x, label):
        y =self.model_func(x)
        y= (np.exp(y - np.max(y, axis=-1, keepdims=True)) / np.exp(y- np.max(y, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True))
        return - np.log(y[:, label]+1e-12)[..., np.newaxis]


class ShiftExplainerMatching(ShiftExplainer):
    """
    Abstract class for all shift explainer classes that are based on matching source and target samples.
    """

    def __init__(self, 
                 model_func: Callable[[np.array], np.array],
                 matcher: ot.da.BaseTransport,
                 Xs: np.array,
                 Xt: np.array,
                 explainer: BaseExplainer,
                 abs: bool = False):
        
        super().__init__(model_func, explainer, abs)
       
        self.matcher = matcher
        self.coupling = self.match(Xs, Xt)
        self.idx_match = np.argmax(self.coupling, axis=0)


    def match(self,
              Xs: np.array,
              Xt: np.array,
              ) -> Tuple[np.array, np.array]:
        """Match source and target samples with underlying matcher

        Args:
            Xs (np.array): Source samples of shape (n_samples, channel, height, width)
            Xt (np.array): Target samples of shape (n_samples, channel, height, width)

        Returns:
             np.array: coupling resulting from matching source and target samples
        """
        self.matcher.fit(Xs=Xs, Xt=Xt)

        return self.matcher.coupling_ 



class AttributionXShift(ShiftExplainer):
    """
    Class for Attribution x Shift (AxS) from Decker et al. (2024)
    """

    def __init__(self, 
                 model_func: Callable[[np.array], np.array],
                 Xs: np.array,
                 Xt: np.array,
                 explainer: BaseExplainer,
                 confidence: float= 0.95,
                 abs: bool = False):
        

        super().__init__(model_func, explainer, abs)

  
        self.confidence = confidence
        self.mask = self.get_shift_mask(Xs, Xt, confidence)

    def explain(self,
                inputs: np.array,
                baseline: Union[np.array, None] = None,
                **kwargs) -> np.array:
        
        if baseline is None:
            baseline = np.zeros_like(inputs[:1])

        # get the target class and forward function
        y = np.argmax(self.model_func(inputs))
        forward_fn = lambda x: self.f_target_logit(x, y)

        # build explainer
        explainer = self.explainer.build_explainer(forward_fn, baseline = baseline)

        exp = explainer.explain(inputs, **kwargs)

        return exp * self.mask

    def get_shift_mask(self,
                       Xs: np.array,
                       Xt: np.array,
                       confidence_threshold: 0.95,
                       **kwargs) -> np.array:
        

        pval, _ = self.KStest(Xs, Xt, **kwargs)
        mask = pval < 1 - confidence_threshold

        return mask.astype(np.float32)


    def KStest(self, Xs, Xt, **kwargs):
        """
        Compute two sided Kolmogorov-Smirnov test for each feature between source and target samples
        (see also https://github.com/SeldonIO/alibi-detect/blob/master/alibi_detect/cd/ks.py for reference)

        Args:
            Xs (np.array): Source samples of shape (n_samples, n_features)
            Xt (np.array): Target samples of shape (n_samples, n_features)
            **kwargs: Additional arguments for ks_2samp
        
        Returns:
            Tuple[np.array, np.array]: p-values and distances of the Kolmogorov-Smirnov test for each feature

        """
        Xt = Xt.reshape(Xt.shape[0], -1)
        Xs = Xs.reshape(Xs.shape[0], -1)
        n_features = Xs.shape[1]
        p_val = np.zeros(n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(n_features):
            dist[f], p_val[f] = ks_2samp(Xs[:, f], Xt[:, f], **kwargs)
        
        return p_val, dist
    


class LocalAttributionDifference(ShiftExplainerMatching):
    """
    Class for Local Attribution Difference (LAD) from Decker et al. (2024)
    """

    def __init__(self, 
                 model_func: Callable[[np.array], np.array],
                 matcher: ot.da.BaseTransport,
                 Xs: np.array,
                 Xt: np.array,
                 explainer: Union[BaseExplainer, None],
                 abs: bool = False):
        
        super().__init__(model_func, matcher, Xs, Xt, explainer, abs)

        self.Xs = Xs

    def explain(self,
                inputs: np.array,
                idx: int,
                baseline: Union[np.array, None] = None,
                **kwargs) -> Dict:
        

        if baseline is None:
            baseline = np.zeros_like(inputs[:1])

        # get the matched source sample
        idx_s = self.idx_match[idx]
        x_s = self.Xs[idx_s][np.newaxis, :]

        # get the target class and forward function
        y = np.argmax(self.model_func(x_s))
        forward_fn = lambda x: self.f_target_logit(x, y)

        # build explainer
        explainer = self.explainer.build_explainer(forward_fn, baseline = baseline)

        # get explanation for source sample
        exp_s = explainer.explain(x_s, **kwargs)[0]


        # get the target class and forward function
        y = np.argmax(self.model_func(inputs))
        forward_fn = lambda x: self.f_target_logit(x, y)

        # build explainer
        explainer = self.explainer.build_explainer(forward_fn, baseline = baseline)

        # get explanation for target sample
        exp_t = explainer.explain(inputs, **kwargs)

        return np.abs(exp_t - exp_s)
        


class XPE(ShiftExplainerMatching):
    """
    Class for XPE (Explanatory Performance Estimation) from Decker et al. (2024)
    """

    def __init__(self, 
                 model_func: Callable[[np.array], np.array],
                 matcher: ot.da.BaseTransport,
                 Xs: np.array,
                 ys: np.array,
                 Xt: np.array,
                 explainer: BaseExplainer,
                 abs: bool = False):
        
        super().__init__(model_func, matcher, Xs, Xt, explainer, abs)
        self.Xs = Xs
        self.ys = ys


    def explain(self,
                inputs: np.array,
                idx: int,
                mode: str = 'best_match',
                **kwargs
                ) -> np.array:
        
        """

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): sample to be explained
            idx (int): index of the sample in Xt
            mode (str): currently only best_match is implemented that computes explanations based on 
                        the best matching source sample

        Returns:
             np.ndarray: Shift Explanation
        
        """

        if mode == 'best_match':
            return self.explain_with_best_match(inputs, idx, **kwargs)
            
        else:
            raise ValueError(f"Unknown mode {mode}. Choose from ['best_match']")

    def explain_with_best_match(self,inputs, idx, **kwargs):

        # get the matched source sample
        idx_s = self.idx_match[idx]
        xs = self.Xs[idx_s][np.newaxis, :]

        # get the anticipated target class from matched source sample to estimate the loss
        y = self.ys[idx_s]
        forward_fn = lambda x: self.f_cross_entropy(x, label=y)

        # build explainer
        explainer = self.explainer.build_explainer(forward_fn, baseline = xs)

        # get explanation for source sample
        return explainer.explain(inputs, **kwargs)
        
    

class XPPE(ShiftExplainerMatching):
    """
    Class for XPPE (Explanatory Performance Proxy Estimation) from Decker et al. (2024)
    """

    def __init__(self, 
                 model_func: Callable[[np.array], np.array],
                 matcher: ot.da.BaseTransport,
                 Xs: np.array,
                 Xt: np.array,
                 explainer: BaseExplainer,
                 abs: bool = False) -> None:
        
        super().__init__(model_func,matcher,Xs, Xt,  explainer, abs)
        self.Xs = Xs


    def explain(self,
                inputs: np.array,
                idx: int,
                mode: str = 'best_match',
                **kwargs
                ) -> np.array :
        
        """

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): sample to be explained
            idx (int): index of the sample in Xt
            mode (str): currently only best_match is implemented that computes explanations based on 
                        the best matching source sample

        Returns:
             np.ndarray: Shift Explanation
        
        """

        if mode == 'best_match':
            return self.explain_with_best_match(inputs, idx, **kwargs)
            
        else:
            raise ValueError(f"Unknown mode {mode}. Choose from ['best_match']")

    def explain_with_best_match(self, 
                                inputs: np.array,
                                idx: int,
                                **kwargs) -> np.array:

        # get the matched source sample
        idx_s = self.idx_match[idx]
        xs = self.Xs[idx_s][np.newaxis, :]


        # build explainer
        explainer = self.explainer.build_explainer(self.f_entropy, baseline = xs)

        # get explanation for source sample
        return explainer.explain(inputs, **kwargs)

