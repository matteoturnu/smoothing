import torch
from scipy.stats import norm, binomtest
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, device, L: str) -> (int, float):
    """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 or Linf radius.
    With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
    robust within a L2 or Linf ball of radius R around x.

    :param x: the input [channel x height x width]
    :param n0: the number of Monte Carlo samples to use for selection
    :param n: the number of Monte Carlo samples to use for estimation
    :param alpha: the failure probability
    :param batch_size: batch size to use when evaluating the base classifier
    :param L: the norm to use ('L2' or 'Linf')
    :return: (predicted class, certified radius)
             in the case of abstention, the class will be ABSTAIN and the radius 0.
    """
    self.base_classifier.eval()
    
    if L == 'L2':
        counts_selection = self._sample_noise(x, n0, batch_size, device)
        counts_estimation = self._sample_noise(x, n, batch_size, device)
    elif L == 'Linf':
        counts_selection = self._sample_noise_linf(x, n0, batch_size, device)
        counts_estimation = self._sample_noise_linf(x, n, batch_size, device)
    else:
        raise ValueError("Norm type not recognized. Use 'L2' or 'Linf'.")
    
    # Guess at the top class
    cAHat = counts_selection.argmax().item()
    
    # Estimate a lower bound on pA
    nA = counts_estimation[cAHat].item()
    pABar = self._lower_confidence_bound(nA, n, alpha)
    
    if pABar < 0.5:
        return Smooth.ABSTAIN, 0.0
    else:
        if L == 'L2':
            radius = self.sigma * norm.ppf(pABar)
        elif L == 'Linf':
            radius = self.sigma * norm.ppf(pABar) / np.sqrt(x.numel())  # Adjust radius for Linf
        return cAHat, radius

def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int, device, L: str) -> int:
    """ Monte Carlo algorithm for evaluating the prediction of g at x under L2 or Linf norm.  
    With probability at least 1 - alpha, the class returned by this method will equal g(x).

    :param x: the input [channel x height x width]
    :param n: the number of Monte Carlo samples to use
    :param alpha: the failure probability
    :param batch_size: batch size to use when evaluating the base classifier
    :param L: the norm to use ('L2' or 'Linf')
    :return: the predicted class, or ABSTAIN
    """
    self.base_classifier.eval()
    
    if L == 'L2':
        counts = self._sample_noise(x, n, batch_size, device)
    elif L == 'Linf':
        counts = self._sample_noise_linf(x, n, batch_size, device)
    else:
        raise ValueError("Norm type not recognized. Use 'L2' or 'Linf'.")
    
    top2 = counts.argsort()[::-1][:2]
    count1 = counts[top2[0]]
    count2 = counts[top2[1]]
    
    if binomtest(count1, count1 + count2, p=0.5).pvalue > alpha:
        return Smooth.ABSTAIN
    else:
        return top2[0]
    
    def _sample_noise(self, x: torch.tensor, num: int, batch_size, device) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device=device) * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _sample_noise_linf(self, x: torch.tensor, num: int, batch_size, device) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x using Linf noise.
    
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
    
                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = (torch.rand_like(batch, device=device) * 2 - 1) * self.sigma  # Uniform noise in [-sigma, sigma]
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
