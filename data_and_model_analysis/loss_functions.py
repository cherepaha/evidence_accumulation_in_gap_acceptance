import numpy as np
from scipy import interpolate, optimize
import ddm
import pandas as pd

class LossWLS(ddm.LossFunction):
    name = "Weighted least squares as described in Ratcliff & Tuerlinckx 2002"
    rt_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    rt_q_weights = [2, 2, 1, 1, 0.5]

    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur

    def get_rt_quantiles(self, x, t_domain, exp=False):
        cdf = x.cdf_corr(T_dur=self.T_dur, dt=self.dt) if exp else x.cdf_corr()
        cdf_interp = interpolate.interp1d(t_domain, cdf / cdf[-1])
        # If the model produces very fast RTs, interpolated cdf(0) can be >0.1, then we cannot find root like usual
        # In this case, the corresponding rt quantile is half of the time step of cdf
        rt_quantile_values = [optimize.root_scalar(lambda x: cdf_interp(x) - quantile, bracket=(0, t_domain[-1])).root
                              if (cdf_interp(0) < quantile) else self.dt / 2
                              for quantile in self.rt_quantiles]
        return np.array(rt_quantile_values)

    def loss(self, model):
        solultions = self.cache_by_conditions(model)
        WLS = 0
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            c = frozenset(comb.items())
            #            print(c)
            comb_sample = self.sample.subset(**comb)
            WLS += 4 * (solultions[c].prob_correct() - comb_sample.prob_correct()) ** 2
            self.comb_rts = pd.DataFrame([[item[0], item[1]["subj_id"]] for item in comb_sample.items(correct=True)],
                                         columns=["RT", "subj_id"])

            # Sometimes model p_correct is very close to 0, then RT distribution is weird, in this case ignore RT error
            if ((solultions[c].prob_correct() > 0.001) & (comb_sample.prob_correct() > 0)):
                model_rt_q = self.get_rt_quantiles(solultions[c], model.t_domain(), exp=False)
                exp_rt_q = self.get_rt_quantiles(comb_sample, model.t_domain(), exp=True)
                WLS += np.dot((model_rt_q - exp_rt_q) ** 2, self.rt_q_weights) * comb_sample.prob_correct()
        return WLS


class LossWLSVincent(LossWLS):
    name = """Weighted least squares as described in Ratcliff & Tuerlinckx 2002, 
                fitting to the quantile function vincent-averaged per subject (Ratcliff 1979)"""

    def get_rt_quantiles(self, x, t_domain, exp=False):
        if exp:
            vincentized_quantiles = (self.comb_rts.groupby("subj_id")
                                     .apply(lambda group: np.quantile(a=group.RT, q=self.rt_quantiles))).mean()
            return vincentized_quantiles
        else:
            return super().get_rt_quantiles(x, t_domain, exp=False)
