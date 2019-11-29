import ddm
import ddm.plot
import numpy as np
import pandas as pd
import paranoid as pns
import matplotlib.pyplot as plt
import seaborn as sns

def run_trial(condition, params, random_increments, n, t_max=4, dt=0.0001):
    tta, distance = condition
    alpha, boundary, tta_crit = params
    nondecision_time = 0.0
    
    t_values = np.arange(0, t_max, dt)
    drift_rate_values = np.zeros_like(t_values)
    evidence_values = np.zeros_like(t_values)
    
    for i, t in enumerate(t_values[1:]):
        drift_rate = alpha*(tta-t)
        drift_rate_values[i+1] = drift_rate
        
        evidence = (evidence_values[i] + drift_rate*dt + random_increments[i] 
                    if t>nondecision_time else 0)
        evidence_values[i+1] = evidence
            
        if (abs(evidence) > boundary):
            break
        
    is_turn_decision = (evidence > boundary)
        
    return pd.DataFrame(data={'n': n,
                              't': t_values[:i+2],
                              'x': evidence_values[:i+2],
                              'd': drift_rate_values[:i+2],
                              'is_turn_decision': is_turn_decision})
    

def run_sim(condition, params, n_sim=1000, t_max=4, dt=0.0001):
    random_increments = np.random.randn(n_sim, int(t_max/dt))*np.sqrt(dt)
    trials = [run_trial(condition, params, random_increments[n], n, t_max, dt) for n in range(n_sim)]
    sim_result = pd.concat(trials).set_index(['n'])
    tta, distance = condition
    sim_result['tta'] = tta
    sim_result['distance'] = distance
    
    return sim_result 


m = ddm.Model(drift=ddm.DriftConstant(drift=1.0),
              noise=ddm.NoiseConstant(noise=1.0),
              bound=ddm.BoundConstant(B=1.0), T_dur=4)

#ddm.plot.plot_decision_variable_distribution(m)

%time sol = m.solve()
#samp = sol.resample(100000)
#sim_sol = m.simulated_solution()
#sim_sol_e = m.simulated_solution(rk4=False)

# Takes 59.9 s with the old algorithm (call randn on each time step)
%time custom_sim_result = run_sim(condition=(4, 90), params=(1,1,1))

response_times = custom_sim_result[custom_sim_result.is_turn_decision].groupby('n').t.last()

fig, ax = plt.subplots()
#ax.plot(sim_sol_e.t_domain(), sim_sol_e.pdf_corr(), label='Euler')
#ax.plot(sim_sol.t_domain(), sim_sol.pdf_corr(), label='RK4')
#ax.plot(samp.t_domain(), samp.pdf_corr(), label='Bootstrap')
ddm.plot.plot_solution_pdf(sol, ax=ax)
sns.distplot(response_times, ax=ax, label='custom', hist=True, kde=False, hist_kws={'density': True})
ax.legend()

#exp_data = pd.read_csv('measures.csv').loc[:,['RT', 'is_turn_decision', 'tta_condition']]
#exp_data.loc[:, 'correct'] = exp_data.is_turn_decision.astype(int)
#sample = ddm.Sample.from_pandas_dataframe(df=exp_data, rt_column_name='RT', correct_column_name='correct')
#
#fitted_model = ddm.fit_model(sample=sample, drift=ddm.DriftConstant(drift=ddm.Fittable(minval=0, maxval=2)),
#                             bound=ddm.BoundConstant(B=ddm.Fittable(minval=0, maxval=2)))
#ddm.display_model(fitted_model)
#
#ddm.plot.model_gui(fitted_model, sample=sample)

#    * Threshold on noisy percept of tau-tau_crit + non-decision time
#    * DDM with linearly collapsing bound
#    * DDM with exponentially collapsing bound    
