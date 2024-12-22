# REINFORCE for Predictive Maintenance - Empirical study of RL algorithms
- 23-Dec-2024: 1:26 AM

------------------------------------
### R-2: 5 - 17: Latest LR

	( "automated reinforcement learning" OR "auto RL" OR "AutoRL" OR "Auto RL" ) AND ( "predictive maintenance" )
	Scopus and WoS = 0 for Auto RL
	( "automated machine learning" OR "auto ML" ) AND ( "predictive maintenance" )
	Scopus: 21 and WoS: 13 for Auto ML
Stats	
	"Conference Paper: 14 -- Article: 7"

	A typical AutoML - such DataRobot (Krzywanski, 2024)
\cite{OLeary2023}
{Thessen2016}"	O'Leary 2023  - conducts a comprehensive review on AutoML of 22 libraries. The number of academic citations relating to a project may be a useful metric for future works, although this is limited to measuring the activity of publishing researchers and may exclude many industry-based practitioners
{krzywanski2024}	original Bahri et al., 2022  (in O'Leary 2023) " AutoML can incorporate many phases of a typical ML pipeline including data cleaning, data augmentation, feature selection, model training, HYPERPARAMETER optimization and Architecture Optimization, 

------------------------------------
### R-2: 9-22 support for methodology

AutoML systems tend to be highly configurable, although increased configuration can be overwhelming for some practitioners (Thessen, 2016).
	
Tornede2020AutoML	"ML-Plan-RUL
- Claims NO available Auto ML for timeseries
- Issue with standard algos: Timeseries RUL prob has varying length of time - so converts to fixed length"
------------------------------------
	
### R-1: 2.3; time

"{Ganeshkumar2020} - Minimising Processing Time when Training Deep Reinforcement Learning Models

{Anderlini2019} - MATLAB suggestions
1) increase sample time, 2) reduce episode duration and 3) reduce size of mini-batch.
One additional thing to try is to parallelize training. You can use Parallel Computing Toolbox for that, and to set this up, you pretty much need to set a flag in training options (see e.g. here).
We are also working on adding more training algorithms for continuous action spaces that are more sample efficient, so I would check back when R2020a goes live.

{ParkerHolder2022AutoRL}:  ""In fact, existing optimization experiments have observed that dynamically adapting learning rates is also beneficial for model-free and model-based RL""

{Yang2023ReducingTime} - JUST QUOTE!"
	PARKER HOLDER: strategy = Multi-Objective Reinforcement Learning (MORL) ; minimizing the memory usage of an algorithm and/or the wall-clock time - IN ADDITION TO base obj (i.e. optimize max reward ), may be important considerations in choosing the algorithm
	
ParkerHolder2022AutoRL	Say in text \cite{ParkerHolder2022AutoRL} -- The most authoratiative survey \footnote{Researchers from Universities - Oxford, Leibniz, Freiburg and from Google, Meta and Amazon}. 
	Holder provides "TABLE.3. High level summary of each class of AutoRL algorithms"
	Talks abt domain dependence - See Intro!! SEE ALL PURPLE - PINK HIGHLIGHTS
	does not cover  predictive moantenance
------------------------------------

### R-2-4-16: Novelty

	ParkerHolder2022AutoRL:: "it is to be expected that the discovered settings are not transferable to other environments" - HENCE tested on untuned
	Eimer2023AutoRL -- This is especially important for as-of-yet unexplored domains, as pointed out by Zhang et al. (2021a).
	
	"ParkerHolder2022AutoRL: Looking at specific algorithms, Andrychowicz et al. (2021) conducted an extensive investigation into design choices for on-policy actor critic algorithms. They found significant differences in performance across loss functions, architectures and even initialization schemes, with significant dependence between these choices"
	
	ParkerHolder2022AutoRL: why empiricla study of different algos-- because "To the best of our knowledge, so far no AutoRL approach has explored the intersection of algorithm selection and algorithm configuration"
	
	"parker-holder - (1) RL is a complete closed-loop system. As such, it is likely each of the components discussed has an influence on others,  (2) The challenge is compounded in RL since evaluations in RL are almost always necessarily stochastic and much more noisy than in supervised learning, due to various sources (e.g. policy, environment), which can be a challenge for any form of automatic tuning."
	
------------------------------------
### R-1-5-7: Acrchitecture

	ParkerHolder2022AutoRL : very little attention has been paid to the design of neural architectures in RL
	ParkerHolder2022AutoRL: In general, there remains little conceptual understanding (and uptake) on architectural design choices and their benefit
	it is common to use two or three hidden layer feedforward MLPs
	for image related RL: there has been little research into alternatives, still using from original DQN paper 

------------------------------------
### Issues with HP

ParkerHolder2022AutoRL: 'Henderson et al. (2018) found that many of the most successful recent algorithms
were brittle with respect to hyperparameters'

ParkerHolder2022AutoRL: 'The non-stationarity of the AutoRL problem poses new challenges for classical (hyperparameter) optimization methods'

------------------------------------
### In conclusion:

 - mention ParkerHolder2022AutoRL	'Finally, beyond the well-flagged hyperparameters, there are even significant code level implementation details. Henderson et al. (2018) identified this and showed that different codebases for the same algorithm produced markedly different results'
 
'While RL as a field has seen many innovations in the last years, small changes to the algorithm or its implementation can have a big impact on its results (Henderson et al., 2018; Andrychowicz et al., 2021; Engstrom et al., 2020)'
	
------------------------------------	
### R2-7-19:

Future areas: 4 add: Multi-agent, Multi-obj and Offline RL and Meta-RL (learning to learn)

	"In addition to the discussion so far, the majority of the work in this survey addresses single agent RL. Multi- Objective Reinforcement Learning (MORL), (3) offline RL, where agents must generalize to an real world environment from a static dataset of experiences ie PHM datasets"

------------------------------------	

## IMPORTANT Ref: mania2018 and Nikolai_Matni_2019_REINFORCE_lecture

### In support of SIMPLER ALGOS!!!!
### Performance is Domain specific
### R-2-13 - Discuss: Computational complexity 

Ref: Nikolai_Matni_2019_REINFORCE_lecture

"Although the approach has a few drawbacks, the simplicity of implementation is often valuable enough to
justify its use. There are two primary applications of this sort of stochastic search approach in reinforcement
learning: policy gradient and pure random search."
direct policy search, derivative-free, can solve "unconstrained optimization problems through function evaluations."
If you can sample efficiently from p(z; theta), then you can run the algorithm on essentially any problem.
"using a derivative-free optimization method, and can not achieve the same perfor-
mance as methods that compute actual gradients. This performance gap is exacerbated when the function
evaluations are noisy. Another drawback to this approach is that our choice of probability distribution can
lead to high variance of stochastic gradients. High variance requires more samples to be drawn in order to
nd a minima or maxima. In other words, sample complexity increases."

Nikolai_Matni_2019: It is difficult to say which approach is better without selecting a specific problem to which to apply them
mania2018: >> Simple random search provides a competitive approach to reinforcement learning
Computationally, our random search algorithm is at least 15 times more ecient than the fastest competing model-free methods on these benchmarks.
