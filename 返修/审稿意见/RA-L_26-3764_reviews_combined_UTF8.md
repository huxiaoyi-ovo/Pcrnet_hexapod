Three expert reviewers evaluated the manuscript and provided detailed
comments and suggestions that should be carefully addressed.

R1: The reviewer finds the paper interesting and appreciates the
extensive ablations and comparisons. The main concerns are the fairness
of the monolithic PPO comparison, possible overfitting due to training
and evaluation in the same environment, limited motivation for the
hexapod embodiment, and insufficient positioning relative to prior
work. The reviewer also requests clarification of several mathematical
and methodological aspects.

R2: The reviewer appreciates the clear structure and thorough
experiments but raises several technical and reproducibility concerns.
These mainly involve the affordance map and robot footprint, the
per-command risk formulation, the lateral-only Avoid expert, training
details, train/test separation, and the limitations of the safety
argument.

R10: The reviewer is positive about the proposed method and its
experimental validation. The main concerns are the restriction to
lateral avoidance, applicability beyond the structured environments
considered in the paper, and missing network architecture details
needed for reproducibility.

In addition to the reviewers’ comments, the AE asks the authors to more
clearly highlight the main contributions and novelty of the work,
particularly with respect to the existing literature and closely
related approaches.


Summary

This is an interesting piece of work, and the authors have evidently
invested considerable effort in the ablation studies and in providing
valuable comparisons to other methods. In particular, the ablations
concerning the gate mechanism offer a compelling motivation for the
proposed approach.

Strengths

- The paper presents a thorough set of ablations, in particular those
concerning the gate mechanism, which nicely motivate the proposed
method.
- The comparisons to other methods are valuable and reflect significant
effort on the part of the authors.
- Overall, the figures are well done. Figure 1 is particularly
effective at grabbing the reader's attention and conveying the goal of
the work. 
- The citations are adequate. 

Weaknesses

Major concerns
- The ablation comparing the proposed method to a monolithic PPO
baseline is unconvincing. This comparison should arguably be the
central one in the paper, and the failure to present it convincingly is
a significant shortcoming. The authors state that the PPO network was
trained with "the same training budget as PCR-Net," without specifying
what this means in practice. If the intention is to argue that the
proposed method is considerably cheaper to train, this is a reasonable
claim, but it should be substantiated by training the PPO baseline to
its fullest potential and then presenting training cost as a separate
performance metric alongside the others. As currently presented, the
comparison does not achieve this.
- The simulation-based ablation is confounded by the fact that the
network was trained on the same environment used for evaluation. This
gives learned methods an inherent advantage, as they benefit from
fine-tuning on the specific evaluation environment. This seems to be
particularly visible in the accompanying video, where the learned
policy seems overfit to the distance between rows of obstacles. I
recommend repeating the evaluation on environments not seen during
training, ideally on multiple such environments, as is standard
practice in the field.
- The hexapod embodiment, while featured in the title, is not well
motivated in the paper. An ablation (in simulation) examining the
impact of embodiment on system performance would strengthen the paper
considerably—for instance, would such a complex architecture be
necessary for a simpler embodiment?
- The related work section does not clearly establish the paper's
contribution relative to prior work. In many instances, the contrast
with existing approaches is presented, but the motivation for why the
proposed approach should be superior is not adequately articulated.
- The overall system appears overengineered, and even after reviewing
the ablations, I remain unconvinced that this level of complexity is
necessary.

Minor concerns
- The writing is at times overly elaborate, which detracts from clarity
and precision. Examples include the phrases "PCR-Net uses
command-direction-conditioned clearance," "matching their bounded
semantics while allowing state-dependent concentration," and "command
risk is used as a clearance prior rather than a reach-avoid value."
- The choice of the name "PCR-Net" is questionable given the prior,
influential work of Sarode et al. [1]; the authors should either change
the name or provide explicit justification for the overlap.
Furthermore, the usage of the name is inconsistent throughout the
paper—at times referring to the learned gate, and at other times to the
entire system. - The latter usage is particularly confusing, as the
"-Net" suffix conventionally implies a monolithic network. In the
Experiments section, the method is referred to inconsistently as both
"PCR-Net" and "Learned-w," which adds further confusion.
- The sentence "PCR-Net addresses this conflict directly in the robot's
command space…" does not clearly specify which conflict is being
referenced. Relatedly, the statement "Standard local navigation
formulations that treat the target only as a moving waypoint do not
explicitly preserve target tracking during local avoidance, and
fixed-priority switching between a follow and an avoid behavior can
lose the target during the maneuver" describes a limitation that does
not appear to be guaranteed to be resolved by the proposed solution
either.
- The term "affordance map" used to describe the occupancy map is
confusing and may benefit from reconsideration.
- The statement "and a single command generally cannot maximize both
target recovery and immediate clearance" appears mathematically
inconsistent if the two objectives are defined as orthogonal vectors.
If they occupy decoupled subspaces, both could in principle be
optimized independently, and therefore jointly maximized, simply
through addition.
- The mathematical exposition is at times difficult to follow. Some
variables are introduced but never subsequently used (e.g., m_t), while
others are used without being introduced at all. In addition, the
parameter y is used to denote both the common coordinate y and a scalar
gating value, which is a source of confusion.
- Several aspects of the method are described only briefly, leaving a
number of important questions unanswered, including: Why are the gate
parameters chosen in the manner described? Is there a benefit to
supplying both the risk value and its memory? How exactly is the memory
supplied to the gate? Why are slew-rate limiting and clearance-based
scaling necessary?
- The contrast drawn with the work of Scheidemann et al. [2] is
unclear. That work also addresses command-space arbitration, albeit in
a less explicit manner, and this distinction should be clarified.
References [3] and [4] are cited together, but only one of the two is
compared against in the paper. The authors should clarify why both are
not compared, or justify the omission.
- The statement "Published leader-following and agile-navigation stacks
often couple different sensors, planners, and low-level controllers"
would benefit from a supporting citation, as it is currently unclear
what specific works are being referenced.
- The bottom row of images in Figure 6 is too small, and the
accompanying text is illegible.

Questions for the authors
- Why is a 3-DoF avoidance policy trained only for two of the output
dimensions to subsequently be dropped? Would it not be more direct to
train a 1-dimensional output from the outset?
- In the PPO training procedure, is the map provided as an unmodified
input? If so, this would deviate from standard practice for
state-of-the-art PPO networks applied to this type of task, and could
substantially degrade training performance.


[1] Sarode, Vinit, et al. "Pcrnet: Point cloud registration network
using pointnet encoding." arXiv preprint arXiv:1908.07906 (2019).
[2] C. Scheidemann et al., "Obstacle-avoidant leader following with a
quadruped robot," in 2025 IEEE International Conference on Robotics and
Automation (ICRA), 2025, pp. 1407–1413.
[3] T. He, C. Zhang, W. Xiao, G. He, C. Liu, and G. Shi, “Agile but
safe:
Learning collision-free high-speed legged locomotion,” in Robotics:
Science and Systems (RSS), 2024.
[4] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M.
Hutter,
“Learning robust perceptive locomotion for quadrupedal robots

The paper addresses target following in cluttered environments and
reframes it as a command-space arbitration problem. Instead of
trajectory-level control or global navigation, the authors arbitrate
directly on the single velocity command. Two experts are used, one for
following (analytic) and one for avoidance (learned), commanding in
orthogonal subspaces (forward-yaw vs. lateral). The two proposals are
blended by a risk-conditioned learned gate on top of a fixed low-level
locomotion policy actuating a hexapod test platform. The method is
validated in simulation against arbitration ablations, an
additive-fusion variant, a monolithic PPO policy and a DWA-style
velocity search, and in 60 real-robot trials with onboard perception.
The main claim is that learned, risk-conditioned soft arbitration gives
the best safety-following trade-off, in particular at above-training
target speed.

## Organization and style

- Organization is good: the hierarchy (perception -> experts -> gate ->
locomotion) is clear and easy to follow, and the contributions are
stated up front.
- Writing is generally precise and readable, the problem statement in
Sec. I/III is convincing.
- Introduction makes heavy use of em-dashes, sometimes it is not clear
why. Please revise, fix some overly complicated sentences
- Generally, level of detail is not very consistent, consider
re-structuring starting with the very fundamental ideas to the
implementation details.
- Avoid inconsistent capitalization.

## Presentation

- Presentation is overall fine, tables and plots are readable and the
trajectory figure (Fig. 4b) is a good addition.
- The supplementary video helps in understanding the setup. Please
consider adding an indicator of how much of it is sped up.
- Fig. 2 is low-res, consider vector graphics.
- Sec. IV: I do not find PCR, Gate, conflict-aware fusion etc. in Fig.
2. Consider unifying the notation between text and figure so it is
easier to see which block is being discussed.

## Strengths

- Simple analytic Follow expert: it highlights that the gain comes from
the arbitration method rather than from already high-performing
experts.
- Clear setup, good hierarchy of the method.
- Learning only the gating: it keeps the underlying controllers in
charge and keeps the behaviour explainable., highlight explainability /
deterministic more
- Thorough experiments with good ablations and relevant baselines
(arbitration ablations, additive fusion, monolithic PPO, DWA-style
search).
- Bounded-residual / convex-hull argument for the fused command is a
nice property to have, even tough not clear to me how this helps in
terms of safety other then preventing anyway capped action magnitude

## Technical remarks

### Perception and affordance map
- The affordance map is unclear: what coverage does it have around the
robot, does it see sideways, and is it a built map or instantaneous
from depth? Please state this explicitly. if its a robo-centric map,
state which method was used
- Related: why is side-stepping possible at all if there is no
affordance information to the side? How is perception handled there?
- Is YOLO used only for the person/target, or does it also contribute
semantic obstacle detection? The masking of target-associated depth
points is mentioned, but a single explicit sentence on the division of
labour between detection and stereo depth would help.
- No footprint / body-size awareness is visible in the risk query.
Given the leg-inclusive envelope of ~0.65 m, how is the robot extent
accounted for? - in particular during rotate-on-spot command
- Concave obstacles: what happens when the avoidance direction points
backwards or into the concavity?
- Dynamic obstacles are not treated at all (all obstacles are static,
only the target moves). At least discuss this as a limitation, ideally
show one experimet / clearly mark as limitation

### Sec. IV-C (per-command risk) - please overhaul and motivate this
subsection better
- Map cell size is given as 0.09375 m; this sub-millimetre precision is
not meaningful, consider rounding or stating map extent and resolution
instead.
- Could you elaborate on what exactly the "clearance proxy" channel is
and how it is computed?
- Why only a 25 deg half-angle cone? If the robot is already close to
an obstacle, does it not need to look further sideways for the full
robot extent to be covered by the swept motion?
- The numeric values for free/safe (d_safe = 0.27 m, d_free = 0.57 m)
appear without justification. Is this also where the clearance margin
used in the results comes from? There is no sensitivity analysis on
these values. Please consider whether they belong in the method at all,
or whether they are hyperparameters to be tuned per robot/environment.
- The speed-scaled risk variant is presented and then discarded inside
the method section (ablation-style). It is not clear which variant is
used for the reported results.

### Command decomposition and avoid expert
- There is no good motivation why the Avoid expert may only command
lateral motion. Is backward motion not necessary for some obstacle
configurations (dead ends, concave geometry)?
- It is not clear how the Avoid expert is trained. Please state the
reward function, observations and termination conditions.
- When the Avoid expert is trained, is the Follow expert already in the
loop? What kind of tuning happens in that stage?
- which simulator setup is used?

### Sec. IV-F (training) and reproducibility
- The simulator and learning stack are mentioned only for the gating
(Isaac Gym, RSL-RL, PPO) please state clearly which simulator/asset
setup is used for each of the three training stages.
- A table listing, for each learned module (Avoid expert and Gate), the
exact observations and actions would help a lot. Does the gate observe
only the risks, or more? What exactly is "robot state" in the gate
observation? Stating this clearly would considerably improve
reproducibility.

### Sec. IV-G (safety argument)
- The convex-hull / bounded-residual guarantee is nice, but: just
because each individual expert command is reasonable, does that imply
the mixture is safe? This feels like a strong assumption, i.e. that
nudging the command slightly is enough to make it safe-ish. Please
discuss the limits of this argument, in particular in the high-conflict
cases where the gate is most active.

### Sec. V-A and results
- "Fixed hardest stage" of the curriculum: are these environments that
were seen during training, or held-out layouts? Please make the
train/test separation explicit.
- Please discuss failure modes more / when and why does it fail, which
limitation triggers failiure, does it recover
- Only the gate is velocity-dependent, correct? It is unclear which
component is expected to break when exceeding the training speed, since
the gating behaviour probably has no strong velocity gradient during
training. Because this above-training speed carries much of the
evaluation argument, please elaborate.
- Monolithic PPO is only evaluated at 0.35 m/s over increasing
difficulty, not across target speeds. Evaluating it at 0.50/0.60 m/s
like the other methods would make the comparison more complete and
would be an interesting addition. - nice to have, not crit

## Literature / citations

- The literature review covers only geometric and control-oriented
work. Human following and human-robot interaction literature is not
mentioned, although the target here is a walking person.
- Some literature backing for the design choice of reward shaping vs.
constrained RL would be nice.
- The mixture-of-experts framing is grounded almost exclusively in very
old work (Jacobs et al. 1991). Please add more recent work on learned
gating / mixture-of-experts composition. where has this been used, why
is it the go-to technology


The manuscript addresses the problem of passability mismatch during
target following in clutter, where a human leader can traverse narrow
gaps that exceed the safe clearance envelope of a legged robot. The
authors formulate this challenge as continuous command-space
arbitration rather than discrete behavior switching or local path
re-planning. The proposed architecture, PCR-Net, decomposes velocity
proposals into orthogonal Follow and Avoid subspaces, blending them via
a learned gating policy conditioned on local affordance maps and
per-command directional risk queries. The approach is evaluated in
Isaac Gym simulation against multiple ablation variants, an additive
alternative, monolithic RL, and DWA-style velocity search across target
speeds up to an above-training stress speed. The proposed method is
deployed on an 18-joint hexapod in straight and sharp-turn cluttered
corridors using onboard YOLOv8 target detection and stereo depth
mapping.  

Required Clarifications
--Longitudinal Avoidance & Continuous Obstacles: Constraining the Avoid
expert strictly to lateral motion presumes all collisions can be
averted by lateral sidestepping. Please discuss how the framework
behaves when encountering continuous transverse obstacles (such as
solid walls, dead-ends, or when the target stops abruptly in front of a
barrier) where lateral motion alone cannot resolve the collision.
--Environment Scope Discussion: The simulation benchmarks and hardware
tests rely primarily on periodic, staggered-row geometries. While
staggered rows isolate lateral passability mismatch, please explicitly
discuss the operational boundaries of this formulation when navigating
unstructured, randomly scattered clutter or non-convex obstacles.  
--Network Architecture & Reproducibility: Section IV-F mentions that
encoders feed a 256–256 ELU trunk with Beta heads for y and w. Please
provide the exact architectural details of the local affordance map
encoder (e.g., 2D CNN kernel sizes, strides, channel depths, or MLP
projection dimensions) and specify how spatial map features are
concatenated with the 31 scalar state inputs before entering the ELU
trunk.	