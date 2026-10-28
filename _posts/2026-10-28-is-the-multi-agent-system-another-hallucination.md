---
layout: post
title: "Is the Multi-Agent System Another Hallucination?"
date: 2026-10-28
categories: [AI Agents, Multi-Agent Systems, LLM]
tags: [LLM Agents, Multi-Agent Systems, Failure Diagnosis, Root Cause Analysis, Agent Security]
---

*A Review of Failure Diagnosis in LLM Agents and Multi-Agent Systems*

## Background and Scope

The concept of an intelligent agent predates large language models
(LLMs). Wooldridge and Jennings (1995) describe agents through
properties such as autonomy, social ability, reactivity, and
pro-activeness, while later multi-agent systems (MAS) research treats
interacting agents as autonomous computational entities that can
coordinate toward individual or collective goals (Wooldridge, 2009).
From a software-engineering perspective, Zambonelli et al. (2003)
further characterize a MAS as a computational organization whose roles,
relationships, and interaction structures can be explicitly designed.

LLM-based agents extend these earlier ideas with natural-language
reasoning, tool use, memory, and orchestration. This extension also
broadens the failure surface: a problem may originate in code or
configuration before deployment, emerge at a particular runtime step,
propagate across a long trajectory, or arise from coordination and
verification problems among multiple agents. The literature reviewed
below therefore forms a progression from pre-deployment security
analysis to trajectory-level root-cause diagnosis and system-level MAS
failure analysis.

## Pre-Deployment Security: Agent Audit

Agent Audit focuses on security failures in the software stack
surrounding an LLM agent rather than on model behavior alone. It
analyzes Python agent code and deployment artifacts for risks such as
untrusted data reaching dangerous operations, exposed credentials,
unsafe prompt construction, and over-privileged Model Context Protocol
(MCP) configurations (Zhang et al., 2026). Its agent-aware analysis
combines tool-boundary detection, intra-procedural taint tracking,
credential detection, structured configuration parsing, and privilege
checks. The accompanying Agent-Vuln-Bench contains 42 annotated
vulnerabilities across 22 samples; Agent Audit reports 40 of them, with
six false positives (Zhang et al., 2026).

The method is useful as a pre-deployment guardrail, but its scope also
defines an important boundary. The current implementation does not track
inter-procedural data flow, primarily targets Python, and does not
execute or simulate runtime prompt-injection payloads (Zhang et al.,
2026). In other words, static analysis can identify a risky tool
boundary or sink, but it cannot by itself establish whether a particular
runtime interaction actually caused a failed trajectory. This limitation
motivates execution-trace approaches that reason over what happened
during an agent run.

## Evidence-Grounded Trajectory Diagnosis: AgentRx

Barke et al. (2026) address runtime failure localization in settings
where executions are probabilistic, long-horizon, multi-agent, and
affected by noisy tool outputs. Their benchmark contains 170 failed
trajectories across 11 task settings, each annotated with a critical
failure step and a root-cause category. AgentRx normalizes heterogeneous
logs into a common trajectory representation, synthesizes verifiable
constraints from tool schemas, domain policies, task requirements, and
the observed trajectory, and checks those constraints step by step. The
resulting validation log records violated obligations and supporting
evidence, which an LLM-based judge then uses to identify the critical
failure step and category (Barke et al., 2026).

**Figure 1**

*Constraint-based failure localization in AgentRx*

![Figure 1]({{ "/assets/images/mas-failure-review/figure-1.png" | relative_url }})

*Note.* Reproduced from “AgentRx: Diagnosing AI Agent Failures from
Execution Trajectories,” by S. Barke, A. Goyal, A. Khare, A. Singh, S.
Nath, and C. Bansal, 2026, Findings of the Association for Computational
Linguistics: EMNLP 2026, p. 12610
(https://aclanthology.org/2026.findings-emnlp.598/). Copyright 2026 by
the Association for Computational Linguistics.

A key design choice is the definition of a critical failure as the first
unrecoverable failure introduced by any agent in the trajectory (Barke
et al., 2026). This creates a clear and economical annotation target and
makes the final diagnosis auditable because the selected step is
supported by an explicit violation record. The benchmark annotations
also illustrate how a root cause is separated from other observed
failures.

**Figure 2**

*Example of an AgentRx root-cause annotation for an
instruction-adherence failure*

![Figure 2]({{ "/assets/images/mas-failure-review/figure-2.png" | relative_url }})

*Note.* Reproduced from Appendix E.3 of “AgentRx: Diagnosing AI Agent
Failures from Execution Trajectories,” by S. Barke et al., 2026,
Findings of the Association for Computational Linguistics: EMNLP 2026,
p. 12627 (https://aclanthology.org/2026.findings-emnlp.598/). Copyright
2026 by the Association for Computational Linguistics.

The single-critical-step formulation, however, leaves a broader
attribution question open. Earlier actions may be recoverable in
isolation yet still increase the likelihood or severity of a later
failure, and similar problematic actions may recur across several
agents. A richer model could therefore distinguish the decisive failure
from contributory steps and assign different degrees of responsibility
across the trajectory. This is especially relevant when the same
behavior appears across agents or when similar tasks fail through
different pathways.

## Cascading Errors and Recovery: AgentErrorTaxonomy, AgentErrorBench, and AgentDebug

Zhu et al. (2025) shift the focus from a single observed failure to
error propagation. Their central premise is that an early mistake can
alter later reasoning and actions, producing a cascade of downstream
errors that are symptoms rather than independent root causes. They
introduce AgentErrorTaxonomy to classify failures across memory,
reflection, planning, action, and system-level operations, and
AgentErrorBench to evaluate whether a model can identify critical errors
and produce actionable corrective feedback from real agent rollouts (Zhu
et al., 2025).

**Figure 3**

*Construction pipeline for AgentErrorTaxonomy and AgentErrorBench*

![Figure 3]({{ "/assets/images/mas-failure-review/figure-3.png" | relative_url }})

*Note.* Reproduced from “Where LLM Agents Fail and How They Can Learn
From Failures,” by K. Zhu et al., 2025, Figure 2, arXiv
(https://doi.org/10.48550/arXiv.2509.25370).

AgentDebug operationalizes this idea through modular, step-level
analysis. For each decision step, the framework examines memory,
reflection, planning, and action behavior and maps detected problems to
the taxonomy. It then considers the complete trajectory, the modular
error profile, the taxonomy, and a critical-error criterion to identify
a minimal set of critical errors and generate corrective feedback. The
earliest critical root-cause step is used as the restart point. Instead
of rerunning the task from the beginning, the agent is re-executed from
that point with targeted feedback; if the new rollout still fails, the
feedback is refined and the process repeats (Zhu et al., 2025).

This approach directly complements AgentRx. AgentRx emphasizes auditable
evidence for the first unrecoverable failure, whereas AgentDebug
explicitly reasons about dependencies between early root causes and
downstream symptoms and connects diagnosis to recovery. Even so,
selecting an earliest restart point still compresses a potentially
distributed causal process into a small set of root causes. In
multi-agent settings, the remaining question is not only which step was
earliest, but how responsibility should be distributed when several
agents, states, or repeated actions jointly contribute to failure.

## System-Level Multi-Agent Failure Patterns: MAST

Cemri et al. (2025) broaden the unit of analysis from individual
trajectory steps to recurrent failure patterns across multi-agent
systems. They define a failure as a case in which the MAS does not
achieve its intended task objective and introduce MAST-Data, comprising
1,642 annotated traces from seven MAS frameworks. The Multi-Agent System
Failure Taxonomy (MAST) identifies 14 failure modes grouped into three
categories: system design issues, inter-agent misalignment, and task
verification (Cemri et al., 2025).

MAST is derived using Grounded Theory rather than imposed a priori. The
authors use open coding to identify observed failure behavior, constant
comparison to refine categories across systems, memoing to preserve
analytical insights, and iterative theorizing until additional analysis
yields no substantively new failure modes (Cemri et al., 2025; Glaser &
Strauss, 1967). Expert annotation achieved high agreement, and an
LLM-as-a-Judge pipeline was subsequently calibrated for scalable
labeling. This use of LLM-based evaluation is practical, although
judge-based pipelines should still be interpreted cautiously because LLM
evaluators can exhibit position, verbosity, self-enhancement, and
reasoning biases (Zheng et al., 2023).

**Figure 4**

*MAST taxonomy of multi-agent system failure modes across conversation
stages*

![Figure 4]({{ "/assets/images/mas-failure-review/figure-4.png" | relative_url }})

*Note.* Reproduced from “Why Do Multi-Agent LLM Systems Fail?” by M.
Cemri et al., 2025, Figure 1, Advances in Neural Information Processing
Systems, 38 (https://doi.org/10.52202/085713-4082).

MAST therefore complements trajectory-localization work by exposing
structural and interaction-level failure modes that may span
pre-execution, execution, and post-execution stages. However, taxonomy
membership does not by itself establish which agent or step causally
contributed most to a failure. MAST is strongest at describing recurring
system-level patterns, while AgentRx and AgentDebug are stronger at
localizing failure within a particular execution. The unresolved problem
lies in connecting these levels of analysis.

**Table 1**

*Connections Across the Reviewed Failure-Analysis Approaches*

| Study | Primary unit | Stage | Core mechanism | Main output | Key limitation |
|---|---|---|---|---|---|
| Agent Audit (Zhang et al., 2026) | Code/configuration | Pre-deployment | Static, taint, and configuration analysis | Security findings and risky boundaries | Cannot establish runtime causal paths |
| AgentRx (Barke et al., 2026) | Trajectory step | Post-failure | Constraint synthesis, validation evidence, LLM judge | First unrecoverable step and category | Focuses attribution on one decisive step |
| AgentDebug (Zhu et al., 2025) | Step/module | Diagnosis + recovery | Modular error analysis, root-cause detection, reroll | Critical errors, feedback, recovery | Distributed responsibility remains compressed |
| MAST (Cemri et al., 2025) | MAS-level pattern | Across stages | Grounded-theory taxonomy + LLM annotation | 14 design, alignment, and verification modes | Labels do not directly prove causal contribution |

*Note.* The table summarizes the conceptual connection across the reviewed studies; the “key limitation” column is a synthesis of their scopes rather than a claim made verbatim by each paper.

## Cross-Study Inferences

- Failure evaluation is moving from endpoint success/failure toward
process-level diagnosis based on trajectories, intermediate actions, and
observable evidence.

- The reviewed methods operate at different
granularities—code/configuration, step, functional module, and
system-level interaction—yet no shared representation currently unifies
these levels.

- Using the first or earliest critical failure is efficient for
localization and reroll, but it may underrepresent cumulative or
distributed responsibility when multiple earlier steps contribute to the
final outcome.

- Evidence of a rule violation and evidence of causal responsibility
are related but not identical: a violated constraint can show what was
wrong without proving how much that step contributed to the terminal
failure.

- Cross-agent and cross-trajectory recurrence is underused. Repeated
failure signatures across agents or similar tasks could help distinguish
one-off mistakes from systematic design weaknesses.

- A robust debugging pipeline likely needs both prevention and
runtime diagnosis: static security analysis can identify risky
boundaries, while trajectory analysis is required to verify
manifestation, propagation, and recovery.

## Remaining Gap and Research Directions

Taken together, the literature has made substantial progress in defining
failure modes, localizing critical steps, grounding diagnoses in
explicit evidence, and using root-cause feedback for recovery. The main
unresolved gap is a unified causal attribution framework that can
connect pre-deployment risk, runtime evidence, multi-step error
propagation, and multi-agent interaction while preserving auditable
explanations. In particular, existing methods do not yet provide a
general mechanism for assigning graded responsibility across several
contributory steps or agents, nor do they systematically compare related
trajectories to determine whether the same underlying behavior produces
different failure paths.

A promising direction is responsibility-weighted causal attribution:
instead of returning only one critical step, a debugger could estimate
the relative contribution of several steps and agents, then validate
those estimates through controlled reruns or counterfactual
interventions. Recent work is already moving toward this direction.
TrajDebug tracks the lifecycle, resolution status, and terminal impact
of multiple local errors rather than treating every detected error as
equally causal (Qi et al., 2026). Pothuru (2026) goes further by using
injected faults and downstream re-execution to evaluate causal failure
localization against a known intervention site. These approaches suggest
that future MAS debugging could combine constraint-based evidence,
error-lifecycle tracking, cross-agent causal graphs, and
cross-trajectory comparison. Such a framework would move failure
analysis from “where did the system first become unrecoverable?” toward
the broader question of “which actions and agents contributed to
failure, by how much, and which intervention would most reliably prevent
recurrence?”

## References

Barke, S., Goyal, A., Khare, A., Singh, A., Nath, S., & Bansal, C.
(2026). AgentRx: Diagnosing AI agent failures from execution
trajectories. *Findings of the Association for Computational
Linguistics: EMNLP 2026*, 12610–12639.
https://aclanthology.org/2026.findings-emnlp.598/

Cemri, M., Pan, M. Z., Yang, S., Agrawal, L. A., Chopra, B., Tiwari, R.,
Keutzer, K., Parameswaran, A., Klein, D., Ramchandran, K., Zaharia, M.,
Gonzalez, J. E., & Stoica, I. (2025). Why do multi-agent LLM systems
fail? *Advances in Neural Information Processing Systems, 38*.
https://doi.org/10.52202/085713-4082

Glaser, B. G., & Strauss, A. L. (1967). *The discovery of grounded
theory: Strategies for qualitative research*. Aldine Publishing Company.

Pothuru, L. (2026). When failures propagate: Causal failure attribution
in agentic retrieval-augmented generation. In *Proceedings of the Second
Workshop for REsearch on Agent Language Models (REALM 2026)* (pp.
675–686). Association for Computational Linguistics.
https://aclanthology.org/2026.realm-1.46/

Qi, Y., Yin, Z., Shi, X., Peng, H., Lu, S., Liu, Y., Xuan, R., Yuhong,
L., Hu, Z., Wang, X., Hou, L., Xu, B., & Li, J. (2026). TRAJDEBUG:
Tracing error lifecycle to identify critical failures in long-horizon
agent trajectories. In *Findings of the Association for Computational
Linguistics: EMNLP 2026* (pp. 9169–9192). Association for Computational
Linguistics. https://aclanthology.org/2026.findings-emnlp.437/

Wooldridge, M. (2009). *An introduction to multiagent systems* (2nd
ed.). Wiley.

Wooldridge, M., & Jennings, N. R. (1995). Intelligent agents: Theory and
practice. *The Knowledge Engineering Review, 10*(2), 115–152.
https://doi.org/10.1017/S0269888900008122

Zambonelli, F., Jennings, N. R., & Wooldridge, M. J. (2003). Developing
multiagent systems: The Gaia methodology. *ACM Transactions on Software
Engineering and Methodology, 12*(3), 317–370.
https://doi.org/10.1145/958961.958963

Zhang, H., Nian, Y., & Zhao, Y. (2026). *Agent Audit: A security
analysis system for LLM agent applications* [Preprint]. arXiv.
https://doi.org/10.48550/arXiv.2603.22853

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y.,
Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., &
Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot
Arena. *Advances in Neural Information Processing Systems, 36*,
46595–46623. https://doi.org/10.52202/075280-2020

Zhu, K., Liu, Z., Li, B., Tian, M., Yang, Y., Zhang, J., Han, P., Xie,
Q., Cui, F., Zhang, W., Ma, X., Yu, X., Ramesh, G., Wu, J., Liu, Z., Lu,
P., Zou, J., & You, J. (2025). *Where LLM agents fail and how they can
learn from failures* [Preprint]. arXiv.
https://doi.org/10.48550/arXiv.2509.25370
