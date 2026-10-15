# M06 — Claim-Labeling Guide

## 1. Purpose

This guide defines how to label sentences in scientific papers for the claim-extraction component of our automated peer-review assistance project. The goal is to identify sentences that describe the paper's own contribution, specific research objective, or findings, rather than general background or results from prior work.

The initial input consists of abstracts and introduction excerpts from research papers about homelessness, including Housing First interventions, community-based support services, housing stability, mental health outcomes, and the effectiveness of interventions for people experiencing homelessness.

## 2. Label definitions

**CLAIM**

A sentence that describes the current paper’s own research objective, contribution, or finding. This includes a specific study aim, a new framework or dataset introduced by the paper, or a result reported from the paper’s own analysis.

**BACKGROUND**

A sentence that provides general context, describes previous research, explains the motivation for further research, or gives routine methodological details without stating the current paper’s own research objective, contribution, or finding.

## 3. Decision procedure

For every sentence, apply the following questions in order.

1. **Does the sentence state a specific objective of the current paper, a contribution it makes, or a finding or conclusion from its own analysis?**
   - If yes, label CLAIM.
   - A specific research aim counts as CLAIM even if the paper has not yet reported results.
   - A conclusion or implication counts as CLAIM when it expresses an interpretation or implication of the current paper's findings.

2. **Does the sentence primarily describe prior research, general context, a research gap, or a reason more research is needed?**
   - If yes, label BACKGROUND, unless the sentence also clearly states a current-paper objective, contribution, finding, or conclusion.

3. **Does the sentence describe a method, dataset, or research resource?**
   - If it explicitly presents a method, dataset, or resource as a contribution of the current paper, label CLAIM.
   - If it describes routine procedures or resources used in the study, label BACKGROUND.

4. **Does the sentence contain both background information and a current-paper claim?**
   - If it clearly expresses a current-paper objective, contribution, finding, or conclusion, label the whole sentence CLAIM.
   - Do not split a sentence into smaller units.

5. **Is the label still ambiguous?**
   - Check the surrounding sentences to determine whether the statement refers to the current paper or previous work.
   - Apply the definitions and difficult-case rules consistently.
   - Record unresolved cases for team discussion rather than silently guessing.

**Determine the label from the sentence's meaning in context, not from keywords, section headings, or first-person wording alone.** Words such as “we,” “found,” “aimed,” and “previous studies” can provide clues, but they do not determine the label by themselves.

## 4. Positive examples: CLAIM

These sentences are labeled CLAIM because they describe the current paper’s research objective, findings, or conclusions. The labels below follow the team’s existing consensus labels in the CSV.

### P1 — Research objective

**Source:** E1, Introduction, Sentence 5

> Therefore, this systematic review aimed to synthesise existing the current evidence on the effectiveness of community-based mental health, social, and health interventions for people experiencing homelessness and mental health conditions.

**Label:** CLAIM

**Reason:** Explicitly states the objective of the current systematic review. A specific research aim counts as CLAIM even though it does not report a result.

### P2 — Finding about interventions

**Source:** E2, Abstract: Results, Sentence 1

> Community-based interventions, particularly Housing First models, were frequently associated with improved housing stability, mental health outcomes, and programme retention.

**Label:** CLAIM

**Reason:** Reports a finding presented in the current paper’s Results section about outcomes associated with community-based interventions.

### P3 — Conclusion and implications

**Source:** E2, Abstract: Conclusion, Sentence 5

> The findings support the implementation of integrated community-based services combining housing, mental health, and social support.

**Label:** CLAIM

**Reason:** States an implication drawn from the current paper’s findings and connects the findings to a proposed approach to service delivery.

### P4 — Finding from the current analysis

**Source:** E4, Discussion, Sentence 5

> However, our results found reductions in problem substance use for both HF and TAU, with no clear difference between them.

**Label:** CLAIM

**Reason:** Explicitly reports the current paper’s results concerning substance use and the comparison between Housing First (HF) and treatment as usual (TAU).

### P5 — Findings about intervention outcomes

**Source:** E5, Discussion, Sentence 7

> Our results indicate that HF-ICM can result in improvements in housing stability, probability of hospitalization, community functioning and outcomes related to alcohol use in both racialized and non-racialized groups in the Canadian setting.

**Label:** CLAIM

**Reason:** Explicitly introduces the statement as the current study’s results and reports outcomes associated with Housing First with Intensive Case Management (HF-ICM).

## 5. Negative examples: BACKGROUND

These sentences are labeled BACKGROUND because they primarily describe prior research, research gaps, or the motivation for further investigation rather than the current paper’s own specific objective, contribution, or finding.

### N1 — Summary of previous research

**Source:** E1, Introduction, Sentence 2

> Although previous systematic reviews have examined specific intervention models, particularly Housing First programmes and selected psychosocial approaches, evidence remains fragmented across the broader range of community-based interventions available for people experiencing homelessness and mental health conditions [2,5,12].

**Label:** BACKGROUND

**Reason:** Describes the scope and limitations of previous systematic reviews. It establishes the research context rather than reporting a finding from the current paper.

### N2 — Gap in the existing literature

**Source:** E1, Introduction, Sentence 4

> Consequently, a comprehensive synthesis of contemporary evidence is needed to improve our understanding of effective intervention components, inform service development, and support evidence-based community care.

**Label:** BACKGROUND

**Reason:** Explains why additional research is needed. It motivates the research but does not explicitly state the current paper’s particular research objective or a result.

### N3 — Findings attributed to previous reviews

**Source:** E4, Discussion, Sentence 4

> Previous reviews have questioned whether abstinence-contingent housing may lead to greater reductions in problem substance use than HF, although at the cost of housing stability.

**Label:** BACKGROUND

**Reason:** Attributes the observation to previous reviews rather than presenting it as a finding of the current paper.

### N4 — Gap in previous research

**Source:** E5, Discussion, Sentence 4

> The effectiveness of the HF approach, in particular HF with ICM, has not previously been rigorously assessed in an ethnically diverse population composed of a large number of immigrants.

**Label:** BACKGROUND

**Reason:** Describes a gap in the existing literature. It explains the context for research rather than reporting a result from the current study.

### N5 — Information attributed to an external source

**Source:** E5, Discussion, Sentence 6

> A recent article has noted the lack of literature on the unique housing needs of individuals of Aboriginal or non-European ethnicity, particularly in Canada.

**Label:** BACKGROUND

**Reason:** Attributes the observation to a recent external article. The sentence discusses the existing literature rather than stating the current paper’s own finding.

## 6. Rules for difficult cases

### A. Research aims versus research motivation

A sentence stating what the current paper specifically aims to investigate counts as CLAIM. A sentence that only states that a problem deserves further study or that more evidence is needed counts as BACKGROUND.

### B. Prior work versus current work

A sentence that only summarizes another paper's method, results, or conclusions counts as BACKGROUND. A sentence that reports the current paper's own analysis or findings in comparison with prior work counts as CLAIM when the current paper's contribution or finding is clear.

### C. Methods versus methodological contributions

Routine descriptions of data collection, database searches, preprocessing, or experimental setup count as BACKGROUND. A sentence that presents a new method, dataset, or research resource as a contribution of the current paper counts as CLAIM.

### D. Mixed sentences

If a sentence contains background information and clearly expresses a current-paper objective, contribution, finding, or conclusion, label the whole sentence CLAIM. Do not split sentences into smaller units.

### E. Hypotheses and expected outcomes

A hypothesis or predicted outcome is not automatically a finding. Label it CLAIM only if it also clearly states the current paper's specific research objective or contribution under the definitions above. A prediction alone, without a clear current-paper objective, contribution, finding, or conclusion, counts as BACKGROUND. Record borderline examples for team discussion.

### F. Abstracts and introductions

Apply the same definitions across all paper sections. Do not automatically label a sentence as CLAIM because it appears in an abstract or near the end of an introduction. Use the sentence's meaning and its context.

### G. Claims are not novelty judgments

This task identifies sentences describing the current paper's own objectives, contributions, findings, or conclusions. It does not determine whether a contribution is genuinely novel, correct, fair, or important. Those judgments belong to other parts of the project.