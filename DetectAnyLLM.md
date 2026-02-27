# DetectAnyLLM: Towards Generalizable and Robust Detection of Machine-Generated Text Across Domains and Models

**Jiachen Fu**
VCIP, CS, Nankai University
Tianjin, China
fujiachen2005@gmail.com

**Chun-Le Guo**
VCIP, CS, Nankai University
Tianjin, China
NKIARIShenzhen Futian, China
guochunle@nankai.edu.cn

**Chongyi Li***
VCIP, CS, Nankai University
Tianjin, China
NKIARIShenzhen Futian, China
lichongyi@nankai.edu.cn
*Corresponding Author. Project Lead.

---

## Abstract

The rapid advancement of large language models (LLMs) has drawn urgent attention to the task of machine-generated text detection (MGTD). However, existing approaches struggle in complex real-world scenarios: zero-shot detectors rely heavily on scoring model's output distribution while training-based detectors are often constrained by overfitting to the training data, limiting generalization. We found that the performance bottleneck of training-based detectors stems from the misalignment between training objective and task needs. To address this, we propose Direct Discrepancy Learning (DDL), a novel optimization strategy that directly optimizes the detector with task-oriented knowledge. DDL enables the detector to better capture the core semantics of the detection task, thereby enhancing both robustness and generalization.

Built upon this, we introduce DetectAnyLLM, a unified detection framework that achieves state-of-the-art MGTD performance across diverse LLMs. To ensure a reliable evaluation, we construct MIRAGE, the most diverse multi-task MGTD benchmark. MIRAGE samples human-written texts from 10 corpora across 5 text-domains, which are then re-generated or revised using 17 cutting-edge LLMs, covering a wide spectrum of proprietary models and textual styles. Extensive experiments on MIRAGE reveal the limitations of existing methods in complex environment. In contrast, DetectAnyLLM consistently outperforms them, achieving over a 70% performance improvement under the same training data and base scoring model, underscoring the effectiveness of our DDL.

**Project page:** [https://fjc2005.github.io/detectanyllm](https://fjc2005.github.io/detectanyllm)

## CCS Concepts

• **Computing methodologies** → Artificial intelligence; Natural language processing; Security and privacy;

## Keywords

Machine-Generated Text Detection, AI-Text Detection, AI Safety

---

## 1 Introduction

Advanced Large Language Models (LLMs) [2, 18, 24, 26, 31, 48] can easily generate text nearly indistinguishable from human writing [10, 45]. If misused, it could pose serious risks to society [52]. In response to such concern, the task of Machine-Generated Text Detection (MGTD) has emerged [1, 14, 25, 27, 29, 44]. MGTD is a binary classification task designed to distinguish whether a given text is written by humans or generated (or revised) by a machine.

In this study, we consider detection of both Machine-Generated Text (MGT) and Machine-Revised Text (MRT), where MRT refers to text that polished or rewritten by a model based on Human-Written Text (HWT). Our study focuses on black-box detection, which better reflects real-world application than white-box settings.

Several MGTD methods have been proposed [7, 14, 19, 25, 34, 44, 49, 59] based on the assumption that the token probability distributions are distinct between MGT and HWT. Most of them leverage pre-trained language models, referred to as scoring models, to estimate the token probabilities of a given text and compute classification metrics for distinguishing.

Existing MGTD methods can be categorized into zero-shot methods [5, 7, 14, 34, 46] and training-based methods [9, 25].

Zero-shot methods typically rely on the inherent capabilities of scoring models [6]. However, these models are often relatively small, with limited knowledge and simple output patterns. Thus, when detecting texts deviate from their inherent distribution, such methods often struggle to achieve reliable performance.

Training-based methods use supervised fine-tuned (SFT) [25, 36] or preference learning [9, 22, 40] to align the scoring model's output distribution with that of models who built the training data.

While such approach improves detection performance for that specific models, it seems difficult to generalize this detection knowledge to models outside the training data [7, 47, 54]. We point out that both SFT and preference learning steer the scoring model toward mimicking the generators rather than being optimized directly for detection. In other words, the training goal of previous training-based methods is model-oriented, rather than task-oriented. This leads to the fact that the scoring model can only learn the knowledge of the generators of the training data, but cannot learn the knowledge of the detection task directly. Ultimately, it jeopardizes the generalization and robustness of the detector.

We propose Direct Discrepancy Learning (DDL), a novel optimization strategy that enables the model to learn to be a detector rather than another language model by directly optimizing the scoring model with the output classification metric. DDL originates from the idea of freeing the scoring model from its identity as a language model and designs a task-oriented loss function so that the scoring model can directly learn the intrinsic knowledge of MGTD instead of simply fitting the distribution of training data.

Furthermore, as the fusion of prior approaches [7, 9] and DDL, we propose DetectAnyLLM, a unified MGTD framework. DetectAnyLLM achieves efficient and robust detection through three steps comprising re-sampling, discrepancy calculation, and reference clustering. Such a framework distills the core insights of existing methods [7, 34] while leveraging DDL to enhance the model's generalization capabilities and improve detection robustness.

Despite various MGTD studies have emerged, there remains a lack of comprehensive benchmarks [54]. Existing benchmarks [12, 21, 30, 55] suffer from several significant deficiencies: 1) Limited focus on MRT: Most benchmark datasets, such as MGTBench [21], focus solely on MGT while neglecting the detection of MRT. 2) Narrow range of source LLMs: Most benchmarks rely on small-scale, open-source models, whereas real-world applications often involve advanced, proprietary LLMs such as GPT-40 [24] and Claude [3]. 3) Restricted domain coverage: Benchmarks such as HC3 [17] sample text from only one or a few domains, neglecting the domain sensitivity of machine-generated text. These deficiencies highlight a significant gap between evaluation and real-world applications. Although some recent studies [5, 54] have recognized such problems, their datasets remain insufficiently comprehensive.

To facilitate a comprehensive evaluation, we construct MIRAGE, the largest multi-task MGTD benchmark that provides the richest variety of proprietary LLMs and the most comprehensive text domains in MGTD research. As shown in Table 1, MIRAGE samples text from 10 corpora in 5 common domains, and uses 17 advanced mainstream LLMs for text generation or revision, creating over 93K HWT-MGT pairs. MIRAGE establishes a more realistic and reliable evaluation standard, bridging the gap between research and real-world applications.

Though existing detection methods have demonstrated seemingly outstanding performance ($AUC>0.9$) on previous benchmarks [9, 55], they exhibit significant weaknesses when evaluated on MIRAGE. This reveals the generalization and robustness of prior methods require substantial improvement. In contrast, DetectAnyLLM still performs well, achieving an average of 0.9352 AUROC and 0.7636 MCC on the MIRAGE-DIG subset. Such performance powerfully demonstrates the efficiency and superior generalization of DDL.

Our contributions can be summarized into the following points:
- We propose Direct Discrepancy Learning (DDL), a novel task-oriented optimization method that improves generalization and robustness with fewer resources and no extra data.
- We construct MIRAGE, a comprehensive MGTD benchmark covering diverse domains, tasks, and novelly focuses on using proprietary LLMs, resulting a more realistic evaluation.
- We present DetectAnyLLM, unifying prior works and DDL, achieving up to 70% performance gains and realizing a generalizable and robust detection across domains and models.

---

## 2 Related Work

### 2.1 Zero-shot Detector

Previous MGTD research emphasized zero-shot detection due to concerns about overfitting during training [4, 38]. Early methods like GLTR [14] leveraged text entropy to detect machine content, while others used likelihood- or ranking-based approaches [25, 44]. Recently, DetectGPT [34] provides a novel view for MGTD, it distinguishes MGT from HWT using perturbation, scoring, and probability curvature estimation. Fast-DetectGPT [7] improves the perturbation step, significantly accelerating the detection process without sacrificing performance. Despite progress, zero-shot methods remain constrained by their dependence on the scoring model's output distribution, as shown by Glimpse [6], which demonstrated performance improvement through stronger scoring models.

### 2.2 Training-based Detector

The training-based detector fine-tunes the scoring model on specific training data. An early representative work is RoBERTa-OpenAI-Detector [44], the researchers fine-tune RoBERTa [32] models using GPT-2 [39]-generated data, performing well on detecting GPT-generated text. RADAR [23] incorporates adversarial training [15] to enhance MGTD robustness and uses PPO [42] to optimize the generator. More recently, ImBD [9] utilizes DPO [40] to optimize the scoring model based on the Fast-DetectGPT [7] framework, aiming to help the scoring model better capture the style features of the training data.

Despite these advancements, most methods simply focus on training the scoring model to approximate the source model's distribution rather than developing a dedicated MGTD detector. This introduces constraints to the scoring model during training, which are detrimental to the MGTD task.

### 2.3 MGTD Benchmark

Early benchmarks like Turingbench [50] focused on news articles generated by neural models, while the emergence of ChatGPT [37] shifted attention to LLM-generated text, exemplified by MGTBench [21] and HC3 [17]. Later efforts such as MAGE [30], MULTITUDE [33], and M4 [53] explored open-domain and multilingual detection. RAID [11] novelly introduced decoding strategy considerations to strengthen evaluation robustness, while DetectRL [55] examined vulnerabilities from a writing-attack perspective. However, most of these benchmarks rely on open-source models (indicating limited variousity) and focus mainly on MGT, overlooking more common real-world applications involving MRT, thus limits their applicability to real-world contexts.

HART [5] marked progress by incorporating both MGT and MRT using six advanced LLMs (only four proprietary LLMs), but it remains limited in generator diversity and domain scope. In this study, we scale up the number of generators to 17, where 13 are proprietary LLMs and 4 are advanced open-source LLMs, covering nearly all major LLMs used in real-world applications. Moreover, we sample HWT [20, 43] from five distinct domains and generate both MGT and MRT, ensuring a more comprehensive and representative evaluation. To advance MGTD research and enable fairer comparisons, we advocate for the adoption of a unified benchmark to ensure consistency in evaluation standards. We hope MIRAGE will serve as a valuable step toward achieving this goal.

---

## 3 DetectAnyLLM Framework

DetectAnyLLM builds upon Fast-DetectGPT [7], which determines whether a text is MGT by measuring the log-probability discrepancy between the original text and its perturbed variants [34]. This method involves three key steps: 1) re-sampling the given text, 2) computing the discrepancy between original text and re-sampled text, and 3) making a decision using the discrepancy. DDL is utilized to train the scoring model to enhance steps 1) and 2) so that the detector can more easily distinguish between MGT and HWT. In Section 3.1, we describe how the log-probability discrepancy is calculated. Next, in Section 3.2, we explain the motivation behind our improvements to this detection process, along with the specific designs we introduce. Finally, in Section 3.3, we detail how discrepancy is ultimately used for MGTD within our proposed framework.

### 3.1 Preliminary

**Basic Hypothesis.** Machine-generated text tends to consist of high-probability tokens at each position, whereas human-written text has greater variability. Although sampling strategies like top-k and top-p introduce some randomness, LLMs still generally select tokens with relatively high probabilities. Thus, features in the probability distribution of tokens can serve as useful cues for distinguishing machine-generated text from human-written.

**Probability Discrepancy.** Given a text $x$ and a scoring model $f_{\theta}$ when using a language model $q_{\phi}$ to produce perturbations, the probability discrepancy (i.e., probability curvature) [34] can be expressed as:

$$d(x,f_{\theta},q_{\phi})=\log f_{\theta}(x)-\mathbb{E}_{\tilde{x}\sim q_{\phi}(\cdot|x)}[\log f_{\theta}(\tilde{x})] \quad (1)$$

where $\tilde{x}$ is the perturbed version of $x$ by $q_{\phi}$.

Based on the hypothesis, machine-generated text $x_{m}$ tends to have a high log-probability, whereas its perturbed version $\tilde{x_{m}}$ shows a lower log-probability. In contrast, human-written text $x_{h}$ generally has a lower log-probability. When perturbed, $\tilde{x_{h}}$ tends to show an increase in log-probability, as the perturbation process replaces words in $x_{h}$ with higher-likelihood alternatives according to the model. Thus, we expect to achieve:

$$d(x_m, f_\theta, q_\phi) > d(x_h, f_\theta, q_\phi) \quad (2)$$

This inequality forms the basis of MGTD [7, 9, 34].

When calculating this discrepancy, achieving $f_\theta$ is straightforward, allowing $\log f_{\theta}(x)$ to be efficiently computed. However, since the log-probabilities are computed using Markov-Chain, even a small perturbation requires recalculating the entire chain. Thus, estimating the expectation of the log-probability of $\tilde{x}$ is complex.

**Conditional Probability.** [7] is a biased yet computationally efficient estimation of the original probability:

$$f_{\theta}(\tilde{x}) = \prod_{i}f_{\theta}(\tilde{x}_{i}|\tilde{x}_{<i}) \approx \prod_{i}f_{\theta}(\tilde{x}_{i}|x_{<i}) = f_{\theta}(\tilde{x}|x) \quad (3)$$

By introducing Eq. (3), the probability discrepancy in Eq. (1) can be further reformulated to the conditional probability discrepancy:

$$d_c(x, f_\theta, q_\phi) = \frac{\log f_\theta(x|x) - \tilde{\mu}}{\tilde{\sigma}} \quad (4)$$

where

$$\tilde{\mu}=\mathbb{E}_{\tilde{x}\sim q_{\phi}(\tilde{x}|x)}[\log f_{\theta}(\tilde{x}|x)]$$

$$\tilde{\sigma}^{2}=\mathbb{E}_{\tilde{x}\sim q_{\phi}(\tilde{x}|x)}[(\log f_{\theta}(\tilde{x}|x)-\tilde{\mu})^{2}] \quad (5)$$

Noticing that a normalization item $\tilde{\sigma}$ is added into the discrepancy function, we further explore how the $\tilde{\sigma}$ affects performance in Section 5.2.

**Re-sample text.** Given a sentence consisting of $s$ tokens, we use the model $q_{\phi}$ to compute $q_{\phi}(t|x_{<i})$ for $i$ from 1 to $s$, where $t$ represents for token. This results in a tensor lprobs of shape $(s \times |V|)$, where $|V|$ denotes the vocabulary size of $q_{\phi}$. With such a tensor, we can efficiently generate $n$ re-sampled samples with only a single line of PyTorch code.

For the original version of probability discrepancy [34], both perturbation generation and discrepancy estimation require calculating a whole Markov-Chain for $n$ times. This leads to a time complexity of $\mathcal{O}(n \times s)$.

By introducing conditional probability [7], the resampling approach can replace the perturbation step. Under this formulation, both generating $n$ samples and computing the discrepancy require running the Markov-Chain only once. As a result, the time complexity is reduced to $\mathcal{O}(s)$.

### 3.2 Optimizing by Direct Discrepancy Learning

As shown in Eq. (4) and Eq. (2), the key to enhance the detector's performance is to increase the distribution difference of the conditional probability discrepancy between MGT and HWT estimated by the scoring model.

While ImBD [9] has achieved significant performance gains by incorporating Direct Preference Optimization (DPO) [40] to optimize the scoring model, we argue that DPO is not the optimal optimization method for the MGTD task.

**DPO.** [40] is derived from the optimization objective of Proximal Policy Optimization (PPO) [42], which is:

$$\max_{\theta}\mathbb{E}_{x\sim f_{\theta}(x)}[r(x)]-\beta\mathbb{D}_{KL}[f_{\theta}(x)||f_{ref}(x)] \quad (6)$$

where $x$ is a text sampled from the scoring model $f_{\theta}$ distribution, and $r$ is a reward function that can judge whether this sample is bad or good. By analyzing and re-parameterizing this optimization objective, we can obtain DPO's optimization objective:

$$\max_{\theta}\mathbb{E}_{x_{m},x_{h}\sim D}\left[\log \sigma\left(\beta \log\frac{f_{\theta}(x_{m})}{f_{ref}(x_{m})}-\beta \log\frac{f_{\theta}(x_{h})}{f_{ref}(x_{h})}\right)\right] \quad (7)$$

where $x_{m}$ denotes MGT and $x_{h}$ stands for HWT. $f_{ref}$ is a reference model, usually the original $f_{\theta}$. The detailed derivation process will be presented in the supplementary material.

**Motivated by redundant KL-regularization.** The KL term between $f_{\theta}$ and $f_{ref}$ is explicitly added to the optimization objective in PPO [42], and its weight is adjusted via $\beta$, as shown in Eq. (6). While in DPO [40], as Eq. (7) shown, such regularization is implicitly embedded in the optimization objective, and its strength can also be adjusted by $\beta$. ImBD [9] directly adopts Eq.(7) as its loss function and leverages paired MGT-HWT data to optimize the scoring model $f_{\theta}$. The KL-regularization forces the scoring model to retain its internal knowledge while learning preferences.

This leads us to question: for the MGTD task, what is the significance of retaining the original knowledge of the scoring model during training?

Since we have introduced training, our direct objective should be enable the scoring model to better capture the knowledge of the MGTD task. Fundamentally, we hope that the training process will teach the scoring model how to become a detector. However, the KL-regularization drastically shifts this objective: from learning the intrinsic knowledge of the MGTD task to aligning the scoring model with the distribution of training data. This shifts the training process from learn a detector to mimic a language model, thereby misleading the scoring model.

**Direct Discrepancy Learning.** Based on the reasoning above, we remove the KL-regularization in the optimization objective. Thus, the optimization goal can be re-written as:

$$\max \mathbb{E}_{x\sim D}[r(x)] \quad (8)$$

We further design a simple but task-oriented reward objective $r(x)$, defined as:

$$r(x) = \begin{cases}
-\| \gamma - d_c(x, f_\theta, f_\theta) \|_1, & \text{when } x \text{ is } x_m \\
-\| d_c(x, f_\theta, f_\theta) \|_1, & \text{when } x \text{ is } x_h
\end{cases} \quad (9)$$

where $\gamma$ is an hyper-parameter. This reward function is designed based on the conclusions discussed in Section 3.1, that is the discrepancy of human-written text $x_{h}$ tends to be low (close to 0) while the discrepancy of machine-generated text $x_{m}$ tends to be positive. The parameter $\gamma$ is introduced to control how positive the discrepancy of $x_{m}$ should be. In our experiment, $\gamma$ is arbitrarily chosen. As shown in Table 4, an experiment on the impact of the value of $\gamma$ shows that the model's performance is not particularly sensitive to this choice, indicating a level of robustness to variations in $\gamma$.

In practice, our input consists of paired HWT-MGT data. We set $q_{\phi}=f_{\theta}$ following the ImBD [9]'s setting, which allows us to use the scoring model's output for optimization:

$$\min_{\theta}\mathbb{E}_{x_{m},x_{h}\sim D}(\|d_{c}(x_{h},f_{\theta},f_{\theta})\|_{1}+\|\gamma-d_{c}(x_{m},f_{\theta},f_{\theta})\|_{1}) \quad (10)$$

We call this optimization method as Direct Discrepancy Learning (DDL), as it helps the scoring model directly learn the expected conditional probability discrepancy of both MGT and HWT.

By removing the KL-regularization, the scoring model can essentially forget its identity as a language model. Furthermore, the reward function based on the discrepancy $d_{c}$, which incorporates a task-oriented prior, can help the scoring model to directly learn the inherent knowledge of MGTD. Specifically, $d_{c}$ for HWT approaches 0, while $d_{c}$ for MGT is positive.

### 3.3 Detecting by Reference Clustering

We use Reference Clustering to achieve the transition from $d_{c}(x)$ to $p_{m}(x)$. Specifically, this algorithm is designed to estimate the probability of a given value belonging to a specific distribution, consisting of: data aggregation and probability estimation.

**Data Aggregation.** We first collect a certain number of MGT texts as the MGT reference dataset $M$, and an approximately equal number of HWT texts as the HWT reference dataset $H$. Then, we employ the scoring model $f_{\theta}$, which will be used for detection, to respectively compute the conditional probability discrepancy $d_c$ for each text in $M$ and $H$. Thereby, we can obtain the conditional probability discrepancy distribution $D_{m}$ and $D_{h}$ of the texts in $M$ and $H$ under scored by $f_{\theta}$.

**Probability Estimation.** We select the value in $M \cup H$ that is $k_{th}$ closest to the target value $d_{c}(x)$ as the search window $\delta$:

$$S=\text{sorted}(\{\|d_{c}(x_{ref})-d_{c}(x)\|_{1} \mid x_{ref}\in M\cup H\})[k] \quad (11)$$

where $k$ is a hyper-parameter that should be determined by the size of reference dataset. For a larger reference dataset, a larger $k$ is better as it can provide higher precision of $p_{m}(x)$.

Then, we count the number of MGT texts and HWT texts within the window range:

$$cnt_{m}=\sum_{d\in D_{m}}I(d_{c}(x)-\delta<d<d_{c}(x)+\delta)$$

$$cnt_{h}=\sum_{d\in D_{h}}I(d_{c}(x)-\delta<d<d_{c}(x)+\delta) \quad (12)$$

Finally, we estimate the probability that text $x$ belongs to MGT using the local statistical ratio:

$$p_m(x) = \frac{cnt_m}{cnt_m + cnt_h} \quad (13)$$

Since the window $\delta$ is adaptively determined by the data distribution, this method can maintain stability under different data densities, thereby improving the robustness of real-world MGTD.

---

## 4 Proposed MIRAGE Benchmark

Current benchmarks exhibit notable limitations in diversity of text domains [50, 55], coverage of source LLMs [11, 53], and evaluation tasks [17, 30]. To facilitate a generalized evaluation that better reflects real-world application, we present the Multi-domain Inclusive Realistic Assessment for machine Generated text detection (MIRAGE) benchmark. MIRAGE constitutes the most comprehensive multi-task MGTD evaluation framework to date, incorporating both generative and revisionary text across diverse domains, employing most advanced LLMs, including 13 proprietary and 4 open-source LLMs.

### 4.1 Benchmark Construction

**Multi-domain Sampling.** Considering that LLMs exhibit varying performance across different text domains, MIRAGE samples HWT of 5 domains from 10 corpora. Detail information is presented in Supplementary Material.

**Pre-Cleaning.** We remove all the '\n' character to prevent the detector from identifying MGT based on the presence of the '\n' symbol. Subsequently, we filter out texts containing 100-200 words from these datasets to control for length-based detection biases.

**Inclusive MGT Tasks.** Following established methodologies in the [7] and [9], we designed three distinct MGT tasks: Generate, Polish, and Rewrite. The Generate task involves creating new text based on the first 30 tokens of an HWT. The Polish task refines an existing HWT while preserving its original details and meaning. The Rewrite task paraphrases a given HWT without altering its meaning or essential details. The detailed prompt of each task will be presented in Supplementary Material.

**Realistic LLM Usage.** In real-world applications, people typically rely on powerful proprietary LLMs to generate or revise text. However, most existing benchmarks [11, 17, 30, 50, 53, 55] rely on open-source LLMs to build data, resulting in a gap between current evaluation and real-world applications. To address this, MIRAGE incorporates 13 mainstream proprietary LLMs, as detailed in Supplementary Material. Concurrently, recognizing the increasing deployment of high-performance open-source models in localized applications, we incorporated four advanced open-source LLMs [16, 57], ensuring comprehensive coverage of the contemporary LLM ecosystem.

**Composition.** We consider two distinct evaluation scenarios to better reflect real-world applications:
- **Disjoint-Input Generation(DIG):** Each LLM generates MGT or MRT based on a unique HWT. Detectors must distinguish between this machine output and its source HWT.
- **Shared-Input Generation (SIG):** Multiple LLMs generate MGT or MRT from the same HWT. Detectors must identify all machine outputs from a common input.

We design each LLM to generate 2,000 samples for each MGT task, equally distributed between DIG and SIG scenarios (1,000 samples each). Both DIG and SIG follow the same domain distribution for consistency, as detailed in Supplementary Material.

Sampling begins with constructing domain level HWT datasets by proportionally merging source datasets within each domain. Such dataset-mixing strategy eases dataset-bias by preventing oversampling from single dataset. During implementation, SIG is treated as an independent "model" and incorporates alongside the 17 individual LLMs in the sampling process. For each model (including SIG), we sequentially sample data from each domain dataset. Once sampled, items are removed from corresponding domain datasets to maintain the distinction between DIG and SIG data. Within each text domain, data is sampled continuously until the number of samples for that domain meets the requirements specified in Supplementary Material. Once the data sampling for one text domain is complete, the process moves to the next, repeating until all text domains have been sampled.

This methodology produces the DIG dataset for each LLM and a comprehensive SIG dataset, which are subsequently combined to form the complete sample set for each LLM across all tasks.

**Data Augmentation.** The language style is a key distinguishing feature between HWT and MGT, with a closer alignment to human language style posing a more challenging task for MGT detectors. With this consideration, we introduce data augmentation in terms of LLM's language styles. Specifically, we incorporated the phrase "in a \<style\> style" into the input prompt. We manually select 16 different language styles, randomly choosing one during each LLM inference to achieve style-diversity. This approach helps assess the robustness of detectors against language styles' attacks.

**Post Cleaning.** After generating MGT or MRT from the above HWT data, we perform data cleaning on the generated data. First, all the '\n' and '\r' are removed, to prevent detection from symbol's feature. Next, we remove texts with fewer than 90 words or more than 220 words, to prevent the impact of text length variations on detection, and finally obtain the MIRAGE benchmark dataset. The statistical results are presented in Supplementary Material.

### 4.2 Evaluation Metrics

Consistent with prior works [7, 9, 34], we adopt the Area Under the Receiver Operating Characteristic Curve (AUROC) as the primary evaluation metric. To assess the performance on specific threshold, we incorporate TPR at a 5% false positive rate (TPR@5%) as a supplementary metric. Furthermore, considering the MIRAGE-SIG is a class-imbalanced dataset, we additionally report the Matthews Correlation Coefficient (MCC) and Balanced Accuracy to provide a more comprehensive evaluation. Together, this diverse set of metrics provides a comprehensive assessment of detector's performance, ensuring that the evaluation reflects both theoretical completeness and real-world applicability.

---

## 5 Experiment

### 5.1 Main Results

**Training settings.** The scoring model and training data used in DetectAnyLLM are set exactly the same as [9] to ensure a fair comparison. Detailed training settings are provided in the Supplementary Material. The $\gamma$ in DDL is set to 100, and we will discuss how the $\gamma$ affects the performance in Section 5.2.

**Baselines.** For a comprehensive comparison, we compare the performance of our method with baseline methods, advanced zero-shot methods, and state-of-the-art training-based methods. The baseline methods including Likelihood [44], Log-Rank [25], LRR [46], and Entropy [14]. The advanced zero-shot methods includes DetectGPT [34], NPR [46], and Fast-DetectGPT [7]. The training-based methods includes RoBERTa series [32, 44] and ImBD [9].

**Results on MIRAGE-DIG.** As the top of Table 2 shown, DetectAnyLLM achieves substantial performance improvement over all baselines across all metrics and tasks. Specifically, it delivers AUROC relative gains of up to +64.78%~+66.71%, with MCC improvements reaching up to +56.44%. DetectAnyLLM also maintains robust TPR@5% across all tasks, outperforming previous training-based SOTA ImBD [9] by large margins (+60.84%~+69.13%).

**Results on MIRAGE-SIG.** As the bottom of Table 2 shown, DetectAnyLLM continues to lead in the MIRAGE-SIG subset, reaching AUROC of 0.9526, Balanced Accuracy of 0.9059, and TPR@5% up to 0.7779, again greatly surpassing all other methods.

The results on MIRAGE highlight DetectAnyLLM's strong generalization capacity and robustness across diverse source LLMs and text-domains, demonstrating the great effectiveness of DDL.

**Detection on the previous test sets.** We evaluate the performance of DetectAnyLLM on the three test sets used by ImBD [9]. As Table 3 shown, DetectAnyLLM consistently outperforms all existing MGTD methods.

Comparing Table 3 and Table 2, we observe that the baseline methods exhibit great performance degradation on MIRAGE. Such an observation reveals the limitations of existing test benchmarks in comprehensively evaluating detectors' abilities, underscoring the importance of MIRAGE as a more challenging benchmark.

**Efficiency.** Since DDL performs optimization without relying on a reference model, it achieves substantial improvements in training efficiency compared to Style Preference Optimization (SPO) [9]. DDL demonstrates a +30.12% reduction in training time and a +35.90% reduction in memory consumption relative to SPO [9]. Details are provided in Supplementary Material.

### 5.2 Ablation Study

**Ablation on parameter $\gamma$.** As shown in Table 4, DDL exhibits strong robustness to the values of $\gamma$. Comparing to the results in Table 2, for all selected values of $\gamma$, the DDL-trained detector consistently outperforms all prior state-of-the-art methods in terms of AUROC. Detailed results, comprehensive analysis, and discussion are provided in Supplementary Material.

**Ablation on KL-strength $\beta$ in SPO [9].** We provide comprehensive experiments and confirm our point in Section 3.2. For more information, please see Supplementary Material.

**Ablation on model.** We retrain Qwen2-0.5B [56], GPT-J-6B [51], and GPT-Neo-2.7B [8] using SPO [9] and DDL. The models are then evaluated on the Rewrite task of MIRAGE-SIG.

As Table 5 shown, the DDL-optimized detector consistently outperforms the SPO [9]-optimized ones across all model sizes, confirming DDL's robustness and adaptability. It is worth noting that the detector trained using a smaller but more advanced LLM, such as the Qwen2-0.5B model, achieves a better performance. This shows that the ability of the scoring model largely affects the upper limit of the detector's ability.

**Ablation on normalization $\tilde{\sigma}$.** As shown in Table 6, the removal of $\tilde{\sigma}$ leads to a substantial degradation in performance across all metrics, highlighting the importance of $\tilde{\sigma}$ for stable and effective optimization. Despite this, DDL without normalization still surpasses previous state-of-the-art methods on most metrics reported in Table 2, underscoring the robustness of DDL. We suggest that the normalization item $\tilde{\sigma}$ helps standardize the output across diverse source LLMs and domains, thereby facilitating more consistent and generalizable learning.

---

## 6 Conclusion

In this study, we have introduced a novel optimization strategy, Direct Discrepancy Learning (DDL), and developed a unified detection framework named DetectAnyLLM. Our approach enables the scoring model to acquire task-oriented knowledge by directly leveraging discrepancy signals and achieves high-precision detection through a technique we call reference clustering. We also proposed MIRAGE, a comprehensive benchmark dataset that spans a wide range of text domains, the most advanced LLMs, and generation tasks. To thoroughly evaluate detector performance, we assessed DetectAnyLLM and existing MGTD methods under two settings: Disjoint-Input Generation and Shared-Input Generation. Experimental results on both MIRAGE and previously established test sets demonstrate that DetectAnyLLM significantly outperforms existing MGTD methods, establishing a new state-of-the-art in this domain.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (62306153, 62225604), the Natural Science Foundation of Tianjin, China (24JCJQJC00020), the Young Elite Scientists Sponsorship Program by CAST (YESS20240686), the Fundamental Research Funds for the Central Universities (Nankai University, 070-63243143), and Shenzhen Science and Technology Program (JCYJ20240813114237048). The computational devices are partly supported by the Supercomputing Center of Nankai University (NKSC).

## Tables

**Table 1:** Comparison between MIRAGE and existing MGTD benchmark datasets. "Size" is the capacity of the test set. "SIG" denotes Shared-Input Generation and "DIG" denotes Disjoint-Input Generation. "Commercial" refers to the use of frontier proprietary LLMs (e.g., GPT-4o). MIRAGE is the most diverse benchmark in terms of domain, tasks, and source LLMs. MIRAGE leverages powerful proprietary LLMs to generate and revise text, increasing the difficulty of detection and the realism of evaluation, enabling a more faithful evaluation of detector robustness. Furthermore, MIRAGE introduces a novel dual-scenario evaluation strategy - DIG and SIG - allowing more comprehensive assessment of both accuracy and generalization capacity.
| Benchmark | Size | Domain Coverage | Corpus | LLMs | Commercial | Generate | Polish | Rewrite | Aug. Other | SIG | DIG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TuringBench [50] | 40K | News | 3 | X | X | X | | | | | |
| HC3 [17] | 85K | QA/Comment/Academic | 5 | 1 | X | | | | | | |
| M4 [53] | 24.5K | QA/Comment/Academic/News | 11 | 2 | X | | | | | | |
| MAGE [30] | 29K | QA/Comment/News/Academic/Story | 10 | 3 | X | | | | | | |
| RAID [11] | 628.7K | News/Academic/Comment/Literature | 11 | 3 | ✓ | X | X | | | | |
| DetectRL [55] | 134.4K | Academic/Comment | 4 | 2 | X | | | | | | |
| HART [5] | 16K | News/Literature/Academic | 4 | 4 | ✓ | X | | | | | |
| **MIRAGE (ours)** | **93.8K** | **Academic/Comment/Email/News/Website** | **10** | **13** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

**Table 2:** Results across three tasks (Generate, Polish, Rewrite) under two evaluation settings (MIRAGE-DIG and MIRAGE-SIG). All methods, except for RoBERTa-Base/Large, employ GPT-Neo-2.7B [8] as the scoring model. Following the experiment settings of [9], NPR [46] and DetectGPT [34] use T5-3B [41] to generate perturbations, while Fast-DetectGPT [7] utilizes GPT-J-6B [51] to generate samples. * indicates a training-based method, whereas $\dagger$ denotes a method that requires multiple model invocations. "Imp." represents Improvement over previous SOTA, computed as $(new - old)/(1.0 - old)$. Metrics: AUROC, Balanced Accuracy, MCC, and TPR@5%. DetectAnyLLM significantly outperforms all baselines across all tasks and settings.

*(Note: Data for Table 2 has been omitted in this markdown due to extreme formatting issues in the raw data, but it summarizes that DetectAnyLLM significantly outperforms baselines by +55% to +69% relative gains across Generate, Polish, and Rewrite).*

**Table 3:** Results of detection on the previous test sets. * indicates a training-based method, whereas $\dagger$ denotes a method that requires multiple model invocations. "Imp." represents Improvement, computed as $(new - old) / (1.0 - old)$.
| Methods | XSum [35] | Writing [13] | PubMed [28] |
| :--- | :--- | :--- | :--- |
| Likelihood [44] | 0.4396 | 0.8077 | 0.4596 |
| LogRank [25] | 0.4002 | 0.7694 | 0.4472 |
| Entropy [14] | 0.6122 | 0.8202 | 0.5899 |
| RoBERTa-Base [32] * | 0.4921 | 0.4774 | 0.2496 |
| RoBERTa-Large [32] * | 0.4782 | 0.4708 | 0.3089 |
| LRR [46] | 0.3095 | 0.6214 | 0.4710 |
| DNA-GPT [58] | 0.4974 | 0.7478 | 0.3151 |
| NPR [46] $\dagger$ | 0.5065 | 0.8444 | 0.3740 |
| DetectGPT [34] $\dagger$ | 0.6217 | 0.8771 | 0.5612 |
| Fast-DetectGPT [7] $\dagger$ | 0.6293 | 0.8324 | 0.6175 |
| ImBD [9] * | 0.9486 | 0.9468 | 0.7743 |
| **DetectAnyLLM (ours) *** | **0.9880** | **0.9671** | **0.8817** |
| *Imp.* | *+80.16%* | *+38.16%* | *+47.59%* |

**Table 4:** Results of different $\gamma$ in DDL. Metrics with subscript $t$ correspond to the training set, and subscript $v$ indicates evaluation on the polish task of MIRAGE-DIG.
| Metric | $\gamma=10$ | $\gamma=20$ | $\gamma=50$ | $\gamma=100$ | $\gamma=500$ | $\gamma=10000$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $AUROC_t$ | 0.9964 | 0.9934 | 0.9883 | 0.9861 | 0.9861 | 0.9861 |
| $AUPR_t$ | 0.9965 | 0.9938 | 0.9888 | 0.9833 | 0.9833 | 0.9833 |
| $AUROC_v$ | 0.8692 | 0.9257 | 0.9347 | 0.9259 | 0.9259 | 0.9259 |
| $AUPR_v$ | 0.8735 | 0.9294 | 0.9458 | 0.9373 | 0.9373 | 0.9373 |

**Table 5:** Ablation study on the impact of scoring model. All models are trained on the same data and evaluated on the MIRAGE-SIG Rewrite task.
| Method | Base Model | AUROC | Accuracy | MCC | TPR@5% |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SPO [9] | GPT-Neo-2.7B | 0.7694 | 0.6920 | 0.3936 | 0.2868 |
| | GPT-J-6B | 0.8367 | 0.7557 | 0.5155 | 0.4722 |
| | Qwen2-0.5B | 0.9370 | 0.8169 | 0.8575 | 0.9071 |
| DDL (ours) | GPT-Neo-2.7B | 0.9158 | 0.8643 | 0.7320 | 0.7574 |
| | GPT-J-6B | 0.8909 | 0.8424 | 0.6878 | 0.6047 |
| | Qwen2-0.5B | 0.8570 | 0.7816 | 0.5632 | 0.4508 |

**Table 6:** Ablated result of $\tilde{\sigma}$. "norm." means "normalization". "w/" means "with" and "w/o" means "without". Scoring model: GPT-Neo-2.7B [8]. Benchmark: MIRAGE-DIG-polish.
| Metric | AUROC | Accuracy | MCC | TPR@5% |
| :--- | :--- | :--- | :--- | :--- |
| DDL w/o norm. | 0.8499 | 0.7759 | 0.5563 | 0.5232 |
| DDL w/ norm. | 0.9297 | 0.8732 | 0.7487 | 0.7756 |

**Table 7:** Source datasets for each domain.
- **Academic:** NeurIPS, ArXiv, PubMed-Abstracts [28]
- **eMail:** Enron-Emails
- **Website:** OpenWebText
- **News:** BigPatent [43], CNNDailyMails, XSum [35], XLSum [20]
- **Comment:** Amazon-Review

**Table 8:** Text domain composition that each LLM is required to perform generation task in both DIG and SIG.
- Academic: 300
- Mail: 100
- Website: 200
- News: 200
- Comment: 200
- Total: 1000

**Table 9:** Statistic result of MIRAGE.
| Tasks | Count |
| :--- | :--- |
| Disjoint-Input Generation Gen. | 16412 |
| Disjoint-Input Generation Pol. | 14776 |
| Disjoint-Input Generation Rew. | 15735 |
| Shared-Input Generation Gen. | 16388 |
| Shared-Input Generation Pol. | 14776 |
| Shared-Input Generation Rew. | 15751 |

**Table 10:** Commercial LLMs are highlighted in bold.
- **GPT:** **GPT-4o [24], GPT-o3-mini [26], GPT-4o-mini [24]**
- **Claude:** **Claude-3.5-Haiku, Claude-3.7-sonnet [3]**
- **DeepSeek:** **DeepSeek-V3 [31], DeepSeek-R1 [18]**
- **Gemini:** **Gemini-2.0-Flash, Gemini-2.0-Flash-Lite [48]**
- **Qwen:** **Qwen-2.5-7B [57], Qwen-2.5-7B-R1-Distill [18], QwQ-Plus**
- **LlaMA:** LLaMA-3.1-8B [16], LLaMA-3.1-8B-R1-Distill [18]
- **Grok:** **Grok2**
- **Moonshot:** **Moonshot-v1**
- **Doubao:** **Doubao-1.5-pro-32k**

**Table 11:** Detail style list.
Style List: formal, oral, academic, literary, critical, narrative, descriptive, lyric, objective, subjective, original, casual, expository, argumentative, journalistic, poetic

---

## Appendix

### A More details on DDL

**A.1 Derivation of DPO**

Direct Preference Learning (DPO) [40] is derived from the optimization objective of Proximal Policy Optimization (PPO) [42], which is formulated as:

$$\max_{\theta}\mathbb{E}_{x\sim f_{\theta}(x)}[r(x)]-\beta\mathbb{D}_{KL}[(f_{\theta}(x)||f_{ref}(x)] \quad (14)$$

where $x$ is a text sampled from the scoring model $f_{\theta}$ distribution, and $r$ is a reward function that can judge whether this sample is bad or good. We can obtain DPO's optimization objective by analyzing and re-parameterizing this optimization objective. The $f_{ref}$ stands for the reference model, usually the original $f_{\theta}$. DPO [40] starts with explicit solution $f_{\theta}=p_{r}$ of Eq. (14):

$$p_{r}(x)=\frac{1}{Z(x)}f_{ref}(x)\exp(\frac{1}{\beta}r(x)) \quad (15)$$

where:

$$Z(x)=\sum_{x}f_{ref}(x)\exp(\frac{1}{\beta}r(x)) \quad (16)$$

Furthermore, we can reparameterize $r$ as:

$$r(x)=\beta \log\frac{p_{r}(x)}{f_{ref}(x)}+\beta \log Z(x) \quad (17)$$

where, $p_{r}$ is the best solution of Eq. (14), which we want $f_{\theta}$ to become. Now, if we introduce the Bradley-Terry model to present the model's preference $L$ between HWT $x_{h}$ and MGT $x_{m}$, we can get:

$$L(x_{m}>x_{h})=\sigma(r(x_{m})-r(x_{h}))$$

$$= \sigma\left(\beta\log\frac{p_r(x_m)}{f_{ref}(x_m)} - \beta\log\frac{p_r(x_h)}{f_{ref}(x_h)}\right) \quad (18)$$

where we surprisingly reduced the partition function $Z(x)$. By replacing $p_{r}$ with $f_{\theta}$ and utilizing Maximum-Likelihood Estimation to Eq. (18), we can finally present the DPO [40] optimize goal:

$$\max_{\theta}\mathbb{E}_{x_{m},x_{h}\sim D}\left[\log \sigma\left(\beta \log\frac{f_{\theta}(x_{m})}{f_{ref}(x_{m})}-\beta \log\frac{f_{\theta}(x_{h})}{f_{ref}(x_{h})}\right)\right] \quad (19)$$

where $x_{m}$ denotes MGT and $x_{h}$ stands for HWT. $f_{ref}$ is a reference model, usually the original $f_{\theta}$.

### B More details on MIRAGE

**B.1 More details on Data Source**

**Time Bound.** To ensure that all sampled texts are human-written and free from contamination by LLM-generated content, most of the source datasets used were constructed prior to 2021. For datasets containing data collected after 2021, we cleaned the data denoted collected after 2021 in these datasets to ensure the purity and authenticity of the human-written source material.

**Source Domains and Datasets.** MIRAGE encompasses a diverse range of text domains, including Academic, E-Mail, Website, News, and Comment. The mapping between these domains and the corresponding source datasets is summarized in Table 7. Additionally, we implement domain-specific pre-processing: extracting only abstracts from academic publications (NeurIPS and ArXiv) and isolating message content from email communications (Enron-Emails dataset).

**Domain Composition.** As the amount of data varies across domains, the text domains are not treated equally in terms of quantity. However, for both DIG and SIG, the proportion of texts from each domain that each LLM is required to generate or revise remains fixed. The detailed domain distribution is shown in Table 8.

**Statistic Result.** Table 9 presents the overall statistics of the MIRAGE dataset across the two task settings: Disjoint-Input Generation and Shared-Input Generation. For each setting, the dataset includes three task types-Generate (Gen.), Polish (Pol.), and Rewrite (Rew.). The number of instances is balanced across both settings, with each task type containing approximately 14,000 to 16,000 samples. This balanced distribution ensures that the dataset supports a comprehensive evaluation of LLMs across different generation and revision scenarios.

**B.2 Source Generator LLMs**

The source LLMs used for data generation in MIRAGE are listed in Table 10. In total, MIRAGE samples machine-generated texts (MGT) using 13 powerful commercial LLMs and 4 advanced open-source LLMs. This selection reflects a strong emphasis on evaluating detection performance in real-world applicant scenarios, while still maintaining attention to the open-source LLM landscape.

**B.3 Prompts for Generation Tasks**

The system prompts used for all three generation tasks are the same, specifically, "You are a professional writing assistant who can write high-quality, coherent, and engaging articles." We add a style control signal to the user prompt, in order to perform data augmentation, thus promoting the robustness of our benchmark.

**Style Control.** The style control signal is directly added to the user prompt, as "in a \<style\> style". The \<style\> is randomly chosen from a prepared style list, detail as shown in Table 11.

**Prompt for Generate.** "Write an article about 150 words in a \<style\> style starting exactly with: \<original\>". The \<original\> is the first 30 tokens of a HWT.

**Prompt for Polish.** "Polish the following text in a \<style\> style without missing any original details. Ensure that the length of the polished text is similar to the original text. Directly output your polished text. Here is the original text: \<original\>" The \<original\> is a complete HWT.

**Prompt for Rewrite.** "Paraphrase the following text in a \<style\> style without missing any original details. Ensure that the length of the paraphrased text is similar to the original text. Directly output your paraphrased text. Here is the original text: \<original\>" The \<original\> is a complete HWT.

**B.4 More Details on Evaluation Metrics**

Consistent with prior work [7, 9], we adopt the Area Under the Receiver Operating Characteristic Curve (AUROC) as the primary evaluation metric for assessing the performance of the MGT Detector. While AUROC provides a threshold-independent measure of classification capability, it does not necessarily reflect the detector's effectiveness at specific operating points, which are often critical in real-world deployments.

To address this limitation, we incorporate TPR at a 5% false positive rate (TPR@5%) as an important supplementary metric. TPR@5% reflects the detector's sensitivity when operating under a strict false positive constraint, which is especially important for applications demanding high precision.

Furthermore, considering the MIRAGE-SIG is a class-imbalanced dataset, we additionally report the Matthews Correlation Coefficient (MCC) and Balanced Accuracy to provide a more comprehensive evaluation. MCC captures the overall quality of binary classifications by considering all four elements of the confusion matrix, making it particularly informative under class imbalance. Balanced Accuracy, used in place of standard accuracy, is computed as the arithmetic mean of the true positive rate and true negative rate, making it better suited for evaluating performance on imbalanced datasets.

Together, this diverse set of metrics provides a comprehensive assessment of the detector's performance, ensuring that the evaluation reflects both theoretical completeness and real-world applicability.

### C More details on Experiment

**C.1 Experiment Setup**

**Device.** All of our experiments are conducted in Linux 4.18.0 (CentOS 7), using a single NVIDIA A40 GPU with 48GB GPU memory. The Python version is 3.10.16, the PyTorch version is 2.5.1, the Transformers version is 4.47.1, and the Datasets version is 3.2.0.

**Training Dataset.** We train DetectAnyLLM in the dataset used in ImBD [9], specifically, 500 pairs of HWT-MGT data, where MGT is machine-polished text created by GPT-3.5-Turbo.

**LoRA Config.** Following the settings of ImBD [9], we adopt a LoRA configuration specifically designed for causal language modeling, with a rank of 8, a LoRA alpha of 32, and a dropout rate of 0.1.

**Settings for Reproducing ImBD.** We reproduced ImBD [9] for comparative evaluation, following the original training configuration described in the paper. Specifically, we set the learning rate to 0.0001 and used a beta coefficient of 0.05. The only modification made was increasing the number of training epochs from 2 (as reported in the original paper) to 5, in order to ensure full convergence of the model. Throughout the training process, we monitored the model's performance on the validation set to prevent overfitting and to verify that the reproduced ImBD model maintained comparable performance to the original.

**Settings for Training DetectAnyLLM.** We train DetectAnyLLM using the exact same hyperparameters as those used in our reproduction of ImBD [9], including a learning rate of 0.0001 and 5 training epochs. For the optimization objective in Direct Discrepancy Learning (DDL), we set the hyperparameter $\gamma$ to 100. That is because increasing $\gamma$ beyond this value did not lead to further improvements in performance, suggesting that the model had fully converged. Moreover, since the model's performance remains stable for larger values of $\gamma$, this setting also ensures compatibility with varying training environments, as the optimal value of $\gamma$ is unknown, we simply choose a sufficiently large value that provides a safe and generalizable configuration.

**C.2 Empirical Validation of Redundant KL-Regularization**

To evaluate the impact of the KL-regularization term in DPO-style training, we conduct ablation studies on a wide range of $\beta$ values, which directly control the strength of the implicit KL constraint. According to the formulation in Section 3.2, a larger $\beta$ enforces stronger alignment between the scoring model $f_\theta$ and the reference model $f_{ref}$, effectively constraining the learning objective toward distributional conformity rather than task-specific discriminability.

*(Table 12 data indicates that KL-regularization can be redundant or detrimental for the MGTD task as performance degrades symmetrically as $\beta$ increases)*

**C.3 More Details on Main Results**

**Efficiency Improvement.** When using the scoring model $f_{\theta}$ as the sampling model $q_{\phi}$, as discussed in Section 3.2, DDL eliminates the need to load a separate reference model during training, unlike SPO [9]. This design enables DDL to train with a single model, leading to notable improvements in training efficiency. Table 13 presents a detailed comparison of training time and memory usage between SPO [9] and DDL.

**Table 13:** Training time cost comparison between DDL and SPO [9]. Device: single NVIDIA A40. Model: GPT-J-6B [51].
| Optim. | Batch Size | Time Cost/Epoch | Memory Usage |
| :--- | :--- | :--- | :--- |
| SPO [9] | 1 | 166s | 31.45GB |
| DDL (ours) | 1 | 116s | 20.16GB |
| Imp. | | +30.12% | +35.90% |

SPO [9] requires loading two large models simultaneously during training, resulting in high memory demands-specifically, 31.45GB for training with GPT-J-6B [51]. This exceeds the capacity of many commonly available GPUs. In contrast, DDL only requires 20.16GB of memory, making it feasible to train on widely accessible GPUs.

**C.4 Discussion of $\gamma$ in DDL**

**Detail Look of $\gamma$'s effect.** Although there are clear performance peaks at specific values (e.g., $\gamma=5$ for the training set and $\gamma=30-40$ for the validation set), the metrics remain consistently high across a wide range of $\gamma$ values.

**DDL's Robustness on $\gamma$.** While the average discrepancies and training metrics vary as $\gamma$ increases, the evaluation performance remains relatively stable across a broad range.

**Explanation of $\gamma$'s Effectiveness.** The robustness of our method to the hyperparameter $\gamma$ arises naturally from the design of the Direct Discrepancy Learning (DDL). In DDL, $\gamma$ serves as a margin that guides the optimization: it encourages the model to keep the discrepancy score $D(x_{m})$ of MGT close to $\gamma$, while minimizing the discrepancy score $D(x_{h})$ for HWT toward zero.

**Explanation of Performance Plateau.** Once $\gamma$ exceeds a certain threshold (e.g., $\gamma\ge30$), we observe that performance metrics are saturated.

**Reason of Setting $\gamma$ to 100.** We choose $\gamma=100$ in our main results because performance has already plateaued by this margin, and higher values help ensure stability.

**C.5 Detection Results on specific LLM**

We expand the main results in Section 5.1 in the specific LLM level to obtain the specific detection capabilities of different methods on texts generated by specific LLMs.

**Results.** DetectAnyLLM achieves consistently strong performance across all metrics, domains, tasks, and source LLMs. On the Polish and Rewrite tasks, it outperforms previous state-of-the-art methods by an average margin of nearly 70%. In certain settings involving text generated by specific LLMs, FastDetectGPT and ImBD slightly outperform DetectAnyLLM on simpler Generate tasks.
