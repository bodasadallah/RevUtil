

################################################## SYNTHETIC DATA GENERATION PROMPTS ##################################################

SYSTEM_PROMPT = '''
You are an expert in evaluating peer review comments with respect to different aspects.
'''
BASE_PROMPT = '''
These aspects are aimed to maximize the utilization of the review comments for the authors. The primary purpose of the review is to help/guide authors in improving their drafts. Keep this in mind while evaluating the review point. Whenever you encounter a borderline case, think: “Will this review point help authors improve their draft?”. There is no correlation between the aspect score and the length of the review point.

{aspect}:
{aspect_description}

Review Point:
{review_point}

Please evaluate the review point based on the aspect description provided above. generate the output as a json object with the following format:
    "rationale": RATIONALE
    "score": SCORE_VALUE,
First output the rationale for the score you have given and then the score value.
'''
BASE_PROMPT_EXAMPLES = '''
This aspect is aimed to maximize the utilization of the review comments for the authors. The primary purpose of the review is to help/guide authors in improving their drafts. Keep this in mind while evaluating the review point. Whenever you encounter a borderline case, think: “Will this review point help authors improve their draft?”. There is no correlation between the aspect score and the length of the review point.
Evaluate the review point based on the aspect description provided next.

{aspect}:
{aspect_description}

Generate a rationale and use it to output the score. 
{examples}

Review Point:
{review_point}
'''
ASPECTS_WITH_EXAMPLES = {
"actionability": 
'''
**Actionability**

**Definition:** Measures the level of actionability in a review point. We evaluate actionability based on two criteria:

1. **Explicit vs. Implicit**:
   - **Explicit:** Actions or suggestions that are direct or apparent. Authors can directly identify modifications they should apply to their draft. Clarification questions should be treated as explicit statements if they give a direct action.
   - **Implicit:** Actions that need to be inferred from the comment. This includes missing parts that need to be added. Authors can deduce what needs to be done after reading the comment.

2. **Concrete vs. Vague**:
   - **Concrete:** Once the action is identified, the authors know exactly what needs to be done and how to apply the action.
   - **Vague:** After identifying the action, the authors still don’t know how to carry out this action.

**Importance:** It’s more important for actions to be concrete so that authors know how to apply them. It's also preferred for actions to be stated directly rather than inferred.

**Actionability Scale (1-5):**

1. **1: Unactionable**
   - **Definition:** The comment lacks meaningful information to help authors improve the paper. Authors do not know what they should do after reading the comment.
   - **Examples:** 
     - The best result (F1) in both table 3 and table 4 (excluding PPDB features) is the 7th row. To my understanding, this variant does not use the proposed contextual representation; in fact, it uses the context2vec representation for the word type.
     - The idea of using positional encodings (PEs) for GNNs on molecular graph regression is not new.
     - The differences in results in Table 2 are very small, making the interpretation of results rather difficult. Furthermore, it is then unclear which proposed methods are really effective.

2. **2: Borderline Actionable**
   - **Definition:** The comment includes an implicitly stated action or an action that can be inferred. However, the action itself is vague and lacks detail on how to apply it.
   - **Examples:**
     - It is not clear if this trend holds across different model architectures.
     - I wonder what would happen if you used another dataset.
     - I am curious about the outcome of changing some of the hyperparameter values.

3. **3: Somewhat Actionable**
   - **Definition:** The comment explicitly states an action but is vague on how to execute it.
   - **Examples:**
     - You should address the lack of technical novelty in this paper.
     - I think some parameter values should be looked at.
     - Comparing your method to other previous related work is highly suggested.

4. **4: Mostly Actionable**
   - **Definition:** The comment implicitly states an action but concretely states how to implement the inferred action.
   - **Examples:**
     - There are some very relevant baselines like X and Y that other people have been comparing their results to.
     - Some items in Table 2 and Table 3 have spaces between accuracy and standard deviation, and some items don't, which affects beauty.
     - It is hard to understand under what conditions one should use SynTextBench over other metrics (e.g., MMLU / Big Bench for language generation).

5. **5: Highly Actionable**
   - **Definition:** The comment contains an explicit action and concrete details on how to implement it. Authors know exactly how to apply it.
   - **Examples:**
     - Given that this is a rare entity prediction problem, it would help to look at type-level accuracies, and analyze how the accuracies of the proposed models vary with frequencies of entities.
     - What will happen if you use the evaluation metric X instead of Y?
     - While the language has been improved, there are still some awkward phrases. I suggest the authors have the paper reviewed by a native English speaker.
''',

"grounding_specificity": '''

**Grounding Specificity**  

**Definition:** Measures how explicitly a review comment refers to a specific part of the paper and how clearly it identifies the issue with that part. This helps authors understand what needs revision and why. Grounding specificity has two key components:  

1. **Grounding:** How well the authors can identify the specific part of the paper being addressed.  
   - **Weak Grounding:** The author can make an educated guess but cannot precisely identify the referenced part.  
   - **Full Grounding:** The author can accurately pinpoint the section, table, figure, or unique aspect being addressed. This can be achieved through:  
     - Literal mentions of sections, tables, figures, etc.  
     - Mentions of unique elements of the paper.  
     - General comments that clearly imply the relevant parts without explicitly naming them.  

2. **Specificity:** How clearly the comment details what is wrong or missing in the referenced part. If external work is mentioned, it also evaluates whether specific examples are provided.  

**Importance:** It's more important for the comment to be grounded than to be specific.  

**Grounding Specificity Scale (1-5):** 

1. **Not Grounded**
   - **Definition**: This comment is not grounded at all. It does not identify a specific area in the paper. The comment is highly unspecific.
   - **Examples**:
     - While the language has been improved, there are still some awkward phrases. I suggest the authors have the paper reviewed by a native English speaker.
     - The paper is not very well-written, possibly hurriedly written, so it is not easy to read. A lot is left desired in presentation and formatting, especially in figures/tables.
     - The paper discusses a hot topic in the field now. However, one major drawback to this draft is that the analysis is poor.

2. **Weakly Grounded and Not Specific**
   - **Definition**: The authors cannot confidently determine which part the comment addresses. Further, the comment does not specify what needs to be addressed in this part.
   - **Examples**:
     - For many of the datasets tested, the improvement over other approaches or even the general adversarial approach is marginal.
     - Add more details while describing your method.
     - Something appears off when reading some figures’ captions.

3. **Weakly Grounded and Specific**
   - **Definition**: The authors cannot confidently determine which part the comment addresses. However, the comment clearly specifies what needs to be addressed in this part.
   - **Examples**:
     - Some figures need their captions to be more precise and to define all variables used in the figure.
     - For many of the datasets tested, the improvement over other approaches or even the general adversarial approach is marginal.
     - The notation used for the equations is not the same, and it varies between different equations.

4. **Fully Grounded and Under-Specific**
   - **Definition**: The comment explicitly mentions which part of the paper it addresses, or it should be obvious to the authors. However, this comment does not specify what needs to be addressed in this part.
   - **Examples**:
     - In Figure 7, the results and supplemental video results show that SurfGAN seems out of place.
     - I don’t like the formatting of L125.
     - The relationship between this work and the previous methods is not exposed.

5. **Fully Grounded and Specific**
   - **Definition**: The comment explicitly mentions which part of the paper it addresses, and it is obvious to the authors. The comment specifies what needs to be addressed in this part.
   - **Examples**:
     - In defining the UFE-layer as graph-based aggregation, the paper presents the motivation that an 'observation map' type approach as used in older single-pixel methods (like [4]) is sub-optimal, because it requires a resolution trade-off. This is not experimentally demonstrated—and should be, because it is central to the whole 'graph-based' premise.
     - I think it would be extremely useful if the authors could evaluate different types of self-supervised features, besides DINO. Other options include MoCo-v3 or MSN, which are all also based on the ViT architecture.
     - The differences in results in Table 2 are very small, which makes the interpretation of results rather difficult. Furthermore, it is then unclear which proposed methods are really effective.
''',

    
"verifiability_extraction" : """

**Claim Extraction**

**Objective:** Determine whether the given text contains claims or merely consists of factual statements.

### **Opinion & Claims**
A statement is considered a claim if it includes any of the following:

- **Subjective statements**, including opinions or disagreements with experimental choices.
- **Suggestions or requests for changes**, such as indicating that something should be removed, added, or discussed.
- **Judgments about sections of the paper**, such as stating that something is unclear, lacks detail, or is not well-written.
- **Deductions or inferred observations** beyond stating mere facts.
- **Any statement where evidence or justification is required** to support the claim.

### **Normal Statements**
A statement is considered a normal (non-claim) statement if it meets these criteria:

- It does **not** contain an opinion, claim, or suggestion but consists solely of factual, descriptive content.
- It **indicates the existence or absence of something** without suggesting changes.
- It makes **general statements about the paper** that do not express an opinion.
- It consists of **objective, factual statements** that do not require verification.
- It includes **requests for clarification or general questions**.
- It presents **logical statements or directly inferable information**.
- **Positive claims** (e.g., "The paper is well written") are considered normal statements as they do not help authors improve their work.

### **Scoring Criteria**
- **Yes:** If the text contains claims, opinions.
- **No:** If the text consists solely of normal statements.

1. **Yes**
   - **Definition:** The text contains claims, opinions, or suggestions that require justification or evidence.
   - **Examples:**
     - "The paper is well written, but the experiments are not convincing."
     - "The authors should consider adding more details to the methodology section."
     - "The results are not consistent with the claims made in the introduction."
2. **No**
   - **Definition:** The text consists solely of factual, descriptive content without expressing opinions or claims.
   - **Examples:**
     - "The paper discusses the impact of climate change on biodiversity."
     - "The authors present a new algorithm for image classification."
     - "The results show a significant improvement in accuracy compared to the baseline."
""",

"verifiability_verification" : '''
**Claim Verification**  

**Definition:** Measures how well a claim in the text is verified. Assess how well the reviewer justifies or proves the claim by providing logical reasoning, using common sense, or referencing external sources. Justifications or validations can appear before or after the claim. Claims may be explicitly stated or inferred.

**Verification Methods:**
- Logical reasoning supports the claim.
- Common sense knowledge in the field verifies the claim (e.g., referencing established practices or standards).
- External references substantiate the claim.

**Scoring Criteria (1-5 Scale):**

1. **1 - Unverifiable**
   - **Definition:** The comment contains a claim without any supporting evidence or justification.
   - **Examples:**
     - "The results fall behind previous work, and the reasons for this should be investigated."
     - "For many datasets tested, the improvement over other approaches or even the general adversarial approach is marginal."
     - "While the language has been improved, there are still some awkward phrases. I suggest the authors have the paper reviewed by a native English speaker."

2. **2 - Borderline Verifiable**
   - **Definition:** The comment provides some support for its claim, but the justification is vague, insufficient, or not fully articulated. Authors may struggle to follow the reasoning.
   - **Examples:**
     - "This method shouldn’t achieve good results. If I remember correctly, I have read a paper that tried to do the same thing, but it didn’t work for them."
     - "It is also unclear whether this momentum term could be a confounding factor in the comparison between PAL and SLS, as the vanilla version of SLS is just a stochastic line search applied to SGD without momentum."
     - "In the experiments, the transfer tasks are too artificial. At the pretraining stage, we train the models with examples from two classes ('bird' vs. 'frog') for CIFAR-10 and four classes (0, 1, 2, and 3) for MNIST."

3. **3 - Somewhat Verifiable**
   - **Definition:** The comment provides support for its claim, but key elements are missing, such as specific examples, detailed explanations, or supporting references. Authors must make a significant effort to follow the justification.
   - **Examples:**
     - "The evaluative framework appears somewhat limited in scope, with considerations restricted to merely three Question-Answering tasks and two language models."
     - "The nature of the contribution with respect to ECE_sweep is not clearly described in the text. Concretely, this amounts to a way to choose the number of bins using data."
     - "The approximation error is defined as the gap between the objective values, which is ambiguous unless one has seen the values in the table."

4. **4 - Mostly Verifiable**
   - **Definition:** The comment’s claim is sufficiently supported but has minor gaps. The reviewer could provide a more detailed explanation or reference.
   - **Examples:**
     - "The statistical analysis appears incorrect because the p-values reported for the t-tests do not align with standard thresholds for significance."
     - "The two used datasets are very related, where the input sequence is cocktail party speech, with one outputting the audio of each stream and the other producing the ASR output of each stream."
     - "As the paper states in the intro, double Q-learning was developed to address the overestimation problem of Q-learning. However, this cannot really be seen directly from the results in the paper. The explanation given suggests that double Q-learning resolves the overestimation problem by achieving a fast convergence rate."

5. **5 - Fully Verifiable**
   - **Definition:** The claim is thoroughly supported by explicit, sufficient, and robust evidence. This can be achieved through:
     - Clear and precise reasoning or explanation.
     - Specific and relevant references to external works or data.
     - Logical and unassailable common-sense arguments.
   - **Examples:**
     - "The landscape results in parameter space look very surprising because they have no assumptions on the generator and discriminator architecture except for sufficient representation. This is surprising because such global optimization results for neural networks usually require strong assumptions on the architecture."
     - "The first weakness of this work is that the wish list presented in the Introduction is broader than the actual techniques proposed. The key difference of this work lies in the dynamic prior, while previous work such as references [21] and [27] had already addressed the three properties mentioned."
     - "The paper’s main idea of mixing transfer-based and query-based attacks is not novel. Several papers [9, 19] have already explored this concept. This work simply combines the best transfer-based attack (TIMI) and one of the best L2 query-based attacks (SimBA) to create SimBA++, which is the main gain over previous approaches."

**Instructions:**
- Use the scoring scale to evaluate the verifiability of each review point.
- Focus on how well-supported the claims are rather than the length of the comment.
''',

"helpfulness" : '''
**Helpfulness**

**Definition:** Assign a subjective score to reflect the value of the review comment to the authors. Helpfulness is rated on a scale from 1 to 5, with the following definitions:

1. **1: Not Helpful at All**  
   - **Definition:** The comment fails to identify meaningful weaknesses or suggest improvements, leaving the authors with no actionable feedback.  
   - **Examples:**  
     - The core idea of this paper is very simple and straightforward. Though the authors justify that they are the first to do it, I am unsure whether this work might count as a novel enough contribution to the NeurIPS community.  
     - It might be good to add more comments.  
     - In the experiments, the transfer tasks are too artificial.  

2. **2: Barely Helpful**  
   - **Definition:** The comment identifies a weakness or improvement area but is vague, lacks clarity, or provides minimal guidance, making it only slightly beneficial for the authors.  
   - **Examples:**  
     - I wonder why learn the noisy data and clean data respectively in Algorithm 1, sample mini-batch d~D, \hat{d} ~ \hat{D}. Whether they can be fused for learning.  
     - For many of the datasets tested, the improvement over other approaches or even the general adversarial approach is marginal.  
     - Section 5: It is unclear why the superspreader model is more realistic or more challenging than the uniform corruption.  

3. **3: Somewhat Helpful**  
   - **Definition:** The comment identifies weaknesses or areas for improvement but is incomplete or lacks depth. While the authors gain some insights, the feedback does not fully address their needs for improving the draft.  
   - **Examples:**  
     - CRUCIAL: The evaluation is unclear. Were agents evaluated on held-out environments from the same task? Or on the N_env training environments? Either way seems fine, but it should be specified!  
     - What are the relative weights $m$ in 2.2.1? Are they hyperparameters? 2.2.2 seems to describe different schemes for using weights $m$ during inference, but aren't they needed during training? Are they fixed the same across the different setups?  
     - There is a gap between the proposed metric and method. Based on post-aggregation node similarity, they propose an aggregation similarity metric. However, the final 3-channel filterbank has nothing to do with the above metric.  

4. **4: Mostly Helpful**  
   - **Definition:** The comment provides clear and actionable feedback on weaknesses and areas for improvement, though it could be expanded or refined to be fully comprehensive and impactful.  
   - **Examples:**  
     - It is hard to find the formal definition of the proposed CRS model. It seems to be the equation after line 175, but the authors did not say it explicitly.  
     - The reviewer would appreciate some discussion on the possibility of accelerating the proposed algorithm and whether it’s optimal rate.  
     - The relationship between this work and previous methods is not exposed. Since the idea of data augmentation in feature space is not new, I expect several papers closely related to this work. However, I cannot see which cited papers are closely related to this work and how it differs from them.  

5. **5: Highly Helpful**  
   - **Definition:** The comment thoroughly identifies weaknesses and offers detailed, actionable, and constructive suggestions that empower the authors to significantly improve their draft.  
   - **Examples:**  
     - I think there’s a problem with EQN 2. I believe you should multiply by 5 instead of 2.  
     - The abstract should act like a compact summary of your draft. The way it is now, it needs extra summarization. Don’t include a lot of details about your proposed algorithm there.  
     - The paper also overstates some claims which should be removed. For example, on line 108, the paper says that "these algorithms often diverge, likely due to the failure of this assumption." Divergence in fitted Q-iteration could also be due to (for example) compounding errors of poor optimization of neural networks and their uncontrolled extrapolations.  
'''


}

################################################## FINETUNING PROMPTS ##################################################

PROMPT_HEADER = '''You are an expert in evaluating peer review comments with respect to different aspects. These aspects are aimed to maximize the utilization of the review comments for the authors. The primary purpose of the review is to help/guide authors in improving their drafts. Keep this in mind while evaluating the review point. Whenever you encounter a borderline case, think: “Will this review point help authors improve their draft?”. There is no correlation between the aspect score and the length of the review point.'''
SCORE_ONLY_PROMPT_TAIL = '''
Evaluate the review based on the given definitions of the aspect(s) above. Output only the score.
Review Point: {review_point}'''

SCORE_AND_RATIONALE_PROMPT_TAIL = '''
Evaluate the review based on the given definitions of the aspect(s) above. Generate a rationale and use it to output the score. Escape the double qoutes inside the rationale. 
Review Point: {review_point}'''


BASE_MODEL_SCORE_ONLY_PROMPT_TAIL = '''
Evaluate the review based on the given definitions of the aspect(s) above. Output only the score. The possbile values for scores are 1-5 and X.

Generate the output in JSON format with the following format:

   "actionability_label": "",
   "grounding_specificity_label": "",
   "verifiability_label": "",
   "helpfulness_label": ""


Review Point: {review_point}'''

BASE_MODEL_SCORE_AND_RATIONALE_PROMPT_TAIL ='''
Evaluate the review based on the given definitions of the aspect(s) above. Generate a rationale and use it to output the score. Escape the double qoutes inside the rationale. The possbile values for scores are 1-5 and X.

Generate the output in JSON format with the following format:

   "actionability_rationale": "",
   "actionability_label": "",
   "grounding_specificity_rationale": "",
   "grounding_specificity_label": "",
   "verifiability_rationale": "",
   "verifiability_label": "",
   "helpfulness_rationale": "",
   "helpfulness_label": ""


Review Point: {review_point}'''


INSTRUCTION_BASE_MODEL_SCORE_ONLY_PROMPT_TAIL = '''
###Instruction:
Evaluate the review based on the given definitions of the aspect(s) above. Output only the score. The possbile values for scores are 1-5 and X. 

Generate the output in JSON format with the following format:

   "actionability_label": "",
   "grounding_specificity_label": "",
   "verifiability_label": "",
   "helpfulness_label": ""


###Review Point:
{review_point}'''

INSTRUCTION_BASE_MODEL_SCORE_AND_RATIONALE_PROMPT_TAIL ='''
###Instruction:
Evaluate the review based on the given definitions of the aspect(s) above. Generate a rationale and use it to output the score. Escape the double qoutes inside the rationale. The possbile values for scores are 1-5 and X.

Generate the output in JSON format with the following format:

   "actionability_rationale": "",
   "actionability_label": "",
   "grounding_specificity_rationale": "",
   "grounding_specificity_label": "",
   "verifiability_rationale": "",
   "verifiability_label": "",
   "helpfulness_rationale": "",
   "helpfulness_label": ""


###Review Point:
{review_point}'''







INSTRUCTION_SCORE_ONLY_PROMPT_TAIL = '''
###Instruction:
Evaluate the review based on the given definitions of the aspect(s) above. Output only the score.

###Review Point:
{review_point}'''

INSTRUCTION_SCORE_AND_RATIONALE_PROMPT_TAIL = '''
###Instruction:
Evaluate the review based on the given definitions of the aspect(s) above. Generate a rationale and use it to output the score. Escape the double qoutes inside the rationale.

###Review Point:
{review_point}'''


ASPECTS_NO_EXAMPLES = {
"actionability": 
'''
**Actionability**

**Definition:** Measures the level of actionability in a review point. We evaluate actionability based on two criteria:

1. **Explicit vs. Implicit**:
   - **Explicit:** Actions or suggestions that are direct or apparent. Authors can directly identify modifications they should apply to their draft. Clarification questions should be treated as explicit statements if they give a direct action.
   - **Implicit:** Actions that need to be inferred from the comment. This includes missing parts that need to be added. Authors can deduce what needs to be done after reading the comment.

2. **Concrete vs. Vague**:
   - **Concrete:** Once the action is identified, the authors know exactly what needs to be done and how to apply the action.
   - **Vague:** After identifying the action, the authors still don’t know how to carry out this action.

**Importance:** It’s more important for actions to be concrete so that authors know how to apply them. It's also preferred for actions to be stated directly rather than inferred.

**Actionability Scale (1-5):**

1. **1: Unactionable**
   - **Definition:** The comment lacks meaningful information to help authors improve the paper. Authors do not know what they should do after reading the comment.

2. **2: Borderline Actionable**
   - **Definition:** The comment includes an implicitly stated action or an action that can be inferred. However, the action itself is vague and lacks detail on how to apply it.

3. **3: Somewhat Actionable**
   - **Definition:** The comment explicitly states an action but is vague on how to execute it.

4. **4: Mostly Actionable**
   - **Definition:** The comment implicitly states an action but concretely states how to implement the inferred action.

5. **5: Highly Actionable**
   - **Definition:** The comment contains an explicit action and concrete details on how to implement it. Authors know exactly how to apply it.
''',


"grounding_specificity": '''

**Grounding Specificity**  

**Definition:** Measures how explicitly a review comment refers to a specific part of the paper and how clearly it identifies the issue with that part. This helps authors understand what needs revision and why. Grounding specificity has two key components:  

1. **Grounding:** How well the authors can identify the specific part of the paper being addressed.  
   - **Weak Grounding:** The author can make an educated guess but cannot precisely identify the referenced part.  
   - **Full Grounding:** The author can accurately pinpoint the section, table, figure, or unique aspect being addressed. This can be achieved through:  
     - Literal mentions of sections, tables, figures, etc.  
     - Mentions of unique elements of the paper.  
     - General comments that clearly imply the relevant parts without explicitly naming them.  

2. **Specificity:** How clearly the comment details what is wrong or missing in the referenced part. If external work is mentioned, it also evaluates whether specific examples are provided.  

**Importance:** It's more important for the comment to be grounded than to be specific.  

**Grounding Specificity Scale (1-5):** 

1. **Not Grounded**
   - **Definition**: This comment is not grounded at all. It does not identify a specific area in the paper. The comment is highly unspecific.

2. **Weakly Grounded and Not Specific**
   - **Definition**: The authors cannot confidently determine which part the comment addresses. Further, the comment does not specify what needs to be addressed in this part.

3. **Weakly Grounded and Specific**
   - **Definition**: The authors cannot confidently determine which part the comment addresses. However, the comment clearly specifies what needs to be addressed in this part.

4. **Fully Grounded and Under-Specific**
   - **Definition**: The comment explicitly mentions which part of the paper it addresses, or it should be obvious to the authors. However, this comment does not specify what needs to be addressed in this part.

5. **Fully Grounded and Specific**
   - **Definition**: The comment explicitly mentions which part of the paper it addresses, and it is obvious to the authors. The comment specifies what needs to be addressed in this part.
''',
    
"verifiability":  
'''  
**Verifiability**  

**Definition:** Evaluates whether a review comment contains a claim and, if so, how well that claim is supported using logical reasoning, common knowledge, or external references.  

### **Step 1: Claim Extraction**  

**Objective:**  
Determine whether the given text contains a claim (i.e., an opinion, judgment, or suggestion) or consists solely of factual statements that require no verification.  

**Claim Definition:**  
A statement is considered a claim if it falls into one or more of the following categories:  
- **Subjective opinions or disagreements** (e.g., criticism of an experimental choice).  
- **Suggestions or requests for changes** (e.g., recommending removal, addition, or discussion).  
- **Judgments about the paper** (e.g., stating something is unclear, not well-written, or lacks detail).  
- **Deductions or inferred observations** that go beyond merely stating facts.  
- **Statements requiring justification** to be understood or accepted.  


**Normal Statements ("No Claim")**  
A statement is classified as "X" if it:  
- Describes facts without suggesting changes.  
- Makes general statements about the paper without an opinion.  
- Presents objective, verifiable facts that require no justification.  
- Asks for clarifications or general questions.  
- States logical statements or directly inferable information.  
- Makes positive claims (e.g., *“The paper is well-written”*), as these do not help improve the work.  

---  

### **Step 2: Verifiability Verification**  

**Objective:**  
Assess how well a claim is verified by examining the reasoning, common knowledge, or external references provided. The purpose is to ensure that the review comment helps the authors improve their work.  

**Verification Methods:**  
A claim is considered verifiable if supported by one or more of the following:  
- **Logical reasoning** – A clear explanation of why the claim is valid.  
- **Common knowledge** – Reference to well-accepted practices or standards.  
- **External references** – Citation of relevant literature, data, or sources.  

**Verifiability Scale (1–5 & X):**  

1. **1: Unverifiable**  
   - The comment contains a claim without any supporting evidence or justification.  
2. **2: Borderline Verifiable**  
   - Some support is provided, but it is vague, insufficient, or difficult to follow.  
3. **3: Somewhat Verifiable**  
   - The claim has some justification but lacks key elements (e.g., examples, references).  
4. **4: Mostly Verifiable**  
   - The claim is well-supported but has minor gaps in explanation or references.  
5. **5: Fully Verifiable**  
   - The claim is thoroughly supported by explicit, sufficient, and robust evidence, such as:  
     - Clear reasoning and precise explanations.  
     - Specific references to external works.  
     - Logical and unassailable common-sense arguments.  
6. **X: No Claim**  
- The comment contains only factual, descriptive statements without claims, opinions, or suggestions.  

---  

### **Instructions for Evaluation:**  
1. **Extract Claims:** Determine whether the review comment contains a claim or is a normal statement. If there is no claim, score it as "X"  
2. **Assess Verifiability:** If a claim exists, score it based on how well it is justified from 1 to 5.  
''',

"helpfulness" : '''
**Helpfulness**

**Definition:** Assign a subjective score to reflect the value of the review comment to the authors. Helpfulness is rated on a scale from 1 to 5, with the following definitions:

1. **1: Not Helpful at All**  
   - **Definition:** The comment fails to identify meaningful weaknesses or suggest improvements, leaving the authors with no actionable feedback.  

2. **2: Barely Helpful**  
   - **Definition:** The comment identifies a weakness or improvement area but is vague, lacks clarity, or provides minimal guidance, making it only slightly beneficial for the authors.  

3. **3: Somewhat Helpful**  
   - **Definition:** The comment identifies weaknesses or areas for improvement but is incomplete or lacks depth. While the authors gain some insights, the feedback does not fully address their needs for improving the draft.  

4. **4: Mostly Helpful**  
   - **Definition:** The comment provides clear and actionable feedback on weaknesses and areas for improvement, though it could be expanded or refined to be fully comprehensive and impactful.  

5. **5: Highly Helpful**  
   - **Definition:** The comment thoroughly identifies weaknesses and offers detailed, actionable, and constructive suggestions that empower the authors to significantly improve their draft.  
'''

}



# ASPECTS = {

# "actionability": 
# '''

# Measures the level of actionability in the review point. We evaluate actionability according to two points: 
# 1. Is the action stated directly, or does the author need to infer it? (Explicit vs. Implicit). 
# 2.  After identifying the action, do you know how to apply it, or the action is vague? (Concrete vs. Vague)
# - It’s more important for actions to be concrete so the authors know how to apply them. It’s also preferred that actions be stated directly rather than inferred.
# Definitions:
# Explicit:  Direct or apparent actions or suggestions. Authors can directly identify modifications that they should apply to their draft. Clarification questions should be treated as explicit statements if they give a direct action.
# Implicit:  Actions that can be deduced. This can be in the form of questions that need to be addressed or missing parts that need to be added. Actions are not stated directly, but the authors can infer what needs to be done after reading the comment.
# Concrete: After Identifying the action, the authors know exactly what needs to be done and how to apply the action.
# Vague: After Identifying the action, the authors still don’t know how to carry out this action.
#  Actionability is rated on a scale from 1-5, and we will now provide a definition for each.
# 1: Unactionable
# Definition: The comment lacks any meaningful information to help the authors to improve the paper. After reading the comment, the authors do not know what they should do.
# Examples: 
# The best result (F1) in both table 3 and table 4 (excluding PPDB features) is the 7th row. To my understanding, this variant does not use the proposed contextual representation; in fact, it uses the context2vec representation for the word type.
# The idea of using positional encodings (PEs) for GNNs on molecular graph regression is not new.
# The differences in results in Table 2 are very small that make the interpretation of results rather difficult. Furthermore, it is then unclear which proposed methods are really effective.
# 2: Borderline Actionable
# Definition: The comment includes an implicitly stated action, or the action can be inferred. Further, the action itself is vague and lacks detail on how to apply it.
# Examples:
# It is not clear if this trend holds across different model architectures.
# I wonder what would happen if you used another dataset.
# I am curious about the outcome of changing some of the hyperparameter values.
# 3: Somewhat Actionable
# Definition: The comment explicitly states an action but is vague on how to execute it.
# Examples:
# You should address the lack of technical novelty in this paper.
# I think some parameter values should be looked at.
# Comparing your method to other previous related work is highly suggested.
# 4: Mostly Actionable
# Definition: The comment implicitly states an action but concretely states how to implement the inferred action.
# Examples:
# There are some very relevant baselines like X and Y that other people have been comparing their results to.
# Some items in Table 2 and Table 3 have Spaces between accuracy and standard deviation, and some items don't, which affects beauty.
# It is hard to understand under what conditions one should use SynTextBench over other metrics (e.g., MMLU / Big Bench for language generation).
# 5: Highly Actionable
# Definition: The comment contains explicit action and concrete details on how to implement it. The authors know exactly how to apply it.
# Examples:
# Given that this is a rare entity prediction problem, it would help to look at type-level accuracies, and analyze how the accuracies of the proposed models vary with frequencies of entities.
# What will happen if you use the evaluation metric X instead of Y?
# While the language has been improved, there are still some awkward phrases. I suggest the authors have the paper reviewed by a native English speaker.


# ''',
    
# "actionability_no_examples": 
#   '''Measures the level of actionability in the review point. We evaluate actionability according to two points: 
#   1. Is the action stated directly, or does the author need to infer it? (Explicit vs. Implicit). 
#   2.  After identifying the action, do you know how to apply it, or the action is vague? (Concrete vs. Vague)
#   - It’s more important for actions to be concrete so the authors know how to apply them. It’s also preferred that actions be stated directly rather than inferred.
#   Definitions:
#   Explicit:  Direct or apparent actions or suggestions. Authors can directly identify modifications that they should apply to their draft. Clarification questions should be treated as explicit statements if they give a direct action.
#   Implicit:  Actions that can be deduced. This can be in the form of questions that need to be addressed or missing parts that need to be added. Actions are not stated directly, but the authors can infer what needs to be done after reading the comment.
#   Concrete: After Identifying the action, the authors know exactly what needs to be done and how to apply the action.
#   Vague: After Identifying the action, the authors still don’t know how to carry out this action.
#   Actionability is rated on a scale from 1-5, and we will now provide a definition for each.
#   1: Unactionable
#   Definition: The comment lacks any meaningful information to help the authors to improve the paper. After reading the comment, the authors do not know what they should do.
#   2: Borderline Actionable
#   Definition: The comment includes an implicitly stated action, or the action can be inferred. Further, the action itself is vague and lacks detail on how to apply it.
#   3: Somewhat Actionable
#   Definition: The comment explicitly states an action but is vague on how to execute it.
#   4: Mostly Actionable
#   Definition: The comment implicitly states an action but concretely states how to implement the inferred action.
#   5: Highly Actionable
#   Definition: The comment contains explicit action and concrete details on how to implement it. The authors know exactly how to apply it.
# ''',

# "grounding_specificity": '''This aspect measures how explicitly a review comment is based on a part of the paper. This is important so the authors know which part of their paper causes the issue and needs to be revised. Further, it measures how specifically the comment identifies what is the issue with this part of the paper. This aspect has two dimensions: (1) what part of the paper does this comment address, and (2) what is wrong with this part?
# Definitions:
# Grounding: Measures how well the authors can identify what is being addressed by the comment. (This can be no grounding, weak grounding, or full grounding). 
# * Weak grounding means that the author can’t precisely identify the part of the paper being addressed by the point, but they have some hint or guess about it. 
# * Full grounding means the authors can accurately identify which part is being addressed. This can be done by:
#    - Making literal mentions of sections, tables, figures, etc.
#      - The point discusses something unique to the paper that the authors can identify.
#      - General comments that do not need to mention specific parts of the paper, but the authors can easily infer which parts are addressed
# Specificity: Measures how much the reviewer detailed what is wrong/missing in this area. If the comment mentions some external work, it also measures whether it mentions specific examples. 
#   Grounding and specificity is rated on a scale from 1-5, and we will now provide a definition for each.
# 1: Not Grounded
# Definition: This comment is not grounded at all. It does not identify a specific area in the paper. The comment is highly unspecific.
# Examples
# While the language has been improved, there are still some awkward phrases. I suggest the authors have the paper reviewed by a native English speaker.
# The paper is not very well-written, possibly hurriedly written, so it is not easy to read. A lot is left desired in presentation and formatting, especially in figures/tables.
# The paper discusses a hot topic in the field now. However, one major drawback to this draft is that the analysis is poor.
# 2: Weakly grounded and not specific
# Definition: The authors can not confidently determine which part the comment addresses. Further, the comment does not specify what needs to be addressed in this part.
# Examples
# For many of the datasets tested, the improvement over other approaches or even the general adversarial approach is marginal. 
# Add more details while describing your method
# Something appears off when reading some figures’ captions.
# 3: Weakly grounded and specific
# Definition: The authors can not confidently determine which part the comment addresses. However, the comment clearly specifies what needs to be addressed in this part.
# Examples
# some figures need their captions to be more precise and to define all variables used in the figure.
# For many of the datasets tested, the improvement over other approaches or even the general adversarial approach is marginal.
# The notation used for the equations is not the same, and it varies between different equations.
# 4: Fully grounded and under-specific
# Definition: The comment explicitly mentions which part of that paper it addresses, or it should be obvious to the authors. However, this comment does not specify what needs to be addressed in this part.
# Examples
# In Figure 7, the results and supplemental video results show that SurfGAN seems out of place.
# I don’t like the formatting of L125.
# The relationship between this work and the previous methods is not exposed.
# 5: Fully grounded and specific
# Definition: The comment explicitly mentions which part of that paper it addresses, it is obvious to the authors. The comment specifies what needs to be addressed in this part.
# Examples
# In defining the UFE-layer as graph-based aggregation, the paper presents the motivation that an 'observation map' type approach as used in older single-pixel methods (like [4]) is sub-optimal, because it requires a resolution trade-off. This is not experimentally demonstrated—and should be, because it is central to the whole 'graph-based' premise.
# I think it would be extremely useful if the authors could evaluate different types of self-supervised features, besides DINO. Other options include MoCo-v3 or MSN, which are all also based on the ViT architecture.
# The differences in results in Table 2 are very small that make the interpretation of results rather difficult. Furthermore, it is then unclear which proposed methods are really effective.
# ''',

# "grounding_specificity_no_examples": '''This aspect measures how explicitly a review comment is based on a part of the paper. This is important so the authors know which part of their paper causes the issue and needs to be revised. Further, it measures how specifically the comment identifies what is the issue with this part of the paper. This aspect has two dimensions: (1) what part of the paper does this comment address, and (2) what is wrong with this part?
# Definitions:
# Grounding: Measures how well the authors can identify what is being addressed by the comment. (This can be no grounding, weak grounding, or full grounding). 
# * Weak grounding means that the author can’t precisely identify the part of the paper being addressed by the point, but they have some hint or guess about it. 
# * Full grounding means the authors can accurately identify which part is being addressed. This can be done by:
#    - Making literal mentions of sections, tables, figures, etc.
#      - The point discusses something unique to the paper that the authors can identify.
#      - General comments that do not need to mention specific parts of the paper, but the authors can easily infer which parts are addressed
# Specificity: Measures how much the reviewer detailed what is wrong/missing in this area. If the comment mentions some external work, it also measures whether it mentions specific examples. 
#   Grounding and specificity is rated on a scale from 1-5, and we will now provide a definition for each.
# 1: Not Grounded
# Definition: This comment is not grounded at all. It does not identify a specific area in the paper. The comment is highly unspecific.
# 2: Weakly grounded and not specific
# Definition: The authors can not confidently determine which part the comment addresses. Further, the comment does not specify what needs to be addressed in this part.
# 3: Weakly grounded and specific
# Definition: The authors can not confidently determine which part the comment addresses. However, the comment clearly specifies what needs to be addressed in this part.
# 4: Fully grounded and under-specific
# Definition: The comment explicitly mentions which part of that paper it addresses, or it should be obvious to the authors. However, this comment does not specify what needs to be addressed in this part.
# 5: Fully grounded and specific
# Definition: The comment explicitly mentions which part of that paper it addresses, it is obvious to the authors. The comment specifies what needs to be addressed in this part.
# ''',

    
# "verifiability": '''This aspect measures whether there is a claim (i.e. a subjective opinion) in the comment and how well it is verified. You need to detect first whether this review comment contains any claims. If there are any, evaluate how well the reviewer justifies or proves this claim by providing logical reasoning, using common sense or providing references. The claims' justification or validation can come before or after the claim. Claims don’t need to be stated directly; they can also be inferred.
# Definitions:
# Opinion & Claims
# Subjective statements. For example, an opinion or a stand that the reviewer takes (like a disagreement with an experimental choice).
# Any suggestions or requests for changes. For example, stating that something is worth discussing, should be removed, or added.
# Any comments judging some parts of the paper. For example, stating something is hard to read, not detailed enough, or comments about how good or bad some section of the paper is.
# Any deductions or inferred observations that go beyond just stating facts or results from the paper.
# Generally, any phrases where the reviewer should provide evidence to back up their claim and help the authors understand it better. This can be direct or indirect:
#     - Ex: “Important methods like X are not discussed”. We can infer that the reviewer suggests that method X should be discussed. Hence, the reviewer should state why this method should be discussed.
# Verification
# The claim is verified by providing logical reasoning.
# The claim is verified through common sense knowledge in the field. For example, referring to certain commonly used practices or standards.
# The claim is verified by providing external references.
# Normal Statements
# Normal statements should be given the label "No Claim".
# Indicating that something exists, or missing without indicating that it should be removed or included.
# General statements about the paper, that don’t include an opinion.
# Objective and factual statements that don’t need any kind of verification.
# Asking for clarifications and general questions.
# Logical statements, or things that can be inferred directly.
# We treat positive claims as normal sentences, as they are of little use to the authors to improve their paper.
#     - Example: This paper is well written, and the experimentation methods are well designed.
# Verifiability is rated on a scale from 1-5, and "No Claim" (for normal statements). We will now provide a definition for each.
# 1: Unverifiable
# Definition: The comment contains a claim without any supporting evidence or justification.
# Examples
# The results fall behind previous work, and the reasons for this should be investigated.
# For many of the datasets tested, the improvement over other approaches or even the general adversarial approach is marginal.
# While the language has been improved, there are still some awkward phrases. I suggest the authors have the paper reviewed by a native English speaker.
# 2: Borderline Verifiable
# Definition: The comment provides some support for its claim, but it is insufficient, vague, or not fully articulated. The authors will struggle to follow the justification.
# Examples
# This method shouldn’t achieve good results. If I remember correctly, I have read a paper that tried to do the same thing, but it didn’t work for them.
# It is also unclear whether this momentum term could be a confounding factor in the comparison between PAL and SLS, as the vanilla version of SLS is just a stochastic line search applied to SGD without momentum.
# In the experiments, the transfer tasks are too artificial. “At the pretraining stage, we train the models with examples from two classes (“bird" vs. “frog") for CIFAR-10 and four classes (0, 1, 2, and 3) for MNIST”.
# 3: Somewhat Verifiable
# Definition: The comment provides support for its claim, but one or more key elements are missing, such as specific examples, detailed explanations, or supporting references. It requires significant effort from the authors to follow the justification.
# Examples
# The evaluative framework appears somewhat limited in scope. With considerations restricted to merely three Question-Answering tasks and two language models.
# The nature of the contribution with respect to ECE_sweep is not clearly described in the text. Concretely, this amounts to a way to choose the number of bins using data
# The approximation error is defined as the gap between the objective values, which is somehow ambiguous unless one has seen the values in the table.
# 4: Mostly Verifiable
# Definition: The comment’s claim is sufficiently supported but has minor gaps. The reviewer could provide a more detailed explanation or reference to support their claims.
# Examples
# The statistical analysis appears incorrect because the p-values reported for the t-tests do not align with standard thresholds for significance.
# The two used datasets are very related, where the input sequence is cocktail party speech, with one outputting the audio of each stream and the other producing the ASR output of each stream
# As the paper states in the intro, double Q-learning was developed to address the overestimation problem of Q-learning. However, this cannot really be seen directly from the results in the paper. The explanation given in the paper suggests that double Q learning resolves the overestimation problem by achieving a fast convergence rate.
# 5: Fully Verifiable
# Definition: The claim is thoroughly supported by explicit, sufficient, and robust evidence. This can be done by:
# Clear and precise reasoning or explanation.
# References to external works/data, when applicable, are specific and relevant.
# Common-sense arguments are logically unassailable.
# Examples
# The landscape results in parameter space looks very surprising because it has no assumptions on the generator and discriminator architecture except for enough representation. This looks surprising to me because usually, this kind of global optimization result for neural networks needs strong assumptions on the architecture.
# The first weakness of this work is that the wish list presented in the Introduction is a bit wider than the real techniques proposed by this work because the key difference of this work lies in the dynamic prior. The three properties were mentioned and basically solved by previous work like reference [21] and [27].
# The paper’s main idea of mixing transfer-based and query-based attacks is not novel. There have already been multiple papers based on this idea [9, 19]. This paper simply proposes to combine the best transfer-based attack (TIMI) and one of the best L2 query-based attacks (SimBA), which results in SimBA++, which is the main gain over the previous approaches reported in the paper.
# No Claim
# Definition: The comment does not contain any claim, opinion, or suggestion and consists of only factual, descriptive statements that do not require any justification.
# Clear and precise reasoning or explanation.
# References to external works/data, when applicable, are specific and relevant.
# Common-sense arguments are logically unassailable.
# Examples
# How would this method perform empirically with multiple constraints?
# Entropy requires significant computation.
# This algorithm is slow, as it relies on an O(N²) algorithm.''',

# "verifiability_no_examples": '''This aspect measures whether there is a claim (i.e. a subjective opinion) in the comment and how well it is verified. You need to detect first whether this review comment contains any claims. If there are any, evaluate how well the reviewer justifies or proves this claim by providing logical reasoning, using common sense or providing references. The claims' justification or validation can come before or after the claim. Claims don’t need to be stated directly; they can also be inferred.
# Definitions:
# Opinion & Claims
# Subjective statements. For example, an opinion or a stand that the reviewer takes (like a disagreement with an experimental choice).
# Any suggestions or requests for changes. For example, stating that something is worth discussing, should be removed, or added.
# Any comments judging some parts of the paper. For example, stating something is hard to read, not detailed enough, or comments about how good or bad some section of the paper is.
# Any deductions or inferred observations that go beyond just stating facts or results from the paper.
# Generally, any phrases where the reviewer should provide evidence to back up their claim and help the authors understand it better. This can be direct or indirect:
#     - Ex: “Important methods like X are not discussed”. We can infer that the reviewer suggests that method X should be discussed. Hence, the reviewer should state why this method should be discussed.
# Verification
# The claim is verified by providing logical reasoning.
# The claim is verified through common sense knowledge in the field. For example, referring to certain commonly used practices or standards.
# The claim is verified by providing external references.
# Normal Statements
# Normal statements should be given the label "No Claim".
# Indicating that something exists, or missing without indicating that it should be removed or included.
# General statements about the paper, that don’t include an opinion.
# Objective and factual statements that don’t need any kind of verification.
# Asking for clarifications and general questions.
# Logical statements, or things that can be inferred directly.
# We treat positive claims as normal sentences, as they are of little use to the authors to improve their paper.
#     - Example: This paper is well written, and the experimentation methods are well designed.
# Verifiability is rated on a scale from 1-5, and "No Claim" (for normal statements). We will now provide a definition for each.
# 1: Unverifiable
# Definition: The comment contains a claim without any supporting evidence or justification.
# 2: Borderline Verifiable
# Definition: The comment provides some support for its claim, but it is insufficient, vague, or not fully articulated. The authors will struggle to follow the justification.
# 3: Somewhat Verifiable
# Definition: The comment provides support for its claim, but one or more key elements are missing, such as specific examples, detailed explanations, or supporting references. It requires significant effort from the authors to follow the justification.
# 4: Mostly Verifiable
# Definition: The comment’s claim is sufficiently supported but has minor gaps. The reviewer could provide a more detailed explanation or reference to support their claims.
# 5: Fully Verifiable
# Definition: The claim is thoroughly supported by explicit, sufficient, and robust evidence. This can be done by:
# Clear and precise reasoning or explanation.
# References to external works/data, when applicable, are specific and relevant.
# Common-sense arguments are logically unassailable.
# X: No Claim
# Definition: The comment does not contain any claim, opinion, or suggestion and consists of only factual, descriptive statements that do not require any justification.
# Clear and precise reasoning or explanation.
# References to external works/data, when applicable, are specific and relevant.
# Common-sense arguments are logically unassailable.
# ''',


# "helpfulness": 
# '''Assign a subjective score to reflect the value of the review comment to the authors.

# Helpfulness is rated on a scale from 1-5, and we will now provide a definition for each.

# 1. The comment is not helpful at all
# Definition: The comment fails to identify any meaningful weaknesses or suggest improvements, leaving the authors with no actionable feedback.
# Examples
# The core idea of this paper is very simple and straightforward. Though the authors justify that they are the first to do it, I am unsure whether this work might count as a novel enough contribution to the NeurIPS community.
# It might be good to add more comments.
# In the experiments, the transfer tasks are too artificial.
# 2. The comment is barely helpful
# Definition: The comment identifies a weakness or improvement area but is vague, lacks clarity, or provides minimal guidance, making it only slightly beneficial for the authors.
# Examples
#  I wonder why learn the noisy data and clean data respectively in Algorithm 1, sample mini-batch d~D, \hat{d} ~ \hat{D}. Whether they can be fused for learning.
#  For many of the datasets tested the improvement over other approaches or even the general adversarial approach is marginal.
#  Section 5: It is unclear why the superspreader model is more realistic or more challenging than the uniform corruption.
# 3. The comment is somewhat helpful
# Definition: The comment identifies weaknesses or areas for improvement but is incomplete or lacks depth. While the authors can gain some insights, the feedback does not fully address their needs for improving the draft.
# Examples
# CRUCIAL: The evaluation is unclear. Were agents evaluated on held-out environments from the same task? Or on the N_env training environments? Either way seems fine, but it should be specified!
# What are the relative weights $m$ in 2.2.1? are they hyperparams? 2.2.2 seems to describe different schemes for using weights $m$ during inference, but aren't they needed during training? Are they fixed the same across the different setups?
# There is a gap between the proposed metric and method. Based on post-aggregation node similarity, they propose an aggregation similarity metric. However, the final 3-channel filterbank has nothing to do with the above metric.
# 4. The comment is mostly helpful
# Definition: The comment provides clear and actionable feedback on weaknesses and areas for improvement, though it could be expanded or refined to be fully comprehensive and impactful.
# Examples
#  It is hard to find the formal definition of the proposed CRS model. It seems to be the equation after line 175, but the authors did not say it explicitly.
# The reviewer would appreciate some discussion on the possibility of accelerating the proposed algorithm and whether it's optimal rate.
# The relationship between this work and the previous methods are not exposed. Since the idea of data augmentation in feature space is not new, I expect several papers closely related to this work. However, I cannot see which cited papers are closely related to this work and how it differs from them.
# 5. The comment is highly helpful
# Definition: The comment thoroughly identifies weaknesses and offers detailed, actionable, and constructive suggestions that empower the authors to significantly improve their draft.
# Examples
# I think there’s a problem with EQN 2. I believe you should multiply by 5 instead of 2.
# The abstract should act like a compact summary of your draft. The way it is not, it needs extra extra summarization. Don’t include a lot of details about your proposed algorithm there.
# The paper also overstates some claims which should be removed. For example on line 108 the paper says that "these algorithms often diverge, likely due to the failure of this assumption". Divergence in fitted Q-iteration could also be due to (for example) compounding errors of poor optimization of neural networks and their uncontrolled extrapolations.
# ''',
# "professional_tone": '''Definition: This aspect evaluates the level of formality, respect, and clarity in the language used in the peer review. It ensures the feedback is delivered constructively, without being overly critical or dismissive, and maintains a respectful and impartial tone throughout. A professional tone avoids personal attacks, uses appropriate language, and focuses on the work rather than the individual. It encourages collaboration and improvement by fostering a positive and productive dialogue. 
# Based on our preliminary results, most of the review points are addressed in a professional tone. But, we want to identify ones that are not, if there are any.
# If the comment is so negative, it doesn’t mean it’s not professional.
# Examples of comments conducted in an unprofessional/inappropriate tone:
# This is a poorly written paper, and I don't understand how it got submitted in the first place.
# The authors clearly have no idea what they are doing. This entire draft is a waste of time.
# I can't believe the authors didn't realize how bad this method is.
# ''',


# "helpfulness_no_examples": 
# '''Assign a subjective score to reflect the value of the review comment to the authors.

# Helpfulness is rated on a scale from 1-5, and we will now provide a definition for each.

# 1. The comment is not helpful at all
# Definition: The comment fails to identify any meaningful weaknesses or suggest improvements, leaving the authors with no actionable feedback.
# 2. The comment is barely helpful
# Definition: The comment identifies a weakness or improvement area but is vague, lacks clarity, or provides minimal guidance, making it only slightly beneficial for the authors.
# 3. The comment is somewhat helpful
# Definition: The comment identifies weaknesses or areas for improvement but is incomplete or lacks depth. While the authors can gain some insights, the feedback does not fully address their needs for improving the draft.
# 4. The comment is mostly helpful
# Definition: The comment provides clear and actionable feedback on weaknesses and areas for improvement, though it could be expanded or refined to be fully comprehensive and impactful.
# 5. The comment is highly helpful
# Definition: The comment thoroughly identifies weaknesses and offers detailed, actionable, and constructive suggestions that empower the authors to significantly improve their draft.
# ''',
# }
