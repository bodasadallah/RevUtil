

PROMPTS = {
    'base_prompt' : '''You are an advanced language model specialized in scientific and academic content. Your task is to evaluate a review of a scientific paper, considering the aspects that are most important to the author of the paper. Evaluate the review based on the following aspect. Just output the score on a scale from 1 to 5. With 1 being the worst and 5 being the best.
 The aspect is: 
    {aspect}: {aspect_description}
 Review: {review}

 Output only the score. 
 ''',

    'binary_score_prompt' : '''You are an advanced language model specialized in scientific and academic content. The following is part of the review of a scientific paper. We want to make sure that some aspect is met by this point. Evaluate the point based on the following aspect. Just output the score of 0 or 1, with 0 being the review didn't meet the aspect and 1 being the review met the aspect.
 The aspect is: 
    {aspect}: {aspect_description}
 Review point: {review}

 Output only the score.
''',


'system_prompt' : '''

You are fair expert critic of scientific paper reviews. Your task is providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. ''',
'ternary_score_prompt' : '''

###Task Description:
A review point, an aspect, and an evaluation criteria are given.
1. Write detailed feedback that assesses the quality of the review point strictly based on the given score rubric, not evaluating it in general.
2.  This review point is addressed to the authors of the paper, who have full knowledge of their paper content.
3. After writing feedback, write a score that is an integer number that is either -1, 0, or 1. You should refer to the score criteria.
4. The output format should look as follows: "(write a feedback for criteria). score: [SCORE] (an integer number that is either -1, 0, or 1)"
5. Please do not generate any other opening, closing, and explanations.

###Review point: {review}

###Aspect: {aspect}
###Evaluation Criteria: {aspect_description}

###Feedback:


''',

'actionability' : '''

This aspect measures the actionability of the review point. This review point should be assigned to one of three types. The first type is for review points that contain one or more direct, explicit, or apparent actions or suggestions. The second type is for comments that contain implicit actions that can be deduced. This can be in the form of questions that need to be addressed, or missing parts that need to be added. The third type is for review points that are not actionable at all. These comments leave the authors without any clear or infrared direction of what needs to be done. This review point is addressed to the authors of the paper, who have full knowledge of the paper. The review point does not have to contain much detail about how to apply, or execute these actions, but rather mentioning or hinting actions or suggestions. We assume the authors know how to add or discuss the reviewer's comments.

Scoring Criteria:
A score of 1: The feedback gives one or more direct and explicit actionable comments that the authors should make. This can also happen in the form of a suggestion, that the authors can address in their paper. 

A score of -1: The feedback does not give any hint, or direct actionable comments. After reading the feedback, the authors still do not know or can not deduce what needs to be done. 

A score of 0:  The feedback gives implicit actionable comments, that the authors can infer after reading the feedback. This can be in the form of questions that need to be addressed.  This feedback does not have any explicit actionable comments.


 Examples of Actionability in review point, corresponding to a SCORE of 1:
* Try to show the value of the dataset -- to show the use of tertiary claim classes and visual claim classes can actually help in fake news detection performance. Also, it will be interesting to see whether using the classification results on visual claim classes can help the classification performance of tertiary claim classes.
* Increase the number of annotators for each data to avoid conflicts. If possible, annotate more data.
* Section 4 is very tersely written (maybe due to limitations in space) and could have benefitted from a slower development for an easier read. Why would the field be interested in such method?

Examples of lacking Actionability in review point, corresponding to a SCORE of -1:
* The related work section is severely lacking.
* Compared with DyTox, the proposed method obtains a little performance improvement with more parameters and variance.
*  Several works have shown that entity embeddings provide effective features in cross-lingual tasks, and the contribution of this paper is incremental.

Examples of Actionability in review point, corresponding to a SCORE of 0:
* While the theorems require the activation function to be smooth, in the numerical experiments ReLU is used. Maybe it is more illustrative if a smooth activation function can be used. Is there a reason for not using tanh, sigmoid, etc?
* Temporal logic as such is often useful when considering infinite traces. Signal temporal logic is quite useful in the case of finite-time traces when dealing with continuous time systems. Neither of them are under consideration here, and I think this is the biggest draw back. 
*  During training, you sample 0-3 reference tokens as constraints, since the BPE is used, does it mean that it might be the case that only part of the word is considered as a constraint? Or do you handle such a situation in some way?



'''

,
'politeness' : '''

measures the tone that the review point is delivered in. Is it aggressive, neutral, or constructive? 

Scoring Criteria:
A score of 1:  means that the tone of the review point is helpful, or encouraging.
A score of -1:  means that the tone of the review point is rude, disrespectful, or aggressive.
A score of 0:  means that the tone of the review point is neutral, factual, or passive. This is also the default score if the review point does not fit well with the other two scores. 

Examples of Politeness in review point, corresponding to a SCORE of 1:
* The bounds seem really hard to compute. It would have been nice had the paper provided a (perhaps more conservative) yet simpler expression alongside the current one.
* The introduction could be strengthened by providing a clearer context for your research question. It might be helpful to start with a brief overview of the existing literature to highlight the gap your study aims to fill.
* Building on the proposed method, it will be a big strength if you can show that this algorithm can generalize to different domains as well. 

Examples of lacking Politeness in review point, corresponding to a SCORE of -1:
* This introduction is terrible and makes no sense.
* No empirical evaluation whatsoever is provided, there is no comparison (except for on an abstract level) with other methods. In its current form this paper is not suitable for a publication at NeurIPS.
* This poor quality of experiments and writing doesn’t live up to the level of this conference. Authors should do better to at least be on the minimum standards. 

Examples of Politeness in review point, corresponding to a SCORE of 0:
* During training, you sample 0-3 reference tokens as constraints, since the BPE is used, does it mean that it might be the case that only part of the word is considered as a constraint?
* Although applying GCN on FVQA is interesting, the technical novelty of this paper is limited.
* It feels Section 6 doesn't belong in this paper, and that the extra space could be used for extending the other parts of the paper. 


''',
'verifiability': '''

Your task is to decide if this review point makes any sort of claim or judgment. Or, it just mentions some facts about the paper. If you decide that the review point has a claim, then you should decide if the reviewer has tried to justify or verify it. We are judging whether the reviewer tried to validate the claim using any method, rather than evaluating the method they used to verify the claim.

Sentences can be considered as claims if:
* Stating that something is worth discussing, removed, or added.
* They are subjective statements that need to be justified. 
* They can be explicit, or easy to infer.
* They show an opinion or a stand that the reviewer takes. 
* Any subjective statements should be considered as claims. 
* Any suggestions, or requests for changes should be considered as claims. 
* Comments about how good or bad some section of the paper is should be considered claims.
 * Any deductions or inferred observations that go beyond just stating facts or results from the paper should be considered as claims. 

To justify claims:
* The reviewer mentioned any relevant information to the claim. 
* They try to explain why they think their claim is true. This can be an example of why they believe their claim is true. 
* Any attempt to validate a claim is valid. The reviewer does not have to provide substantial proof. Any attempt to address the claim is acceptable. 

Normal sentences are: 
* General statements about the paper. 
* They generally are objective and factual and don’t need any kind of verification. 
* They describe an objective fact, that anyone can reach if they read the paper. 
* Asking for clarifications, and general questions are considered normal sentences. 
* Stating the absence or existence of something in the paper is a normal sentence. 


Scoring Criteria:
A score of 1: This review point contains one or more claims, and the author provided any kind of explanation of verification. 
A score of -1: This review point contains one or more claims, and the author did not mention anything to strengthen their claim. 
A score of 0: This review point does not contain any claims.


Examples of  Verifiability in the review point, corresponding to a SCORE of 1:
* The statistical analysis appears to be incorrect because the p-values reported for the t-tests do not match the standard thresholds for significance. According to the guidelines in 'Statistics for Biologists' (Smith et al., 2020), the p-values should be below 0.05 for the results to be considered statistically significant.
* Universal Texture Synthesis: The paper claims universal texture synthesis. However, it has been demonstrated to work with regular texture patterns alone. There is a large variety of non-stationary textures (Zhou et al. [61]) that I think this work cannot address because of the fundamental regularity assumption or repetitive or stationary texture.
* Section 2.2 mentions examples of DBpedia properties that were used as features. Do the authors mean that all the properties have been used or there is a subset? If the latter please list them. In the authors' response, the authors explain in more detail this point and I strongly believe that it is crucial to list all the features in detail in the final version for clarity and replicability of the paper.

Examples of lacking  Verifiability in the review point,  corresponding to a SCORE of -1:
* The method here looks promising, but the results fall behind previous work, It will be so useful to investigate the reason behind this.
*  using MapNet as the one example of a fully learned system. I appreciate that this choice might be motivated by the fact that the agent uses a mapping representation compatible with the others leveraged in the paper. At the same time, I think it is fair to say that the conclusions the authors derive from the experiments are not warranted. The choice of the p values is not quite accurate, as it doesn’t contribute much to the overall score. 
* Inclusion of the Jacobean in the output transformation renders the ML solution a better characterization of the Bayesian solution.

Examples of  Verifiability in the review point, corresponding to a SCORE of 0:
*  What is the relationship between the CDF and percentile rank in this case? Is there a way to express one with the other?
*  I'm missing a discussion of why values for p different from 2 are not interesting to consider.
* Only one constraint is used for the experiments. How would FOCOPS perform empirically when there are multiple constraints?


''',
'specificity' : '''

This aspect measures how specific the review point is. The review point can be classified into one of three categories.
The first is for review points that address specific sections, ideas, and algorithms in the paper. After reading, the authors can identify the parts of the paper that are being addressed by the review point.
The second type hints to general ideas, or methods, but after reading the review point, the author has a guess about which part is being addressed, but they are not fully confident. 
The third type is for general comments, that doesn’t address any parts of the paper. These review points leave the author with no clue about which part is being addressed by the reviewer. 

Scoring Criteria:
A score of 1: After reading the review point, the authors can be confident about which part of the paper is being addressed by the review point. This can also happen if the review point addresses a general theme or idea that is spread across many paper sections. 
A score of -1: After reading the review point, the authors do not know which parts of the paper are being addressed by the review point.
A score of 0:  After reading the review point, the authors can guess which parts are being addressed in the review point, but are not fully confident. 

Examples of Specificity in review point, corresponding to a SCORE of 1:
* In section 5.2.1, ''the reason for models to perform better with definition part'' → is presented as a hypothesis. The study would benefit from additional experiments to validate this hypothesis.
* The rationale for dividing the sequence of intermediate results into 11 sets is not provided.
* The keyword control mechanism proposed in this method has been introduced in the CTRLSum paper (He et al, 2020) for keyword-controlled summarization.

Examples of lacking Specificity in review point, corresponding to a SCORE of -1:
* The paper is hard to follow and certain sections need to be made clearer. It was tough to understand exactly where the exemplar/target syntax was obtained for different settings, and how these differed between training and inference for each of those settings.
* The writing should be improved. Some points in the paper are unclear to me.
* Some sections of this draft need to be rewritten, and some figure has a big size, and need to be reformatted again.

Examples of Specificity in review point,  corresponding to a SCORE of 0:
* The paper doesn't show how tertiary claim classes and visual claim classes can help in fake news detection. So it leads to the question, why do we care about the proposed label classes in claim detection?
* The method is mainly heuristic, there is no guarantee for the performance of the new method. Accordingly, the quality of the method can be judged only empirically on the datasets that have been tested.
*  Have you experimented with having _both_ BN and VCL? Post-rebuttal I will stick to my rating. This is good work and I thank the authors for clarifying my questions in the rebuttal.


''',

'PROM_ternary_score_prompt' : '''
You are an expert critic of scientific paper reviews. Given a feedback paragraph of a review, an aspect and an aspect description, your task is to evaluate the feedback regarding the provided aspect.

Give a score of 1 if you think this aspect is relevant to this review, and the aspect is followed or respected in the review. 
Give a score of -1 if you think this aspect is relevant to this review, but the aspect is ignored or not respected in the review. 
Give a score of 0 if you think this aspect is not relevant to this review. 

ASPECT: {aspect}
ASPECT DESCRIPTION: {aspect_description}
REVIEW: {review}

Reason step by step whether this aspect is relative to this review point, and then if the aspect is followed or not. At the end, give the corresponding score. 
The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number that is either -1, 0, or 1)"
''',

'PROM_SCORE_RUBRIC_TEMPLATE_TERNARY':"""
[{criteria}]
Score -1: {score_negone_description}
Score 0: {score_zero_description}
Score 1: {score_one_description}
"""

,

'PROMETHEUS_PROMTS': """###Task Description:
An instruction, a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer number that is either -1, 0, or 1. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number that is either -1, 0, or 1)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """,


'PROM_INSTRUCTION' :  '''

As an expert critic of scientific paper reviews, you are tasked with evaluating the feedback paragraph of a review based on the provided aspect.
Give a score of 1 if you think this aspect is relevant to this review, and the aspect is followed or respected in the review. 
Give a score of -1 if you think this aspect is relevant to this review, but the aspect is ignored or not respected in the review. 
Give a score of 0 if you think this aspect is not relevant to this review.
'''


}






ASPECTS_CRITERIA = {
    
"actionability" : '''
[How actionable is the feedback? Does it provide clear and practical suggestions that the author can use to improve their paper? These actions can be stated directly, or they can be inferred or hinted on in the feedback.]

Score -1: This feedback is related to the actionability aspect, but it lacks in giving a specific direction or hints of what needs to be done. Or, this feedback gives general comments that don't give a specific action to be taken by the authors.
Score 0: the nature and content of the feedback does not require it to be actionable, i.e. the aspect is not relevant for the feedback.
Score 1: This feedback is related to the actionability aspect, and the feedback directs, hints, or suggests specific steps or edits that should be made.
''',
"politeness" : '''
   [{criteria}]
Score -1: {score_negone_description}
Score 0: {score_zero_description}
Score 1: {score_one_description}
''',
"verifiability" : '''
   [{criteria}]
Score -1: {score_negone_description}
Score 0: {score_zero_description}
Score 1: {score_one_description}
''',
"specificity" : '''
   [{criteria}]
Score -1: {score_negone_description}
Score 0: {score_zero_description}
Score 1: {score_one_description}
'''



}
    
