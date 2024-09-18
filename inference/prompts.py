

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


'ternary_score_prompt' : '''
You are an expert critic of scientific paper reviews. Given a feedback paragraph of a review, an aspect and an aspect description, your task is to evaluate the feedback regarding the provided aspect.

Give a score of 1 if you think this aspect is relevant to this review, and the aspect is followed or respected in the review. 
Give a score of -1 if you think this aspect is relevant to this review, but the aspect is ignored or not respected in the review. 
Give a score of 0 if you think this aspect is not relevant to this review. 

ASPECT: {aspect}
ASPECT DESCRIPTION: {aspect_description}
REVIEW: {review}

Reason step by step whether this aspect is relative to this review point, and then if the aspect is followed or not. At the end, give the corresponding score, using the format:

The aspect score is: [SCORE]
''',

'actionability' : '''Refers to the extent to which the feedback provided in the review includes specific, clear, and practical suggestions that the author can use to improve their paper. Actionable comments go beyond identifying issues; they offer concrete steps or recommendations that guide the author in addressing those issues. Note that these actions can be stated directly, or they can be inferred or hinted on in the feedback.
Actionability enhances the usefulness of a review by providing the author with a clear path for revision.

A score of 1, means that this feedback should be actionable, and it directs, hints, or suggests steps or edits that should be made.
A score of -1, means that this feedback should be actionable, but it lacks in giving a direction or hints of what needs to be done.
A score of 0, means that the nature and content of the feedback does not require it to be actionable, i.e. the aspect is not relevant for the feedback.

 Examples of Actionability in feedback, corresponding to a SCORE of 1:
- Try to show the value of the dataset -- to show the use of tertiary claim classes and visual claim classes can actually help in fake news detection performance. Also, it will be interesting to see whether using the classification results on visual claim classes can help the classification performance of tertiary claim classes.
- Increase the number of annotators for each data to avoid conflicts. If possible, annotate more data.
- No comparison to competing methods only an ablation of the proposed exploration strategy with greedy improvement and approximate Thompson Sampling.

Examples of lacking Actionability in feedback, corresponding to a SCORE of -1:
- Related to the point above, the related work section is severely lacking.
- Some of the baselines in the experiments seem weak or missing.
-  While this model is one of its kind in this area, the language model’s scope is too narrow to have a wide range of applications. Other similar BERT-based models (e.g. BioBERT, SciBERT) has a wider coverage. The authors should strengthen the claim of the importance of this model by discussing important problems in this area and how this language model will be essential in addressing a wide variety of problems in this domain.

Examples of feedback that are not relevant for Actionability, corresponding to a SCORE of 0:
- Figure 1, what are the labels for horizontal and vertical axis?
- And while sampling self-constrains, do you sample from the word buckets based on the sentence itself or do you consider ALL possible words (from all sentences) in the bucket?
- During training, you sample 0-3 reference tokens as constraints, since the BPE is used, does it mean that it might be the case that only part of the word is considered as a constraint? Or do you handle such a situation in some way?
''',
'politeness' : '''Refers to the manner in which feedback is delivered, focusing on being helpful, respectful, and encouraging. Polite feedback maintains a respectful and professional tone, ensuring that the author feels motivated and supported in their revision process. All feedback points are relevant to this aspect, however, the level of respect and constructiveness varies.

A score of 1, means that this feedback is polite, positive, or trying to encourage the authors.
A score of -1, means that the tone of the feedback is negative or aggressive.
A score of 0, means that the tone of the feedback is neutral.

Example of Politeness in feedback, corresponding to a SCORE of 1:
- Including generation from baseline systems for the same examples would help illustrate the differences better.
- The introduction could be strengthened by providing a clearer context for your research question. It might be helpful to start with a brief overview of the existing literature to highlight the gap your study aims to fill.
- Building on the proposed method, it will be a big strength if you can show that this algorithm can generalize to different domains as well. 

Example of lacking Politeness in feedback, corresponding to a SCORE of -1:
- This introduction is terrible and makes no sense.
- The paper is poorly written with grammatical errors and in a very stilted style. This paper would benefit from a thorough rewrite with the assistance of a native speaker.
 - Some of the baselines in the experiments seem weak or missing.

Examples of feedback that are not relevant for Politeness, corresponding to a SCORE of 0:
- During training, you sample 0-3 reference tokens as constraints, since the BPE is used, does it mean that it might be the case that only part of the word is considered as a constraint?
- The size of the explanatory dataset (used in one of the experiments) is 25, which is very small.
- The paper doesn't show how tertiary claim classes and visual claim classes can help in fake news detection.  So it leads to the question, why do we care about the proposed label classes in claim detection? 
''',
'verifiability': '''
Refers to the extent to which the feedback provided in the review is supported by references to external sources, references to parts of the paper, or logical reasoning that can be independently verified. Verifiable feedback ensures that the reviewer’s claims and suggestions can be checked against external sources or through logical validation. This can happen as well, if the claim is followed by an explaination or clarification part that try to justify the claim.

A score of 1, means that this feedback imposes a certain claim and then supports it using internal references from the paper, citing external work, or it’s a logical claim. 
A score of -1, means that this feedback imposes a certain claim without providing any evidence for it. 
A score of 0, means that it's not relative for this feedback to cite or reference any work. This can be for general statements or observations in the review.

Example of Verifiability in feedback, corresponding to a SCORE of 1:
- The statistical analysis appears to be incorrect because the p-values reported for the t-tests do not match the standard thresholds for significance. According to the guidelines in 'Statistics for Biologists' (Smith et al., 2020), the p-values should be below 0.05 for the results to be considered statistically significant.
- The paper does not compare with any lexically constrained decoding methods (see references below). Moreover, the keyword control mechanism proposed in this method has been introduced in CTRLSum paper (He et al, 2020) for keyword-controlled summarization.
- The number of examples in the fine-tuning dataset (see section. 5) needs to be put in perspective with the pre-training dataset size ( section 3). 

Example of lacking Verifiability in feedback, corresponding to a SCORE of -1:
- The method here looks promising, but the results fall behind previous work, It will be so useful to investigate the reason behind this.
- The usual size for datasets in this field is in order of millions, so having a dataset set with this size doesn't give a clear signal for the results. 
- Inclusion of the Jacobean in the output transformation renders the ML solution a better characterisation of the Bayesian solution.

Examples of feedback that are not relevant for Verifiability in feedback, corresponding to a SCORE of 0:
-  What is the relationship between the CDF and percentile rank in this case? is there a way to express one with the other?
- I'm missing a discussion why values for p different from 2 are not interesting to consider.
- Only one constraint is used for the experiments. How would FOCOPS perform empirically when there are multiple constraints?
''',
'specificity' : '''Specificity in a review refers to the extent to which the feedback is detailed and directly relevant to the content of the draft. Specific comments make it clear which sections, statements, or elements in the paper are addressed. Specificity ensures that the feedback is tailored to the content of the paper, and not a general comment for any scientific document.

A score of 1, means that this feedback addresses a certain point or design decision in the paper, so it should be easy to identify which part of the document is being discussed in the feedback.
A score of -1, means that this feedback should be specific, but the review doesn't mention specific parts or methods in the paper, and it’s hard to identify the parts that are being addressed in the document.
A score of 0, means that it's not relative for this feedback to be specific as it may describe general concepts or abstract critics.

Example of Specificity in feedback, corresponding to a SCORE of 1:
- In section 5.2.1, ''the reason for models to perform better with definition part'' → is presented as a hypothesis. The study would benefit from additional experiments to validate this hypothesis.
- "In the introduction, the statement 'various factors influence the outcome' is too vague. It would be more impactful to specify which factors you are referring to.
- The keyword control mechanism proposed in this method has been introduced in the CTRLSum paper (He et al, 2020) for keyword-controlled summarization.

Example of lacking Specificity in feedback, corresponding to a SCORE of -1:
- The paper is hard to follow and certain sections need to be made clearer. It was tough to understand exactly where the exemplar/target syntax was obtained for different settings, and how these differed between training and inference for each of those settings.
- The paper should include examples of generated paraphrases using all control options studies.
- Some sections of this draft need to be rewritten, and some figure has a big size, and need to be reformatted again.

Examples of feedback that are not relevant for Specificity in feedback, corresponding to a SCORE of 0:
- The paper doesn't show how tertiary claim classes and visual claim classes can help in fake news detection. So it leads to the question, why do we care about the proposed label classes in claim detection?
- Increase the number of annotators for each data to avoid conflicts. If possible, annotate more data.
- I think there is much more relevant previous work that is similar to this paper. The authors need to do a more in depth literature review and include it in their paper.
'''

}