

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
You are an expert critic for scientific paper reviews. The following is a part of a review.  You need to evaluate the review concerning a certain aspect. 

 The aspect is: 
 {aspect}: {aspect_description}
 Review point: {review}
Give a score of 0 if you think this aspect is not relative to this review. 
Give a score of 1 if you think this aspect is relative to this review, and the aspect is followed or respected in the review. 
Give a score of -1 if you think this aspect is relative to this review, but the aspect is ignored or not respected in the review. 

reason step by step whether this aspect is relative to this review, and then if the aspect is followed or not, then finally give the corresponding score, using the format:
 Score: [SCORE]

''',

'Actionability' : '''refers to the extent to which the feedback provided in the review includes specific, clear, and practical suggestions that the author can use to improve their paper. Actionable comments go beyond identifying issues; they offer concrete steps or recommendations that guide the author in addressing those issues.
Actionability enhances the usefulness of a review by providing the author with a clear path for revision.
A score of 1, means that this review should be actionable, and it directs, hints, or suggests steps or edits that should be made.
A score of -1, means that this review should be actionable, but it lacks in giving a direction or hints of what needs to be done.
A score of 0, means that it's not relative for this review to be actionable in the first place.

 Example of Actionability in a Review

Reviews with a score of 1:
- Try to show the value of the dataset -- to show the use of tertiary claim classes and visual claim classes can actually help in fake news detection performance. Also, it will be interesting to see whether using the classification results on visual claim classes can help the classification performance of tertiary claim classes.
- Increase the number of annotators for each data to avoid conflicts. If possible, annotate more data.


Review with a score of -1:
- Related to the point above, the related work section is severely lacking.
- Despite the motivation and presented statistics, the paper could transmit a better overview of the proposed challenge. As it is, I find it hard to evaluate its merits. 

Review with a score of 0:
- The paper doesn't show how tertiary claim classes and visual claim classes can help in fake news detection. So it leads to the question, why do we care about the proposed label classes in claim detection?
- And while sampling self-constrains, do you sample from the word buckets based on the sentence itself or do you consider ALL possible words (from all sentences) in the bucket?
- During training you sample 0-3 reference tokens as constraints, since the BPE is used, does it mean that it might be the case that only part of the word is considered as a constraint? Or do you handle such a situation in some way?
''',




'Constructiveness or Politeness' : '''
refers to the manner in which feedback is delivered, focusing on being helpful, respectful, and encouraging. Constructive feedback aims to aid the author in improving their work by offering positive reinforcement and suggestions for enhancement, rather than simply criticizing. Polite feedback maintains a respectful and professional tone, ensuring that the author feels motivated and supported in their revision process.
A score of 1, means that this review should be constructive and positive.
A score of -1, means that this review should be constructive and positive, but instead, the tone of the review is negative or aggressive.
A score of 0, means that it's not relative for this review to be constructive or positive in the first place.

Example of Constructiveness or Politeness in a Review

Reviews with a score of 1:
- Including generation from baseline systems for the same examples would help illustrate the differences better.
- The introduction could be strengthened by providing a clearer context for your research question. It might be helpful to start with a brief overview of the existing literature to highlight the gap your study aims to fill.


Review with a score of -1:

- This introduction is terrible and makes no sense.
- The paper is poorly written with grammatical errors and in a very stilted style. This paper would benefit from a thorough rewrite with the assistance of a native speaker.
- This table is rife with problems. Here are some examples:


Review with a score of 0:
- During training you sample 0-3 reference tokens as constraints, since the BPE is used, does it mean that it might be the case that only part of the word is considered as a constraint?
- The size of the explanatory dataset (used in one of the experiments) is 25, which is very small.
- The paper doesn't show how tertiary claim classes and visual claim classes can help in fake news detection.  So it leads to the question, why do we care about the proposed label classes in claim detection? 

''',

'Credibility or Verifiability': '''

refers to the extent to which the feedback provided in the review is supported by evidence, references, or logical reasoning that can be independently verified. Credible comments are based on established knowledge, empirical data, or clearly articulated reasoning, making them reliable and trustworthy. Verifiable feedback ensures that the reviewer’s claims and suggestions can be checked against external sources or through logical validation.

A score of 1, means that this review imposes a certain claim, and then supports it using internal references from the paper, or citing external work. 
A score of -1, means that this review imposes a certain claim without providing any evidence for it. 
A score of 0, means that it's not relative for this review to cite or reference any work. This can be for general statements or observations in the review.

Example of Credibility or Verifiability in a Review

Reviews with a score of -1
- The method here looks promising, but the results fall behind previous work, It will be so useful to investigate the reason behind this.
- The usual size for datasets in this field is in order of millions, so having a dataset set with this size doesn't give a clear signal for the results. 

Reviews with a score of 1:
- The statistical analysis appears to be incorrect because the p-values reported for the t-tests do not match the standard thresholds for significance. According to the guidelines in 'Statistics for Biologists' (Smith et al., 2020), the p-values should be below 0.05 for the results to be considered statistically significant.
- The paper does not compare with any lexically constrained decoding methods (see references below). Moreover, the keyword control mechanism proposed in this method has been introduced in CTRLSum paper (He et al, 2020) for keywork-controlled summarization.


Reviews with a score of 0:
- Including more models in the evaluation would provide a more robust comparison and solidify the study’s claims.
- The dataset for wordplay type detection consists of only 25 samples. This small sample size can significantly impact accuracy, as even one incorrect answer can alter the accuracy by 4%. Increasing the dataset size to a few hundred samples would provide a more reliable basis for conclusions.
- The dataset has conflicts in annotated data. Why not add more annotators?



''',

'Specificity' : '''in a review refers to the extent to which the feedback is detailed and directly relevant to the content of the draft. Specific comments identify particular sections, statements, or elements within the paper and provide clear, focused suggestions or critiques. Specificity ensures that the feedback is tailored to the draft in question and helps the author understand exactly what needs to be addressed.
A score of 1, means that this review is addressing a certain point or design decision in the paper, so it should be specific, and the review manages to mention some parts of the paper specifically.
A score of -1, means that this review should be specific, but the review doesn't mention specific parts or methods in the paper.
A score of 0, means that it's not relative for this review to be specific as it may describe general concepts or abstract critics.

Example of Specificity in a Review

Reviews with a score of -1:
- The paper is hard to follow and certain sections need to be made clearer. It was tough to understand exactly where the exemplar/target syntax was obtained for different settings, and how these differed between training and inference for each of those settings.
- The paper should include examples of generated paraphrases using all control options studies.
Reviews with a score of 1:
- In section 5.2.1, ''the reason for models to perform better with definition part'' → is presented as a hypothesis. The study would benefit from additional experiments to validate this hypothesis.
- "In the introduction, the statement 'various factors influence the outcome' is too vague. It would be more impactful to specify which factors you are referring to.
- The keyword control mechanism proposed in this method has been introduced in CTRLSum paper (He et al, 2020) for keyword controlled summarization.

Reviews with a score of 0:
- The paper doesn't show how tertiary claim classes and visual claim classes can help in fake news detection. So it leads to the question, why do we care about the proposed label classes in claim detection?
- Increase the number of annotators for each data to avoid conflicts. If possible, annotate more data

'''

}