

PROMPTS = {
    'base_prompt' : '''You are an advanced language model specialized in scientific and academic content. Your task is to evaluate a review of a scientific paper, considering the aspects that are most important to the author of the paper. Evaluate the review based on the following aspect. Just output the score on a scale from 1 to 5. With 1 being the worst and 5 being the best.
    The aspect is: 
    {aspect}: {aspect_description}
    Review: {review}

    output only the score. 
    ''',

    'Actionability' : '''refers to the extent to which the feedback provided in the review includes specific, clear, and practical suggestions that the author can use to improve their paper. Actionable comments go beyond identifying issues; they offer concrete steps or recommendations that guide the author on how to address those issues. Actionability enhances the usefulness of a review by providing the author with a clear path for revision.

    Example of Actionability in a Review

    Non-Actionable Comment:
    "The methodology section is unclear."

    Actionable Comment:
    "The methodology section would benefit from a more detailed explanation of the sample selection process.
''',
'Constructiveness or Politeness' : '''refers to the manner in which feedback is delivered, focusing on being helpful, respectful, and encouraging. Constructive feedback aims to aid the author in improving their work by offering positive reinforcement and suggestions for enhancement, rather than simply criticizing. Polite feedback maintains a respectful and professional tone, ensuring that the author feels motivated and supported in their revision process.

Example of Constructiveness or Politeness in a Review

Non-Constructive Comment:
"This introduction is terrible and makes no sense."

Constructive and Polite Comment:
"The introduction could be strengthened by providing a clearer context for your research question. It might be helpful to start with a brief overview of the existing literature to highlight the gap your study aims to fill. ''',

'Credibility or Verifiability': ''' refers to the extent to which the feedback provided in the review is supported by evidence, references,z or logical reasoning that can be independently verified. Credible comments are based on established knowledge, empirical data, or clearly articulated reasoning, making them reliable and trustworthy. Verifiable feedback ensures that the reviewer’s claims and suggestions can be checked against external sources or through logical validation.

Example of Credibility or Verifiability in a Review

Non-Credible Comment:
"The statistical analysis seems incorrect."

Credible Comment:
"The statistical analysis appears to be incorrect because the p-values reported for the t-tests do not match the standard thresholds for significance. According to the guidelines in 'Statistics for Biologists' (Smith et al., 2020), the p-values should be below 0.05 for the results to be considered statistically significant.
''',

'Specificity' : '''in a review refers to the extent to which the feedback is detailed and directly relevant to the content of the draft. Specific comments identify particular sections, statements, or elements within the paper and provide clear, focused suggestions or critiques. Specificity ensures that the feedback is tailored to the draft in question and helps the author understand exactly what needs to be addressed.

Example of Specificity in a Review

Non-Specific Comment:
"The introduction is too vague."

Specific Comment:
"In the introduction, the statement 'various factors influence the outcome' is too vague. It would be more impactful to specify which factors you are referring to.

'''

}