PROMPT_HEADER = '''You are an expert in evaluating peer review comments with respect to different aspects. These aspects are aimed to maximize the utilization of the review comments for the authors. The primary purpose of the review is to help/guide authors in improving their drafts. Keep this in mind while evaluating the review point. Whenever you encounter a borderline case, think: “Will this review point help authors improve their draft?”. There is no correlation between the aspect score and the length of the review point.'''
SCORE_ONLY_PROMPT_TAIL = '''
Evaluate the review based on the given definitions of the aspect(s) above. Output only the score.
Review Point: {review_point}'''
SCORE_AND_RATIONALE_PROMPT_TAIL = '''
Evaluate the review based on the given definitions of the aspect(s) above. Output the score and rationale for the score.
Review Point: {review_point}'''


ASPECT_DEFINITIONS = {

"actionability": 
'''
**Actionability**

**Definition:** Measures the level of actionability in a review point. We evaluate actionability based on two criteria:

1. **Explicit vs. Implicit**:
   - **Explicit:** Actions or suggestions that are direct or apparent. Authors can directly identify modifications they should apply to their draft. Clarification questions should be treated as explicit statements if they give a direct action.
   - **Implicit:** Actions that need to be inferred from the comment. This includes questions that need to be addressed or missing parts that need to be added. Authors can deduce what needs to be done after reading the comment.

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

# Grounding Specificity

This aspect measures how explicitly a review comment is based on a part of the paper. This is important so the authors know which part of their paper causes the issue and needs revision. It also assesses how specifically the comment identifies what is the issue with this part of the paper. This aspect has two dimensions: 
1. What part of the paper does this comment address?
2. What is wrong with this part?

## Definitions

### Grounding:
Measures how well the authors can identify what is being addressed by the comment. This can be categorized as no grounding, weak grounding, or full grounding.
- **Weak grounding**: The author can’t precisely identify the part of the paper being addressed by the point, but they have some hint or guess about it.
- **Full grounding**: The authors can accurately identify which part is being addressed. This can be done by:
  - Making literal mentions of sections, tables, figures, etc.
  - Discussing something unique to the paper that the authors can identify.
  - Giving general comments that do not need to mention specific parts of the paper, but the authors can easily infer which parts are addressed.

### Specificity:
Measures how much the reviewer detailed what is wrong/missing in this area. If the comment mentions some external work, it also measures whether it mentions specific examples.

### Grounding and Specificity Scale (1-5):

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
- **Judgments about the paper** (e.g., stating something is unclear, well-written, or lacks detail).  
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




"helpfulness": """
**"Helpfulness:"**

Assign a subjective score to reflect the value of the review comment to the authors. Helpfulness is rated on a scale from 1 to 5, and we will now provide a definition for each.

**1. The comment is not helpful at all**  
*Definition*: The comment fails to identify any meaningful weaknesses or suggest improvements, leaving the authors with no actionable feedback.  
**2. The comment is barely helpful**  
*Definition*: The comment identifies a weakness or improvement area but is vague, lacks clarity, or provides minimal guidance, making it only slightly beneficial for the authors.  
**3. The comment is somewhat helpful**  
*Definition*: The comment identifies weaknesses or areas for improvement but is incomplete or lacks depth. While the authors can gain some insights, the feedback does not fully address their needs for improving the draft.  
**4. The comment is mostly helpful**  
*Definition*: The comment provides clear and actionable feedback on weaknesses and areas for improvement, though it could be expanded or refined to be fully comprehensive and impactful.  
**5. The comment is highly helpful**  
*Definition*: The comment thoroughly identifies weaknesses and offers detailed, actionable, and constructive suggestions that empower the authors to significantly improve their draft.  
"""

}