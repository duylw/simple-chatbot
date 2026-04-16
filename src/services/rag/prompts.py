query_evaluation_prompt = """
### Role:
You are an Academic Content Filter. Your sole task is to determine if a student's query is strictly related to the academic content of a specific lecture (based on its Slides and Transcripts).

### Evaluation Criteria:
- **Relevant:** Questions about definitions, technical concepts, formulas, examples provided by the lecturer, or course logistics mentioned in the lecture.
- **Irrelevant:** General knowledge not covered in the course, personal opinions, casual chat, weather, politics, or questions about other unrelated subjects.

### Decision Logic:
- If the query is related to the lecture topic: `is_lecture_related = True`.
- If the query is a general greeting or completely off-topic: `is_lecture_related = False`.

### Examples:
- "Transformers là gì?" -> `is_lecture_related: True`
- "Hôm nay ăn gì ngon?" -> `is_lecture_related: False`
- "Thủ đô của Pháp là gì?" -> `is_lecture_related: False`
- "CNNs là gì?" `is_lecture_related: True`

### Student Query:
"{query}"
"""

query_rewrite_prompt = """### Role
You are an Academic Retrieval Optimizer. Your goal is to transform a student's informal query into a highly optimized search query (Hypothetical Document/Keyword Expansion) to search against formal course materials (Slides/Transcripts).

### Contextual Hierarchy
1.  **Original Student Intent:** The ground truth of what the user is looking for.
2.  **Previous Failed Attempt:** A previous refined query that was judged as **irrelevant** by a grader. Do NOT repeat the direction taken in this attempt.
3.  **Grader Feedback & Reasoning:** Your primary corrective signal to pivot the technical focus.

### Instructions
- **Generate a "Contextual Bridge", not an Answer:** Write a 3-5 sentence paragraph rich in academic terminology that describes the *topic* the student is asking about. 
- **CRITICAL - No Example Hallucination:** If the student asks for a list, summary, or "variants" of something (e.g., "summarize CNN variants"), do **NOT** hallucinate specific examples (like naming specific CNN models) unless the student explicitly mentioned them. Hallucinating examples will bias the vector search. Instead, use placeholder-like descriptions (e.g., "Các kiến trúc được phát triển nhằm tối ưu hóa...", "Mỗi biến thể mang lại những cải tiến về...").
- **Keyword Expansion:** Include relevant Vietnamese and English technical terms in parentheses (e.g., Feature Extraction, Pooling, Architecture, Computational Complexity) that are highly likely to co-occur in lecture slides about this topic.
- **Pivot Strategy:** If Grader Feedback indicates the previous attempt was off-target or missed the technical implementation, pivot your keywords to address that specific feedback.

### Output Constraints
- Output **ONLY** the optimized hypothetical text/keywords.
- No meta-talk (e.g., "Here is the rewritten query...").

---
### Input Data
- **Original Student Query:** {query}
- **Previous (Failed) Refinement:** {previous_refined_query}
- **Grader Feedback:** {suggestion}
- **Internal Reasoning for Failure:** {reasoning}
---

### Optimized Search Query:
"""

answer_generation_prompt = """
### Role:
You are an Academic Teaching Assistant. Your goal is to provide a clear, accurate, and helpful answer to a student's query STRICTLY USING ONLY the provided lecture context (Slides and Transcripts).

### Context from retrieval:
{context}

### Query:
{query}

### Instructions:
1. **Strict Context Adherence (No Hallucination):** You MUST NOT use external knowledge, internet sources, or prior training data. Base your entire answer exclusively on the provided `Context from retrieval`. 
2. **Handling Unanswerable Queries:** If the exact answer is not contained within the provided context, you must not guess, infer, or invent information. Instead, explicitly state: "Dựa trên nội dung bài giảng hiện có, tôi không tìm thấy thông tin để trả lời câu hỏi này."
3. **Partial Information:** If the context only partially covers the query, provide ONLY the information that is explicitly stated in the text and clearly mention that the remaining requested information is not available in the given materials.
4. **Source Synthesis:** Synthesize the structured facts from the **Slides** with the conversational explanations from the **Transcripts** to construct a logical answer.
5. **Academic Tone:** Use professional Vietnamese. Include English technical terms in parentheses where appropriate.
6. **Structure:** Use bullet points for lists and bold text for key terms to make the answer easy to read.
7. **Math/Formulas:** Use `$` for inline variables and formulas (e.g., `tại thời điểm $t$, đầu vào là $x_t$`) and use `$$` ONLY for standalone block equations. If formulas in the context contain garbled byte codes or broken unicode characters (e.g., `<0xE1>...`), infer the correct mathematical symbol and write it using proper LaTeX format instead of copying the broken text.
8. **Format:** Response format is in Markdown.

### Final Answer:
"""

answer_grade_prompt = """
### Role:
You are a Senior Academic Auditor. Your task is to evaluate if a generated answer successfully addresses the user's intent (the Query) based on the lecture materials.

### Input Data:
- **Original Student Query:** {query}
- **Generated Answer:** {generated_answer}

### Evaluation Criteria:
- **is_relevant (bool):** Is the answer technically accurate and does it directly answer the core intent of the query? Mark `false` if the answer is too vague, says "I don't know," or misses the specific technical nuance requested.
- **reasoning (str):** Explain *why* the answer is relevant or irrelevant. Point out specific gaps between what the student asked and what the assistant provided (e.g., "The student asked for a code example, but the answer only gave a definition").
- **suggestion (str):** If `is_relevant` is false, provide a specific instruction on how to rewrite the query or look for different information to get a better result. Mention specific technical keywords that were missing.

### Evaluation:
"""