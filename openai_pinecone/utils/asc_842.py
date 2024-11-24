from langchain_core.prompts import PromptTemplate

asc_842_prompt_template = PromptTemplate.from_template(
    """
    Your task is to read the provided Source Document and respond to the questions provided in Inquiries.
    Inquiries is an array of objects, each representing a question and its parameters.
    Each object in the Inquiries array contains the following keys:
    reader_question: the value notes the question you have to answer;
    id: the value is the question's id;
    restrictions: the value notes any requirements for or restrictions on the response you can provide for that specific question.
    user_response: the value is optional. The reader_question will explain how this value is to be utilized.
    type: the value represents the expected format of the <<answer>> you provide.
    Expected format of the <<answer>>:
    Your response must be provided as a JSON object in the following format - each question should be identified by its question id:
    {{"<<the question's id>>": {{"question": "<<reader_question>>", "answer": "<<answer>>", "meta": "<<meta>>"}} }}

    - If type='string', then provide the <<answer>> in the form of an array of strings in which each paragraph is a separate string in the array. If the question asks to provide the response in the form of a list,
    then the first string in the array should be an introduction of the list and the remaining strings should be the list. 
    - If type='boolean', then the <<answer>> can only be "0" or "1". Reply with "0" if false, "1" if true, or nothing if unknown or there is an error. 
    - If type='number', then only provide the <<answer>> as a number or nothing if unknown or there is an error. 
    - If type='date', then only provide the <<answer>> as date formatted as YYYY-MM-DD or nothing if unknown or there is an error.
    
    If required in the response, refer to the Source Document as the "source document."
    In <<meta>>, where possible, you should first quote the sentence where you found this answer.
    And then, if possible, you should explain how you determined your answer. But this is not required. If nothing is provided in <<meta>>, then leave it blank.
    
    If your response does not meet the following requirements then fix it:
    - Your answer must come from the provided Source Document. Remove any information from outside the Source Document.
    - You must respond to all questions in the Inquiries array. 
    - Return nothing in the <<answer>> if there is any type of error, you have no response or you do not know the answer to the question.
    - Confirm that all the restrictions associated with a question are enforced.
    - Do not mention, disclose or reference the restrictions associated with a question anywhere in the response you provide.

    Inquiries: {questions}
    Source Document: {context}
    """
)


def asc_842_prompt_template_structured(template_string: str | None = None):
    template = (
        template_string
        if template_string
        else """Your task is to read the provided Source Document and respond to the questions provided in Inquiries.
    Inquiries is an array of objects, each representing a question and its parameters.
    Each object in the Inquiries array contains the following keys:

    - id: the value is the question's id;
    - reader_question: the value notes the question you have to answer;
    - restrictions: the value notes any requirements for or restrictions on the response you can provide for that specific question.
    - user_response: the value is optional. The reader_question will explain how this value is to be utilized.
    - type: the value represents the expected format of the <<answer>> you provide.

    - If type='string', then provide the <<answer>> in the "answer_string" key in the form of an array of strings in which each paragraph is a separate string in the array. If the question asks to provide the response in the form of a list,
    then the first string in the array should be an introduction of the list and the remaining strings should be the list. 
    - If type='boolean', then the <<answer>> in the "answer_boolean" key. It can only be "0" or "1". Reply with "0" if false, "1" if true.
    - If type='date', then provide the <<answer>> in the "answer_date" key as a date formatted as YYYY-MM-DD.
    
    In <<meta>>, where possible, you should first quote the sentence where you found this answer. And, if possible, you should explain how you determined your answer.
    
    If your response does not meet the following requirements then fix it:
    - Your answer must come from the provided Source Document. Remove any information from outside the Source Document.
    - You must respond to all questions in the Inquiries array. 
    - Return nothing in the <<answer>> if there is any type of error, you have no response or you do not know the answer to the question.
    - Confirm that all the restrictions associated with a question are enforced.
    - Do not mention, disclose or reference the restrictions associated with a question anywhere in the response you provide.

    Inquiries: {questions}
    Source Document: {context}
    """
    )
    return PromptTemplate.from_template(template)
