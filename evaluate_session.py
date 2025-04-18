# evaluate_session_v2.py

from openai import OpenAI

def evaluate_counselling_session(api_key, conversation_history):
    """
    Evaluates a counselling session transcript OR continues a conversation with the AI supervisor.

    Arguments:
    - api_key: The OpenAI API key
    - conversation_history: A list of dicts (role: 'user' or 'assistant', content: str)
    
    Returns:
    - A string containing the AI-generated evaluation of the session
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    system_prompt = """
    You are a clinical supervisor providing feedback on a counselling session transcript
    between a trainee (user) and a simulated client (assistant).
    Assess the trainee's counselling techniques based on:
    1. Empathy
    2. Communication Clarity
    3. Active Listening
    4. Appropriateness of Responses
    5. Overall Effectiveness

    - Provide a concise analysis, with specific conversation details as examples when appropriate, highlight strengths and weaknesses, 
    and suggest 2-3 actionable improvements for the trainee. 
    - If the trainee asks follow-up questions, answer them concisely with relevant examples.
    - Maintain a supportive and constructive tone.
    """.strip()

    # Reconstruct transcript
    transcript_text = "\n".join([
        f"{'Client' if msg['role'] == 'assistant' else 'Trainee'}: {msg['content']}"
        for msg in conversation_history
    ])

    # OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": transcript_text}
        ],
        max_tokens=2000,
        temperature=0.7
    )

    return response.choices[0].message.content
