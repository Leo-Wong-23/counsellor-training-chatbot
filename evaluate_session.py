# evaluate_session.py

from openai import OpenAI  # Ensure you use the latest OpenAI client

def evaluate_counseling_session(api_key, conversation_history):
    """
    Evaluates a counseling session transcript using an OpenAI LLM call.
    
    Arguments:
    - api_key: The OpenAI API key
    - conversation_history: A list of dicts (role: 'user' or 'assistant', content: str)
    
    Returns:
    - A string containing the AI-generated evaluation of the session
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Define the system prompt for evaluation
    system_prompt = """
    You are a clinical supervisor providing feedback on a counseling session transcript
    between a trainee (user) and a simulated client (assistant).
    Assess the trainee's counseling techniques based on:
    1. Empathy
    2. Communication Clarity
    3. Active Listening
    4. Appropriateness of Responses
    5. Overall Effectiveness

    Provide a concise analysis, highlight strengths and weaknesses, 
    and suggest 2-3 actionable improvements for the trainee.
    """.strip()

    # Reconstruct transcript
    transcript_text = "\n".join([
        f"{'Client' if msg['role'] == 'assistant' else 'Trainee'}: {msg['content']}"
        for msg in conversation_history
    ])

    # Updated OpenAI API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_text}
        ],
        max_tokens=300,
        temperature=0.7
    )

    return response.choices[0].message.content
