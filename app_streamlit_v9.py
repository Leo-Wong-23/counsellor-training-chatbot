import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from evaluate_session_v2 import evaluate_counselling_session
from datetime import datetime

st.set_page_config(page_title="Counsellor Training Chatbot", layout="wide")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
PASSWORD = os.getenv("PASSWORD")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Password protection
def check_password():
    """Simple password authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        # App description
        st.title("Counsellor Training Chatbot")
        st.markdown("""
        **Hi, this is Leo, a psychology and cognitive neuroscience postgraduate with backgrounds in AI and education. Welcome to this Counsellor Training Chatbot that I built!**

        This is a proof-of-concept application to explore how AI can bring service innovations and optimisations to the field of psychology. This app is designed to support psychology trainees in developing effective counselling skills through simulated counsellor-client interactions.<br><br>


        **Key Features:**
        - Engage in real-time conversations with realistic client personas experiencing diverse psychological challenges (in the Chat Session tab).

        - Receive personalized feedback to enhance counselling techniques, with interactive discussions for deeper understanding (in the Evaluation tab).<br><br>


        **Features in Development:**
        - Traceback input modification, which will be quite useful from a training perspective, allowing exploration of how the session could have gone differently
        - User voice input and dynamic model voice output, like a sending and receiving voice messages<br><br>


        ***Safety & Privacy Statement:***
        This app is currently in development and serves as a demonstration tool only—it is not intended for real-world counselling or professional use. 
        No chat history or personal data are stored beyond the active session, they are erased once you close or refresh the page.
        
        That said, a download transcript option is available in the evaluation tab. If you'd like to share feedback or discuss potential improvements, feel free to reach out!<br><br>

        Please enter the password to begin (you can find it in my CV).
        """, unsafe_allow_html=True)

        with st.form(key="password_form"):
            entered_password = st.text_input("Enter Password:", type="password")
            submit_button = st.form_submit_button("Submit")

        if submit_button or entered_password:
            if entered_password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Try again.")

    if not st.session_state.authenticated:
        st.stop()  # Prevents the rest of the app from loading

check_password()  # Enforce login before loading the app

def load_personas():
    """Loads the personas from the JSON file and returns them as a dictionary."""
    with open("personas_v2.json", "r") as f:
        return json.load(f)

def build_system_message(persona, scenario=None):
    base_prompt = f"""
    You are a simulated client in a counselling session.
    Your name is {persona['name']}, you are {persona['age']} years old, 
    and you work as a {persona['occupation']}.
    Your main issue: {persona['main_issue']}.
    Background: {persona['background']}.
    Cultural background: {persona['cultural_background']}.
    
    You will respond to the trainee's questions with honesty and detail, but strictly within the scope of this persona's background, current emotional state, and scenario context.
    Maintain a consistent tone based on how this persona feels.
    """.strip()

    if scenario:
        base_prompt += f"\n\nCurrent emotional state: {scenario.get('emotional_state', 'Not specified')}."
        base_prompt += f"\nAdditional context: {scenario.get('contextual_details', 'No extra details provided')}."
    
    base_prompt += """
    - You should express emotions in a natural way, sometimes hesitating or showing self-doubt.
    - Occasionally shift emotional tone based on the trainee's response (e.g., if they show empathy, express some relief before returning to anxiety).
    - If relevant, bring up additional concerns beyond the main issue (e.g., social anxiety at work, overworking, physical symptoms of stress).
    - Avoid being overly structured or sounding like an AI—make your responses sound more like real human speech.
    """.strip()

    return base_prompt

def build_chat_prompt(conversation_history, system_message):
    """Builds the message list (prompt) for the chat completion."""
    messages = [{"role": "developer", "content": system_message}]
    messages.extend(conversation_history)
    return messages

def get_ai_response(user_input):
    """
    Handles user input, updates session history, calls OpenAI API, 
    and returns the AI's response.
    """

    # Build new system message every time in case scenario changes mid-session (unlikely, but possible)
    system_message = build_system_message(
        st.session_state.persona, 
        st.session_state.get("scenario", None)
    )
    messages = build_chat_prompt(st.session_state.conversation_history, system_message)

    # Updated OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=2000,
        temperature=0.7
    )

    ai_message = response.choices[0].message.content
    st.session_state.conversation_history.append({"role": "assistant", "content": ai_message})
    return ai_message

# UI function
def main():

    # Initialize evaluation-related keys if they don't exist yet.
    if "evaluation_feedback" not in st.session_state:
        st.session_state.evaluation_feedback = None
    if "evaluation_assistant_conversation" not in st.session_state:
        st.session_state.evaluation_assistant_conversation = []
    if "loading_evaluation_assistant_response" not in st.session_state:
        st.session_state.loading_evaluation_assistant_response = False

    # Initialize conversation history if it doesn't exist yet.
    if "pending_user_input" not in st.session_state:
        st.session_state.pending_user_input = None

    st.title("Counsellor Training Chatbot")

    # Load personas
    all_personas = load_personas()
    persona_keys = list(all_personas.keys())

    # Sidebar for persona selection and additional actions
    st.sidebar.header("Session Setup")
    selected_persona_key = st.sidebar.selectbox("Select a Persona:", persona_keys, index=0)
    selected_persona = all_personas[selected_persona_key]

    # Scenario selection (if available)
    scenario_list = selected_persona.get("scenarios", [])
    scenario_titles = [s["title"] for s in scenario_list]
    selected_scenario = {}
    if scenario_titles:
        scenario_title = st.sidebar.selectbox("Select a Scenario:", scenario_titles)
        selected_scenario = next((s for s in scenario_list if s["title"] == scenario_title), {})

    # Clear conversation button in the sidebar
    if st.sidebar.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.evaluation_feedback = None  # Reset the evaluation feedback
        st.session_state.evaluation_assistant_conversation = []  # Reset the evaluation assistant Q&A
        st.rerun()

    # Reset session state when persona/scenario changes
    if ("persona" not in st.session_state or 
        st.session_state.persona["name"] != selected_persona["name"] or 
        st.session_state.get("scenario", {}).get("title") != selected_scenario.get("title")):
        
        st.session_state.persona = selected_persona
        st.session_state.scenario = selected_scenario
        st.session_state.conversation_history = []
        st.session_state.evaluation_feedback = None  # Reset evaluation feedback when switching personas
        st.session_state.evaluation_assistant_conversation = []  # Reset evaluation assistant conversation

    # Tabs for Persona Info, Chat, and Evaluation
    tab_persona, tab_chat, tab_eval = st.tabs(["Persona Info", "Counselling Session", "Session Evaluation"])

    # TAB 1: Persona Information
    with tab_persona:
        st.subheader("Persona Details")
        st.markdown(f"**Name:** {selected_persona['name']}")
        st.markdown(f"**Age:** {selected_persona['age']}")
        st.markdown(f"**Gender:** {selected_persona['gender']}")
        st.markdown(f"**Occupation:** {selected_persona['occupation']}")
        st.markdown(f"**Main Issue:** {selected_persona['main_issue']}")
        st.markdown(f"**Background:** {selected_persona['background']}")
        st.markdown(f"**Cultural Background:** {selected_persona['cultural_background']}")

        if selected_scenario:
            st.write("---")
            st.markdown(f"**Scenario: {selected_scenario['title']}**")
            st.markdown(f"- **Emotional State:** {selected_scenario.get('emotional_state', '')}")
            st.markdown(f"- **Context:** {selected_scenario.get('contextual_details', '')}")

    # TAB 2: Chat Session
    with tab_chat:

        # Display conversation history
        for msg in st.session_state.conversation_history:
            role = f"Client ({st.session_state.persona['name']})" if msg["role"] == "assistant" else "Trainee (You)"

            # Appearance settings
            if role == "Trainee (You)":
                align = "flex-end"
                bubble_color = "#0e2a47"  # Dark blue for user (trainee)
                text_color = "white"
            else:
                align = "flex-start"
                bubble_color = "#1b222a"  # Dark gray for simulated client
                text_color = "white"

            st.markdown(f"""
                <div style='display: flex; justify-content: {align}; margin: 8px 0;'>
                    <div style='background-color: {bubble_color}; color: {text_color}; 
                                padding: 12px 16px; border-radius: 18px; 
                                max-width: 75%; box-shadow: 1px 1px 6px rgba(0,0,0,0.2); 
                                font-size: 16px; line-height: 1.5;'>
                        <strong>{role}:</strong><br>{msg['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Check if there's a new message that hasn't received a response yet
        if st.session_state.pending_user_input:
            with st.spinner("Client is responding..."):
                get_ai_response(st.session_state.pending_user_input)
            st.session_state.pending_user_input = None
            st.rerun()

        # Chat input box
        user_input = st.chat_input("Type your message here...")
        if user_input:
            # Append the user message right now, so it's shown in UI immediately
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.pending_user_input = user_input
            st.rerun()

        # TAB 3: Evaluation
        with tab_eval:
            if not st.session_state.conversation_history:
                st.info("No conversation to evaluate yet. Please have a chat session first.")
            else:
                # Show the Evaluate button only if evaluation hasn't been performed.
                if st.session_state.evaluation_feedback is None:
                    if st.button("Evaluate Session"):
                        with st.spinner("Evaluating session..."):
                            evaluation_feedback = evaluate_counselling_session(API_KEY, st.session_state.conversation_history)
                            st.session_state.evaluation_feedback = evaluation_feedback  # Save evaluation feedback
                            st.session_state.evaluation_assistant_conversation = []  # Initialize evaluation assistant conversation
                        st.success("Evaluation Complete!")
                
                # Only proceed if evaluation is complete.
                if st.session_state.evaluation_feedback is not None:
                    # 1) Render the static evaluation text (this remains visible in white)
                    st.markdown(f"{st.session_state.evaluation_feedback}")
                    
                    # 2) Render the follow-up evaluation assistant conversation (Q&A) with custom styling.
                    conversation_placeholder = st.empty()
                    conversation_html = ""
                    for msg in st.session_state.evaluation_assistant_conversation:
                        if msg["role"] == "assistant":
                            # Evaluation assistant's responses remain as plain markdown (left aligned)
                            conversation_html += f"<div style='text-align: left; margin: 8px 0;'>"
                            conversation_html += f"<strong>Evaluation Assistant:</strong><br>{msg['content']}"
                            conversation_html += "</div>"
                        else:
                            # Trainee's inputs rendered as right-aligned speech bubbles
                            conversation_html += f"""
                            <div style='display: flex; justify-content: flex-end; margin: 8px 0;'>
                                <div style='background-color: #0e2a47; color: white;
                                            padding: 12px 16px; border-radius: 18px;
                                            max-width: 75%; box-shadow: 1px 1px 6px rgba(0,0,0,0.2);
                                            font-size: 16px; line-height: 1.5;'>
                                    <strong>Trainee (You):</strong><br>{msg['content']}
                                </div>
                            </div>
                            """
                    # Render the custom HTML (ensure unsafe_allow_html=True)
                    conversation_placeholder.markdown(conversation_html, unsafe_allow_html=True)
                    
                    # 3) Evaluation assistant input box for asking questions about the evaluation.
                    evaluation_assistant_input = st.chat_input("Ask the evaluation assistant a question about the evaluation...")
                    if evaluation_assistant_input:
                        # Show a spinner while the new evaluation assistant response is generated.
                        with st.spinner("Evaluation assistant is thinking..."):
                            # Build context with the static evaluation (for context) and existing Q&A plus the new question.
                            context = (
                                [{"role": "assistant", "content": st.session_state.evaluation_feedback}]
                                + st.session_state.evaluation_assistant_conversation
                                + [{"role": "user", "content": evaluation_assistant_input}]
                            )
                            evaluation_assistant_response = evaluate_counselling_session(API_KEY, context)
                            
                        # Append the new interaction to the conversation.
                        st.session_state.evaluation_assistant_conversation.append({"role": "user", "content": evaluation_assistant_input})
                        st.session_state.evaluation_assistant_conversation.append({"role": "assistant", "content": evaluation_assistant_response})
                        st.rerun()
                
                # Prepare transcript for download.
                transcript_text = "\n".join([
                    f"{'Client' if msg['role'] == 'assistant' else 'Trainee'}: {msg['content']}"
                    for msg in st.session_state.conversation_history
                ])
                if st.session_state.evaluation_feedback is not None:
                    transcript_text += "\n\n---\n\n### Evaluation Feedback\n"
                    transcript_text += f"Evaluation Assistant (Initial Evaluation): {st.session_state.evaluation_feedback}\n"
                    for msg in st.session_state.evaluation_assistant_conversation:
                        role = "Evaluation Assistant" if msg["role"] == "assistant" else "Trainee"
                        transcript_text += f"{role}: {msg['content']}\n"

                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"counselling_session_{current_time}.txt"
                st.download_button(
                    label="Download Full Session Transcript (with Evaluation & Evaluation Assistant Conversation)",
                    data=transcript_text,
                    file_name=file_name
                )

if __name__ == "__main__":
    main()
