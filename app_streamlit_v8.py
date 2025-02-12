# app_streamlit_v8.py

import os
import json
import streamlit as st
from openai import OpenAI  # Updated API usage
from dotenv import load_dotenv

from evaluate_session import evaluate_counseling_session

st.set_page_config(page_title="Counselor Training Chatbot", layout="wide") # Set page title and layout

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Password protection
PASSWORD = "qwer7890uiop"  # Change this to your own secure password

def check_password():
    """Simple password authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.sidebar.header("Secure Login")
        entered_password = st.sidebar.text_input("Enter Password:", type="password")

        if entered_password:  # Only check after user enters something
            if entered_password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()  # Refresh the app to hide the login UI
            else:
                st.sidebar.error("Incorrect password. Try again.")

    if not st.session_state.authenticated:
        st.stop()  # Prevents rest of app from running until authenticated

check_password()  # Enforce login before loading the app

def load_personas():
    """Loads the personas from the JSON file and returns them as a dictionary."""
    with open("personas_v2.json", "r") as f:
        return json.load(f)

def build_system_message(persona, scenario=None):
    base_prompt = f"""
    You are a simulated client in a counseling session.
    Your name is {persona['name']}, you are {persona['age']} years old, 
    and you work as a {persona['occupation']}.
    Your main issue: {persona['main_issue']}.
    Background: {persona['background']}.
    Cultural background: {persona['cultural_background']}.
    
    You will respond to the trainee's questions with honesty and detail,
    but strictly within the scope of this persona's background, current emotional state, and scenario context.
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
    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    # Build new system message every time in case scenario changes mid-session (unlikely, but possible)
    system_message = build_system_message(
        st.session_state.persona, 
        st.session_state.get("scenario", None)
    )
    messages = build_chat_prompt(st.session_state.conversation_history, system_message)

    # Updated OpenAI API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )

    ai_message = response.choices[0].message.content
    st.session_state.conversation_history.append({"role": "assistant", "content": ai_message})
    return ai_message

def main():
    """Streamlit UI function."""
    st.title("Counselor Training Chatbot")

    # Load personas
    all_personas = load_personas()
    persona_keys = list(all_personas.keys())

    # Sidebar for persona selection
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

    # Reset session state when persona/scenario changes
    if ("persona" not in st.session_state or 
        st.session_state.persona["name"] != selected_persona["name"] or 
        st.session_state.get("scenario", {}).get("title") != selected_scenario.get("title")):
        
        st.session_state.persona = selected_persona
        st.session_state.scenario = selected_scenario
        st.session_state.conversation_history = []
        st.session_state.evaluation = None  # Reset evaluation when switching personas

    # Tabs for Persona Info, Chat, and Evaluation
    tab_persona, tab_chat, tab_eval = st.tabs(["Persona Info", "Chat Session", "Evaluation"])

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
        st.subheader("Counseling Session")

        # Display conversation history
        for msg in st.session_state.conversation_history:
            if msg["role"] == "assistant":
                st.markdown(f"**{selected_persona['name']} (Client):** {msg['content']}")
            elif msg["role"] == "user":
                st.markdown(f"**You (Trainee):** {msg['content']}")

        # Chat input box
        user_input = st.chat_input("Type your message here...")
        if user_input:
            get_ai_response(user_input)
            st.rerun()

        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.evaluation = None  # Reset evaluation if clearing chat
            st.rerun()

    # TAB 3: Evaluation
    with tab_eval:
        st.subheader("Session Evaluation")

        if not st.session_state.conversation_history:
            st.info("No conversation to evaluate yet. Please have a chat session first.")
        else:
            if st.button("Evaluate Session"):
                with st.spinner("Evaluating session..."):
                    evaluation_feedback = evaluate_counseling_session(api_key, st.session_state.conversation_history)
                    st.session_state.evaluation = evaluation_feedback  # Store evaluation in session state
                st.success("Evaluation Complete!")
                st.markdown("### Supervisor Feedback")
                st.write(st.session_state.evaluation)

            # Prepare transcript including evaluation
            transcript_text = "\n".join([
                f"{'Client' if msg['role'] == 'assistant' else 'Trainee'}: {msg['content']}"
                for msg in st.session_state.conversation_history
            ])

            # Append evaluation (if available)
            if st.session_state.evaluation:
                transcript_text += "\n\n---\n\n### Evaluation Feedback\n" + st.session_state.evaluation

            # Download button for transcript with evaluation
            st.download_button(
                label="Download Full Session Transcript (with Evaluation)",
                data=transcript_text,
                file_name="counseling_session_with_evaluation.txt"
            )

if __name__ == "__main__":
    main()

