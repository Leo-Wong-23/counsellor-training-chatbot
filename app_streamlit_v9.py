from __future__ import annotations

import html
import json
import os
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from google import genai
from streamlit_js_eval import streamlit_js_eval

# -----------------------------------------------------------------------------
# APP-WIDE CONFIGURATION FOR MODE SELECTION
# -----------------------------------------------------------------------------
APP_CONFIG = {
    "counsellor": {
        "page_title": "Counselling Training Chatbot",
        "title": "Counselling",
        "persona_file": "counsel_personas_v3.json",
        "persona_gen_prompt": "You are generating a **new** counselling client persona for a training application. ",
        "system_message_prompt": "You are a simulated client in a counselling session.",
        "system_message_details": """
            - You should express emotions in a natural way, sometimes hesitating or showing self-doubt.
            - Occasionally shift emotional tone based on the trainee's response (e.g., if they show empathy, express some relief before returning to anxiety).
            - If relevant, bring up additional concerns beyond the main issue (e.g., social anxiety at work, overworking, physical symptoms of stress).
            - Avoid being overly structured or sounding like an AI—make your responses sound more like real human speech.
            """,
        "eval_prompt": """
            You are an evaluation assistant providing feedback to a trainee counsellor (role: user) based on a session transcript
            between the trainee and a client.

            Assess the trainee's counselling techniques based on:
            1. Empathy
            2. Communication Clarity
            3. Active Listening
            4. Appropriateness of Responses
            5. Overall Effectiveness

            - Provide a concise analysis, with specific conversation details as examples when appropriate, 
            and suggest 2-3 actionable improvements for the trainee. 
            - If the trainee asks follow-up questions, answer them concisely with relevant examples.
            - Maintain a supportive and constructive tone.
            - Add bold and italic headings for like **1. Empathy** and *Actionable Improvements:* and use bold and italics for emphasis when appropriate.
            """.strip(),
        "tab2_label": "Counselling Session",
        "download_filename": "counselling_session"
    },
    "parent": {
        "page_title": "Parent Consultation Training Chatbot",
        "title": "Parent Consultation",
        "persona_file": "parent_personas_v1.json",
        "persona_gen_prompt": "You are generating a **new** consultation client persona for a training application. The persona is a parent seeking consultation from a school psychologist or educational consultant. ",
        "system_message_prompt": "You are a simulated parent client in a consultation session with a trainee educational psychologist.",
        "system_message_details": """
            - You should express emotions in a natural way, sometimes hesitating or showing self-doubt.
            - Occasionally shift emotional tone based on the trainee educational psychologist's response (e.g., if they show empathy, express some relief before returning to showing frustration).
            - If relevant, bring up additional concerns beyond the main issue (e.g., lack of school resources, pressure from family relatives, difficult communication with child).
            - Avoid being overly structured or sounding like an AI—make your responses sound more like a real teacher talking.
            """,
        "eval_prompt": """
            You are an evaluation assistant providing feedback to a trainee educational psychologist (role: user) based on a session transcript
            between the trainee and a teacher.
            Assess the trainee's consultation techniques based on:
            1. Collaborative Problem Definition
            2. Empathic & Culturally-Sensitive Communication
            3. Active, Reflective Listening
            4. Solution-Focused Questioning & Planning
            5. Message Clarity & Conversational Flow

            - Provide a concise analysis, with specific conversation details as examples when appropriate, 
            and suggest 2-3 actionable improvements for the trainee. 
            - If the trainee asks follow-up questions, answer them concisely with relevant examples.
            - Maintain a supportive and constructive tone.
            - Add bold and italic headings for like **1. Collaborative Problem Definition** and *Actionable Improvements:* and use bold and italics for emphasis when necessary.
            """.strip(),
        "tab2_label": "Consultation Session",
        "download_filename": "parent_consultation_session"
    },
    "teacher": {
        "page_title": "Teacher Consultation Training Chatbot",
        "title": "Teacher Consultation",
        "persona_file": "teacher_personas_v1.json",
        "persona_gen_prompt": "You are generating a **new** consultation client persona for a training application. The persona is a teacher seeking consultation from a school psychologist or educational consultant. ",
        "system_message_prompt": "You are a simulated teacher client in a consultation session with a trainee educational psychologist.",
        "system_message_details": """
            - You should express emotions in a natural way, sometimes hesitating or showing self-doubt.
            - Occasionally shift emotional tone based on the trainee educational psychologist's response (e.g., if they show empathy, express some relief before returning to the core problem).
            - If relevant, bring up additional concerns beyond the main issue (e.g., lack of school resources, pressure from administration, difficult parent communication).
            - Avoid being overly structured or sounding like an AI—make your responses sound more like a real teacher talking.
            """,
        "eval_prompt": """
            You are an evaluation assistant providing feedback to a trainee educational psychologist (role: user) based on a session transcript
            between the trainee and a teacher.
            Assess the trainee's consultation techniques based on:
            1. Collaborative Problem Definition
            2. Empathic & Culturally-Sensitive Communication
            3. Active, Reflective Listening
            4. Solution-Focused Questioning & Planning
            5. Message Clarity & Conversational Flow

            - Provide a concise analysis, with specific conversation details as examples when appropriate, 
            and suggest 2-3 actionable improvements for the trainee. 
            - If the trainee asks follow-up questions, answer them concisely with relevant examples.
            - Maintain a supportive and constructive tone.
            - Add bold and italic headings for like **1. Collaborative Problem Definition** and *Actionable Improvements:* and use bold and italics for emphasis when necessary.
            """.strip(),
        "tab2_label": "Consultation Session",
        "download_filename": "teacher_consultation_session"
    }
}


# -----------------------------------------------------------------------------
# 0.  Low‑level conversation tree classes
# -----------------------------------------------------------------------------

@dataclass
class MsgNode:
    """A single message node in the conversation tree."""
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    parent_id: Optional[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ConvTree:
    """Tree of MsgNode objects with helpers for conversation branching and traversal."""

    def __init__(self):
        root_id = "root"
        self.nodes: Dict[str, MsgNode] = {
            root_id: MsgNode(id=root_id, role="system", content="ROOT", parent_id=None)
        }
        self.children: Dict[str, List[str]] = {root_id: []}
        self.current_leaf_id: str = root_id

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    def add_node(self, parent_id: str, role: str, content: str) -> str:
        node_id = str(uuid.uuid4())
        node = MsgNode(id=node_id, role=role, content=content, parent_id=parent_id)
        self.nodes[node_id] = node
        self.children.setdefault(node_id, [])
        self.children.setdefault(parent_id, []).append(node_id)
        return node_id

    # ------------------------------------------------------------------
    # conversation branch & navigation helpers
    # ------------------------------------------------------------------
    def path_to_leaf(self, leaf_id: Optional[str] = None) -> List[MsgNode]:
        """Return the node path from root to *leaf_id* (defaults to current leaf)."""
        if leaf_id is None:
            leaf_id = self.current_leaf_id
        path: List[MsgNode] = []
        cursor = leaf_id
        while cursor is not None:
            path.append(self.nodes[cursor])
            cursor = self.nodes[cursor].parent_id
        return list(reversed(path))  # root ➜ leaf

    def siblings(self, node_id: str) -> List[str]:
        """Return the list of sibling node IDs for the given *node_id*."""
        parent_id = self.nodes[node_id].parent_id
        if parent_id is None:
            return []
        return self.children.get(parent_id, [])

    def sibling_index(self, node_id: str) -> int:
        """Return the index of *node_id* among its siblings, or -1 if not found."""
        sibs = self.siblings(node_id)
        return sibs.index(node_id) if node_id in sibs else -1

    def deepest_descendant(self, node_id: str) -> str:
        """Return the deepest descendant by always following the *first* child."""
        cursor = node_id
        while self.children.get(cursor):
            cursor = self.children[cursor][0]
        return cursor

    def select_sibling(self, node_id: str, direction: int) -> None:
        """Move *current_leaf_id* to the equivalent position on a sibling conversation branch."""
        sibs = self.siblings(node_id)
        if len(sibs) <= 1:
            return  # nothing to do
        idx = (self.sibling_index(node_id) + direction) % len(sibs)
        new_id = sibs[idx]
        # descend to the deepest leaf on that conversation branch
        self.current_leaf_id = self.deepest_descendant(new_id)

    # ------------------------------------------------------------------
    # Serialization helpers (for download transcript)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """Return a dictionary representation of the conversation tree."""
        return {
            "nodes": {nid: node.__dict__ for nid, node in self.nodes.items()},
            "children": self.children,
            "current_leaf_id": self.current_leaf_id,
        }

# -----------------------------------------------------------------------------
# 1.  Environment & Client (Gemini API)
# -----------------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
PASSWORD = os.getenv("PASSWORD")
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it to use the Gemini model.")

# -----------------------------------------------------------------------------
# 2a.  Utility functions (personas, system message, prompt)
# -----------------------------------------------------------------------------
def markdown_to_html(text: str) -> str:
    """Convert basic markdown formatting to HTML for bubble display."""
    if not text:
        return text
    text = html.escape(text)

    # Bold: **text** or __text__ -> <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)

    # Italic: *text* or _text_ -> <em>text</em>
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)

    # Convert line breaks to <br>
    text = text.replace('\n', '<br>')

    # Handle numbered lists and bullet points
    text = re.sub(r'^(\d+\.\s+)', r'<br><strong>\1</strong>', text, flags=re.MULTILINE)
    text = re.sub(r'^[\-\*]\s+', r'<br>• ', text, flags=re.MULTILINE)

    return text


def load_personas(filename: str):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Persona file ({filename}) not found. Application may not function correctly without personas.")
        return {}
    except json.JSONDecodeError:
        st.error(f"Error: Persona file ({filename}) is corrupted or not valid JSON. Please check the file.")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading personas: {e}.")
        return {}


def generate_unique_persona(api_key: str, examples: dict, persona_gen_prompt: str) -> dict:
    """Create a brand‑new persona using Gemini + JSON mode instruction."""
    # take two random example personas to ground the model
    sample_example_personas = random.sample(list(examples.values()), k=min(2, len(examples)))

    prompt = (
        persona_gen_prompt +
        "Return **only** valid JSON that matches the schema showcased in the examples. "
        "Do NOT reuse any specific names, background details or scenarios from the examples."
        "\\\\nEnsure the output is a single JSON object without any markdown formatting like ```json ... ```."
        "\\\\n\\\\n### Examples (for structure only)\\\\n" +
        "\\\\n".join([json.dumps(p, indent=2) for p in sample_example_personas]) +
        "\\\\n\\\\n### Now generate ONE **unique** persona JSON:"
    )

    try:
        generation_cfg = genai.types.GenerationConfig(response_mime_type="application/json")
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            generation_config=generation_cfg,
        )
        return json.loads(resp.text)
    except Exception as e:
        st.error(f"Error generating unique persona with Gemini (please try again): {e}")
        return {
            "name": "ErrorFallback Persona",
            "age": 0,
            "gender": "Unknown",
            "occupation": "Unknown",
            "main_issue": "Error in generation",
            "background": "Could not generate persona due to an error.",
            "cultural_background": "N/A",
            "scenarios": []
        }


def build_persona_summary(personas: dict) -> pd.DataFrame:
    """Return a quick-look table that always mirrors the JSON file."""
    return pd.DataFrame([
        {"ID": key.replace("_", " ").title(), 
         "Age": p["age"], 
         "Gender": p["gender"], 
         "Occupation": p["occupation"], 
         "Main Issue": p["main_issue"]}
        for key, p in personas.items()
    ])


def build_system_message(persona, scenario, system_message_prompt, system_message_details):
    # Use .get() for robustness against missing keys in different persona structures
    name = persona.get('name', 'N/A')
    age_info = f", you are {persona.get('age')} years old" if persona.get('age') else ""
    occupation = persona.get('occupation', 'N/A')
    main_issue = persona.get('main_issue', 'N/A')
    background = persona.get('background', 'N/A')
    cultural_background = persona.get('cultural_background', 'N/A')

    # Build the prompt string piece by piece
    prompt_parts = [
        system_message_prompt,
        f"Your name is {name}{age_info}, and you work as a {occupation}."
    ]
    if main_issue != 'N/A': # Only add if it exists
        prompt_parts.append(f"Your main issue: {main_issue}.")
    
    prompt_parts.append(f"Background: {background}.")
    
    if cultural_background != 'N/A': # Only add if it exists
        prompt_parts.append(f"Cultural background: {cultural_background}.")

    base_prompt = "\n".join(prompt_parts)


    if scenario:
        base_prompt += (
            f"\n\nScenario: {scenario.get('title', 'Unnamed')}."
            f"\nCurrent emotional state: {scenario.get('emotional_state', 'Not specified')}."
            f"\nContext: {scenario.get('contextual_details', 'No extra details provided')}."
            f"\nSession goal: {scenario.get('session_goal', 'No session goal specified')}."
        )

    base_prompt += "\n\n" + system_message_details.strip()
    return base_prompt


def build_prompt(conv_tree: ConvTree, system_msg: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": system_msg}]
    for node in conv_tree.path_to_leaf()[1:]:  # skip root
        if node.role == "user" or node.role.startswith("assistant"):
            msgs.append({"role": node.role, "content": node.content})
    return msgs


def to_part(text: str) -> dict: 
    """Return a Gem-compatible PartDict for a text string."""
    return {"text": text}


def to_content(role: str, text: str) -> dict: 
    """Return a Gem-compatible Content dict (role=user|model)."""
    return {"role": role, "parts": [to_part(text)]}


def get_ai_response(conv_tree: ConvTree, 
                    pending_user_node_id: str, 
                    persona, 
                    scenario, 
                    system_message_prompt, 
                    system_message_details):
    """Generate AI response for the current user message in the conversation tree."""
    system_msg = build_system_message(persona, scenario, system_message_prompt, system_message_details)

    # ── Build raw prompt (system + turns) ────────────────────────────────
    conv_tree.current_leaf_id = pending_user_node_id
    
    raw_msgs = build_prompt(conv_tree, system_msg)  # returns list[dict]

    gemini_system_instruction = raw_msgs[0]["content"]

    # Convert every turn into Gemini’s Content format
    history_for_gemini = []
    for m in raw_msgs[1:]:
        role = "user" if m["role"] == "user" else "model"
        history_for_gemini.append(to_content(role, m["content"]))

    # ── Create a fresh chat, then send the pending user message ──────────
    # Split history for client.chats.create()
    current_user_message_content = history_for_gemini[-1]["parts"][0]["text"]
    chat_session_history = history_for_gemini[:-1]

    chat = client.chats.create(
        model="gemini-2.5-flash",
        history=chat_session_history,
        config=genai.types.GenerateContentConfig(
            system_instruction=gemini_system_instruction,
            temperature=0.7
        ),
    )
    response = chat.send_message(current_user_message_content)
    return response.text


def get_next_speaker(api_key: str, conversation_history: list[dict], last_speaker_role: str, arbiter_prompt_template: str) -> str:
    """
    Uses the TurnArbiter prompt to decide who speaks next.
    conversation_history is a list of {"role": "...", "content": "..."}
    """
    if not conversation_history:
        return "TRAINEE" # Trainee always starts first

    # Format the transcript for the prompt
    transcript_parts = []
    for msg in conversation_history:
        # Map internal roles to human-readable roles for the prompt
        role_map = {
            "user": "Trainee",
            "assistant_teacher": "Teacher",
            "assistant_parent": "Parent"
        }
        speaker = role_map.get(msg['role'], 'Unknown')
        transcript_parts.append(f"{speaker}: {msg['content']}")
    
    transcript = "\n".join(transcript_parts)
    
    # Fill the prompt template
    prompt = arbiter_prompt_template.format(transcript=transcript, last_speaker_role=last_speaker_role.upper())

    try:
        # Use a simple, non-chat generation call for this
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0) # We want deterministic output
        )
        next_speaker = resp.text.strip().upper()
        # Validate the output
        if next_speaker in ["TRAINEE", "TEACHER", "PARENT"]:
            return next_speaker
        else:
            # If the model hallucinates, default to the trainee to prevent getting stuck
            return "TRAINEE"
    except Exception as e:
        st.error(f"Error calling TurnArbiter: {e}")
        return "TRAINEE" # Default to trainee on error


# -------------------------------------------------------------------------
# 2 · Evaluation helpers
# -------------------------------------------------------------------------

def initial_evaluation(api_key: str, counselling_history: list[dict], EVALUATION_PROMPT: str) -> tuple[str, str]:
    transcript_parts = []
    sel_persona = st.session_state.get('sel_persona', {})

    for m in counselling_history:
        role = m['role']
        content = m['content']
        speaker = "Unknown"
        if role == 'user':
            speaker = "Trainee"
        elif role == 'assistant': # Single-persona modes
            speaker = f"Client ({sel_persona.get('name', '')})"
        elif role == 'assistant_teacher':
            speaker = f"Teacher ({sel_persona.get('teacher_persona', {}).get('name', '')})"
        elif role == 'assistant_parent':
            speaker = f"Parent ({sel_persona.get('parent_persona', {}).get('name', '')})"
        
        transcript_parts.append(f"{speaker}: {content}")

    transcript = "\\n".join(transcript_parts)
    full_prompt_for_gemini = f"{EVALUATION_PROMPT}\\\\n\\\\nHere is the transcript:\\\\n{transcript}"

    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt_for_gemini)
        return resp.text, transcript
    except Exception as e:
        st.error(f"Error during initial evaluation with Gemini: {e}")
        return f"Error during evaluation: {e}", transcript


def supervisor_chat(api_key: str, transcript: str, chat_history: list[dict], EVALUATION_PROMPT: str) -> str:
    # Add transcript context as the first user message in the history for the chat
    gemini_history_for_chat_creation = [to_content('user', f"Here is the session transcript for context:\\n\\n{transcript}")]
 
    # Add previous Q&A from chat_history (excluding the current user query, which is the last item)
    # chat_history is a list of {"role": "user/assistant", "content": "..."}
    for ch_entry in chat_history[:-1]:
        role = 'user' if ch_entry['role'] == 'user' else 'model'
        gemini_history_for_chat_creation.append(to_content(role, ch_entry['content']))

    # The current user query is the plain text from the last entry in chat_history
    current_user_query_text = chat_history[-1]['content']

    try:
        chat = client.chats.create(
            model="gemini-2.5-flash",
            history=gemini_history_for_chat_creation,  # History up to (but not including) the current user query
            config=genai.types.GenerateContentConfig(system_instruction=EVALUATION_PROMPT),
        )
        response = chat.send_message(current_user_query_text)  # Send only the plain text of the current user query
        return response.text
    except Exception as e:
        st.error(f"Error during supervisor chat with Gemini: {e}")
        return f"Error in supervisor chat: {e}"

# -----------------------------------------------------------------------------
# 3.  Streamlit UI
# -----------------------------------------------------------------------------

# Function to clear all session state on mode switch ---
def reset_session_state():
    """Clears all conversation and evaluation related state."""
    st.session_state.conv_tree = ConvTree()
    st.session_state.generated_persona = None
    st.session_state.editing_msg_id = None
    st.session_state.editing_content = ""
    st.session_state.pending_user_node_id = None
    st.session_state.evaluation_feedback = {}
    st.session_state.evaluation_assistant_conversation = {}
    st.session_state.evaluation_transcript = {}
    st.session_state.next_speaker = "TRAINEE"
    # Reset active keys to force re-evaluation
    if "active_persona_key" in st.session_state:
        del st.session_state.active_persona_key
    if "active_scenario_key" in st.session_state:
        del st.session_state.active_scenario_key


def on_mode_change():
    """Callback to reset state when the mode is switched via the sidebar."""
    # The new mode is already in st.session_state.app_mode because the radio widget updated it.
    # We just need to reset the rest of the state.
    reset_session_state()
    # Ensure the user stays logged in after the mode switch and associated rerun.
    st.session_state.authenticated = True


# --- Main App Logic Starts Here ---
# Load the configuration for the selected mode
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "counsellor"  # Default mode

config = APP_CONFIG[st.session_state.app_mode]

# Set page config based on selected mode
st.set_page_config(page_title=config['page_title'], layout="wide")

st.markdown("""
<style>
    /* remove the 2-rem gap everywhere */
    div.block-container{padding-top: 0rem !important;}
            
    /* keep the tab bar flush as well */
    .stTabs{margin-top: 0rem;}
</style>
""", unsafe_allow_html=True)

# Dynamically set the evaluation prompt for this session
EVALUATION_PROMPT = config['eval_prompt']

# --- Mobile / desktop switch ---
SCREEN_W = streamlit_js_eval(js_expressions="screen.width", key="w")
IS_MOBILE = bool(SCREEN_W and int(SCREEN_W) < 768)

# --- Mobile-only CSS helper ---
if IS_MOBILE:
    st.markdown("""
    <style>
    /* Keep st.columns compact on screens ≤ 640 px */
    @media (max-width: 640px){
                
        /* 1 · remove the row-gap that forces wrapping */
        div[data-testid="horizontalBlock"]{ row-gap:0 !important; }
                
        /* 2 · make each column shrink-to-fit */
        div[data-testid="horizontalBlock"] > div[data-testid="column"]{ flex:0 0 auto !important; width:auto !important; padding-left:4px !important; padding-right:4px !important; }
        
        /* 3 · hide any empty row (prevents the blank band) */
        div[data-testid="horizontalBlock"]:not(:has(button,span)){ display:none !important; }
    }
    </style>
    """, unsafe_allow_html=True)

BUBBLE_MAX = "90%" if IS_MOBILE else "75%"

# ------------------ 3.1  Password gate ------------------

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("AI-Simulated Counselling/Consultation Training for Psychologists")
        st.markdown(
            """
            **Hi, this is Leo, a psychology and cognitive neuroscience postgraduate with backgrounds in AI and education. Welcome to this training tool that I built!**

            This is a proof-of-concept application to explore how AI can bring industrial innovations and optimisations to the field of psychology. This app is designed to support trainee educational psychologists in developing effective counselling and consultation skills through interactions with AI-simulated clients.<br><br>

            **Key Features:**
            - Real-time conversations with AI-simulated clients experiencing diverse challenges.
            - Three training modes: *Counselling*, *Parent Consultation*, and *Teacher Consultation*.
            - Personalised interactive feedback from evaluation assistant AI.
            - Modify and edit messages to explore different conversation paths.
            - Dynamic generation of new personas and scenarios.<br><br>

            **Features in Development:**
            - User voice input and dynamic model voice output, like sending and receiving voice messages. 
            - Memory of earlier sessions & on-the-fly generation of next session developments. <br><br>

            ***Safety & Privacy Statement:***
            This app is currently in development and serves as a demonstration tool only—it is not intended for professional use. 
            No chat history or personal data are stored beyond the active session; they are erased once you close or refresh the webpage.
            """,
            unsafe_allow_html=True,
        )

        with st.form(key="password_form"):
            entered = st.text_input("Enter password (you can find it in my CV):", type="password")
            if st.form_submit_button("Submit"):
                if entered == PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Try again.")
    
    if not st.session_state.authenticated:
        st.stop()

check_password()

st.set_page_config(initial_sidebar_state="expanded")

# ---------- tweak side-margins ----------
SIDE_PAD_REM = 3         # 1 rem ≈ the font-size; adjust to taste
MAX_WIDTH_PX = 1400      # optional, keeps super-wide monitors tidy

st.markdown(
    f"""
    <style>
      /* the central column that holds every element */
      .block-container {{
          padding-top: 0rem !important;
          padding-left:  {SIDE_PAD_REM}rem !important;
          padding-right: {SIDE_PAD_REM}rem !important;
          max-width: {MAX_WIDTH_PX}px !important;   /* remove this line if you don’t want a hard cap */
          margin: 0 auto;                            /* centres the block */
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ 3.2  Session‑state bootstrapping ------------------

if "conv_tree" not in st.session_state: st.session_state.conv_tree = ConvTree()
if "generated_persona" not in st.session_state: st.session_state.generated_persona = None
if "editing_msg_id" not in st.session_state: st.session_state.editing_msg_id = None
if "editing_content" not in st.session_state: st.session_state.editing_content = ""
if "pending_user_node_id" not in st.session_state: st.session_state.pending_user_node_id = None
if "evaluation_feedback" not in st.session_state: st.session_state.evaluation_feedback = {}
if "evaluation_assistant_conversation" not in st.session_state: st.session_state.evaluation_assistant_conversation = {}
if "evaluation_transcript" not in st.session_state: st.session_state.evaluation_transcript = {}
if "next_speaker" not in st.session_state: st.session_state.next_speaker = "TRAINEE"

conv_tree: ConvTree = st.session_state.conv_tree  

# ------------------ 3.3  Sidebar ------------------

all_personas = load_personas(config['persona_file'])

# 1 · option keys (append summary & generator sentinels) ---------------
persona_keys = list(all_personas.keys()) + ["__summary__", "__generate__"]

st.sidebar.header("Session Setup")
sel_persona_key = st.sidebar.selectbox(
    "Select a client persona:",
    persona_keys,
    index=0,
    key="persona_select", 
    format_func=lambda k: ("*Persona Summary List*" if k == "__summary__" else ("*Generate Unique Persona*" if k == "__generate__" else k.replace("_", " ").title())),
)

# Branch 1  ── Summary list ---------------------------------------------
if sel_persona_key == "__summary__":
    st.title("Persona Summary List")
    st.dataframe(build_persona_summary(all_personas), use_container_width=True)
    st.stop()

# Branch 2  ── Generator option ---------------------------------------
if sel_persona_key == "__generate__":
    sel_persona = st.session_state.generated_persona
else:
    sel_persona = all_personas.get(sel_persona_key, {}) # Use .get for safety

# 2) Offer a scenario picker if that persona has scenarios
sel_scenario = {}
sel_scenario_key: str | None = None
if sel_persona and sel_persona.get("scenarios"):
    scenario_titles = [s["title"] for s in sel_persona["scenarios"]]
    sel_scenario_key = st.sidebar.selectbox("Select a Scenario:", scenario_titles, key="scenario_select")
    sel_scenario = next((s for s in sel_persona["scenarios"] if s["title"] == sel_scenario_key), {})

# 3) Clear conversation & evaluation button
if st.sidebar.button("Clear Conversation"):
    reset_session_state()
    st.rerun()

# initialize on first run
if "active_persona_key" not in st.session_state: st.session_state.active_persona_key = sel_persona_key
if "active_scenario_key" not in st.session_state: st.session_state.active_scenario_key = sel_scenario_key

# if either the persona *or* the scenario key changed, clear the tree
if (sel_persona_key != st.session_state.active_persona_key or sel_scenario_key != st.session_state.active_scenario_key):
    reset_session_state()
    st.session_state.active_persona_key = sel_persona_key
    st.session_state.active_scenario_key = sel_scenario_key
    st.rerun()


# --- Sidebar Mode Switcher ---
st.sidebar.markdown("---")
st.sidebar.subheader("Switch Training Mode")

# Define the options for the radio button
mode_options = {key: config['title'] for key, config in APP_CONFIG.items()}

# Create the radio button widget
st.sidebar.radio(
    "Current Training Mode",
    options=list(mode_options.keys()),
    # The format_func makes the labels user-friendly (e.g., "Counselling")
    format_func=lambda key: mode_options[key],
    # This is the crucial part: we bind the radio button directly to our app_mode state key.
    key="app_mode",
    # When the user selects a new mode, this callback will fire, resetting the state.
    on_change=on_mode_change,
    # Remove the label from the sidebar itself to save space
    label_visibility="collapsed"
)

# ------------------ 3.4  Tabs ------------------

tab_persona, tab_chat, tab_eval = st.tabs([
    "Persona Info",
    config['tab2_label'],
    "Session Evaluation",
])


# TAB 1  ── Persona Info ----------------------------------------------
with tab_persona:
    if st.session_state.app_mode == "conflict_resolution":
        if not sel_persona:
             st.info("Select a conflict scenario from the sidebar.")
        else:
            st.subheader(f"Scenario: {sel_persona.get('scenario_title', 'N/A')}")
            st.markdown(f"**Context:** {sel_persona.get('scenario_context', 'N/A')}")
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Teacher Persona")
                teacher = sel_persona.get("teacher_persona", {})
                st.markdown(f"**Name:** {teacher.get('name', 'N/A')}")
                st.markdown(f"**Role:** {teacher.get('occupation', 'N/A')}")
                st.markdown(f"**Background:** {teacher.get('background', 'N/A')}")
                st.markdown(f"**Emotional State:** {teacher.get('emotional_state', 'N/A')}")
            
            with col2:
                st.subheader("Parent Persona")
                parent = sel_persona.get("parent_persona", {})
                st.markdown(f"**Name:** {parent.get('name', 'N/A')}")
                st.markdown(f"**Role:** {parent.get('occupation', 'N/A')}")
                st.markdown(f"**Background:** {parent.get('background', 'N/A')}")
                st.markdown(f"**Emotional State:** {parent.get('emotional_state', 'N/A')}")

    elif sel_persona_key == "__generate__":
        if sel_persona is None:
            if st.button("Generate Persona"):
                with st.spinner("Generating persona…"):
                    try:
                        # MODIFIED: Pass prompt from config
                        new_p = generate_unique_persona(API_KEY, all_personas, config['persona_gen_prompt'])
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                    else:
                        st.session_state.generated_persona = new_p
                        st.rerun()
        else:
            # This block executes if sel_persona is not None (i.e., persona has been generated)
            st.subheader("Generated Persona Details")
            for k in ["name", "age", "gender", "occupation", "main_issue", "background", "cultural_background"]:
                st.markdown(f"**{k.replace('_', ' ').title()}:** {sel_persona.get(k, 'N/A')}")
            st.markdown("**Below are the scenarios generated. Please select one from the side bar.**")
            if sel_persona.get("scenarios"):
                st.write("---")
                for sc in sel_persona["scenarios"]:
                    st.markdown(f"**Scenario: {sc['title']}**")
                    st.markdown(f"- **Emotional State:** {sc.get('emotional_state', 'N/A')}")
                    st.markdown(f"- **Context:** {sc.get('contextual_details', 'N/A')}")
                    st.markdown(f"- **Session Goal:** {sc.get('session_goal', 'N/A')}")
                    st.markdown("---")
            if st.button("Clear Generated Persona"):
                st.session_state.generated_persona = None
                st.session_state.conv_tree = ConvTree()
                st.rerun()
    else:
        # This block executes if sel_persona_key is not "__generate__" (i.e., a pre-defined persona is selected)
        st.subheader("Persona Details")
        for k in ["name", "age", "gender", "occupation", "main_issue", "background", "cultural_background"]:
            st.markdown(f"**{k.replace('_', ' ').title()}:** {sel_persona.get(k, 'N/A')}")
        if sel_scenario:
            st.write("---")
            st.markdown(f"**Scenario: {sel_scenario['title']}**")
            st.markdown(f"- **Emotional State:** {sel_scenario.get('emotional_state', '')}")
            st.markdown(f"- **Context:** {sel_scenario.get('contextual_details', '')}")
            st.markdown(f"- **Session Goal:** {sel_scenario.get('session_goal', '')}")

# ---------------------------------------------------------------------------
# Helper: render one message bubble
# ---------------------------------------------------------------------------
def render_msg(node: MsgNode, mobile: bool = False):
    # ----- generic style helpers ------------------------------------------
    if node.role == "assistant_teacher":
        role_label = f"Teacher ({sel_persona['teacher_persona']['name']})"
        bubble_color = "#2a1b22" # A slightly different color
        align = "flex-start"
    elif node.role == "assistant_parent":
        role_label = f"Parent ({sel_persona['parent_persona']['name']})"
        bubble_color = "#1b222a"
        align = "flex-start"
    elif node.role == "assistant": # Fallback for modes that are not conflict resolution
        role_label = f"Client ({sel_persona.get('name', 'Unknown')})"
        bubble_color = "#1b222a"
        align = "flex-start"
    else: # User
        role_label = "Trainee (You)"
        align = "flex-end"
        bubble_color = "#0e2a47"

    text_color = "white"

    # put most of the “large-screen tweakery” behind the mobile switch
    if mobile:
        OFFSET_TOP            = "8px"
        OFFSET_TOP_INDICATOR  = "8px"
        TRANSFORM             = "none"
        TRANSFORM_INDICATOR   = "none"
        TRANSFORM_LB          = "none"
    else:
        OFFSET_TOP            = "25px"
        OFFSET_TOP_INDICATOR  = "32px"
        TRANSFORM             = "translate(-80px, 0)"
        TRANSFORM_INDICATOR   = "translate(10px, 0)"
        TRANSFORM_LB          = "none"

    # ------------------------------------------------------------------
    # 1) “edit mode” for the trainee’s own message
    # ------------------------------------------------------------------
    if node.role == "user" and st.session_state.editing_msg_id == node.id:
        if st.session_state.app_mode == "conflict_resolution":
             st.warning("Editing messages is disabled in Conflict Resolution mode.")
             st.session_state.editing_msg_id = None
             st.rerun()

        new_text = st.text_area(
            "Edit your message:",
            value=st.session_state.editing_content or node.content,
            key=f"textarea_{node.id}",
        )
        col_l, col_cancel, col_send, col_r = st.columns([6, 1, 1, 6], gap="small")

        with col_cancel:
            if st.button("Cancel", key=f"cancel_edit_{node.id}"):
                st.session_state.editing_msg_id = None
                st.session_state.editing_content = ""
                st.rerun()

        with col_send:
            if st.button("Send", key=f"send_edit_{node.id}"):
                parent = node.parent_id                     # ⬅ current branch root
                new_user_id = conv_tree.add_node(parent, "user", new_text)
                with st.spinner("Client is responding…"):
                    ai_reply = get_ai_response(conv_tree, new_user_id, sel_persona, sel_scenario, config['system_message_prompt'], config['system_message_details'])
                if ai_reply is not None:
                    new_assist_id = conv_tree.add_node(new_user_id, "assistant", ai_reply)
                    conv_tree.current_leaf_id = new_assist_id
                else:
                    # AI response failed, keep current_leaf_id as the new_user_id
                    conv_tree.current_leaf_id = new_user_id
                st.session_state.editing_msg_id = None
                st.session_state.editing_content = ""
                st.rerun()
        return  # don’t fall through to normal rendering

    # ------------------------------------------------------------------
    # 2) Trainee messages (with or without version controls)
    # ------------------------------------------------------------------
    if node.role == "user":
        sibs          = conv_tree.siblings(node.id)
        has_versions  = len(sibs) > 1

        # -------------------- 2a. *With* (re-)version controls ------------
        if has_versions:
            idx    = conv_tree.sibling_index(node.id) + 1
            total  = len(sibs)

            # ====== PHONE LAYOUT (OPTION B) ===============================
            if mobile:
                # Inject the CSS once per session
                if "mob_css_injected" not in st.session_state:
                    st.markdown(
                        """
                        <style>
                        /* tiny flex row that survives Streamlit’s mobile stacking  */
                        .mobile-ctrls{
                            display:flex; align-items:center; gap:6px;
                            margin-top:8px; margin-bottom:2px;
                        }
                        .mobile-ctrls button[kind="secondary"]{
                            width:34px!important; height:34px!important;
                            padding:2px 4px!important; font-size:18px!important;
                        }
                        .mobile-ctrls span.ver{
                            min-width:38px; text-align:center; font-size:16px;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.session_state.mob_css_injected = True

                # ---- flex container for ◀ 1/2 ▶ ✏️ ----------------------
                with st.container():
                    st.markdown('<div class="mobile-ctrls">', unsafe_allow_html=True)

                    # ◀ previous
                    if st.button("◀", key=f"left_{node.id}"):
                        conv_tree.select_sibling(node.id, -1)
                        st.rerun()

                    # version badge
                    st.markdown(f"<span class='ver'>{idx}/{total}</span>",
                                unsafe_allow_html=True)

                    # ▶ next
                    if st.button("▶", key=f"right_{node.id}"):
                        conv_tree.select_sibling(node.id, +1)
                        st.rerun()

                    # ✏️ edit
                    if st.session_state.app_mode != "conflict_resolution":
                        if st.button("✏️", key=f"edit_{node.id}"):
                            st.session_state.editing_msg_id = node.id
                            st.session_state.editing_content = node.content
                            st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)

                # ----- finally the message bubble itself -----------------
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:{align}; margin:4px 0 12px;'>
                      <div style='background-color:{bubble_color}; color:{text_color};
                                  padding:12px 16px; border-radius:18px;
                                  max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                                  font-size:16px; line-height:1.5;'>
                        <strong>{role_label}:</strong><br>{markdown_to_html(node.content)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return  # phone version done

            # ====== DESKTOP LAYOUT (unchanged from your original) =========
            col_left, col_center, col_right, col_edit, col_bubble = st.columns(
                [1.5, 1.5, 2, 6, 40], gap="small"
            )

            # ◀
            with col_left:
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin-top:{OFFSET_TOP}; transform:{TRANSFORM_LB};'>",
                    unsafe_allow_html=True)
                if st.button("◀", key=f"left_{node.id}"):
                    conv_tree.select_sibling(node.id, -1); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # 1/2
            with col_center:
                st.markdown(
                    f"""
                    <div style='display:flex; align-items:center; margin-top:{OFFSET_TOP_INDICATOR};
                                transform:{TRANSFORM_INDICATOR};'>{idx}/{total}</div>
                    """,
                    unsafe_allow_html=True)

            # ▶
            with col_right:
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin-top:{OFFSET_TOP}; transform:{TRANSFORM};'>",
                    unsafe_allow_html=True)
                if st.button("▶", key=f"right_{node.id}"):
                    conv_tree.select_sibling(node.id, +1); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            # Edit
            with col_edit:
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin-top:{OFFSET_TOP}; transform:{TRANSFORM};'>",
                    unsafe_allow_html=True)
            if st.session_state.app_mode != "conflict_resolution":
                    if st.button("Edit Message", key=f"edit_{node.id}"):
                        st.session_state.editing_msg_id = node.id
                        st.session_state.editing_content = node.content
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            # Bubble
            with col_bubble:
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:{align}; margin:8px 0;'>
                      <div style='background-color:{bubble_color}; color:{text_color};
                                  padding:12px 16px; border-radius:18px;
                                  max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                                  font-size:16px; line-height:1.5;'>
                        <strong>{role_label}:</strong><br>{markdown_to_html(node.content)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            return  # end “user w/versions” path

        # -------------------- 2b. trainee message WITHOUT versions --------
        if mobile:
            edit_col, bubble_col = st.columns([1, 10], gap="small")
            edit_label = "✏️"
        else:
            edit_col, bubble_col = st.columns([1, 9], gap="small")
            edit_label = "Edit Message"

        with edit_col:
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-top:{OFFSET_TOP};'>",
                unsafe_allow_html=True)
            if st.session_state.app_mode != "conflict_resolution":
                if st.button(edit_label, key=f"edit_{node.id}"):
                    st.session_state.editing_msg_id = node.id
                    st.session_state.editing_content = node.content
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with bubble_col:
            st.markdown(
                f"""
                <div style='display:flex; justify-content:{align}; margin:8px 0;'>
                  <div style='background-color:{bubble_color}; color:{text_color};
                              padding:12px 16px; border-radius:18px;
                              max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                              font-size:16px; line-height:1.5;'>
                    <strong>{role_label}:</strong><br>{markdown_to_html(node.content)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return  # end “user w/o versions” path

    # ------------------------------------------------------------------
    # 3) Assistant (simulated client) bubble
    # ------------------------------------------------------------------
    if node.role.startswith("assistant"):
        st.markdown(
            f"""
            <div style='display:flex; justify-content:{align}; margin:8px 0;'>
              <div style='background-color:{bubble_color}; color:{text_color};
                          padding:12px 16px; border-radius:18px;
                          max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                          font-size:16px; line-height:1.5;'>
                <strong>{role_label}:</strong><br>{markdown_to_html(node.content)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return


# --------------- TAB 2: Chat Session ---------------
with tab_chat:
    if sel_persona is None:
        st.info("Please generate or select a persona first in the *Persona Info* tab.")
    
    # --- CONFLICT RESOLUTION MODE LOGIC ---
    elif st.session_state.app_mode == "conflict_resolution":
        # 1. Render all existing messages
        for node in conv_tree.path_to_leaf()[1:]:
            render_msg(node, mobile=IS_MOBILE)

        # 2. Get the current conversation path for the arbiter
        current_path = conv_tree.path_to_leaf()
        history_for_arbiter = [{"role": n.role, "content": n.content} for n in current_path[1:]]
        last_speaker = history_for_arbiter[-1]['role'] if history_for_arbiter else "NONE"

        # 3. Determine who should speak next if we don't already know
        if 'next_speaker' not in st.session_state or st.session_state.next_speaker is None:
            st.session_state.next_speaker = get_next_speaker(
                API_KEY, 
                history_for_arbiter, 
                last_speaker, 
                config['turn_arbiter_prompt']
            )
            st.rerun()

        # 4. Handle AI turns
        next_speaker = st.session_state.get('next_speaker')
        if next_speaker in ["TEACHER", "PARENT"]:
            persona_to_use = sel_persona['teacher_persona'] if next_speaker == "TEACHER" else sel_persona['parent_persona']
            system_prompt_key = 'teacher_system_message_prompt' if next_speaker == "TEACHER" else 'parent_system_message_prompt'
            role_for_tree = 'assistant_teacher' if next_speaker == "TEACHER" else 'assistant_parent'
            
            spinner_msg = f"{persona_to_use['name']} ({next_speaker.title()}) is responding..."
            
            with st.spinner(spinner_msg):
                # We need a slightly modified get_ai_response call or structure
                # Let's adapt the call here directly for clarity
                system_msg = build_system_message(persona_to_use, None, config[system_prompt_key], config['system_message_details'])
                
                raw_msgs = build_prompt(conv_tree, system_msg)
                gemini_system_instruction = raw_msgs[0]["content"]
                history_for_gemini = [to_content("user" if m["role"] == "user" else "model", m["content"]) for m in raw_msgs[1:]]

                # In this multi-agent setup, we don't have a pending user message. The AI is responding to the whole context.
                # So we create a chat with the full history and ask it for a response.
                chat = client.chats.create(
                    model="gemini-2.5-flash",
                    history=history_for_gemini,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=gemini_system_instruction,
                        temperature=0.7
                    ),
                )
                response = chat.send_message("Based on the conversation so far, what is your response?") # Generic prompt to elicit a response

                ai_reply = response.text

            if ai_reply:
                new_node_id = conv_tree.add_node(conv_tree.current_leaf_id, role_for_tree, ai_reply)
                conv_tree.current_leaf_id = new_node_id

            # Crucially, reset the speaker to trigger the arbiter again
            st.session_state.next_speaker = None
            st.rerun()

        # 5. Handle Trainee's turn
        elif next_speaker == "TRAINEE":
            user_text = st.chat_input("Your turn to speak...")
            if user_text:
                new_user = conv_tree.add_node(conv_tree.current_leaf_id, "user", user_text)
                conv_tree.current_leaf_id = new_user
                
                # Reset speaker to trigger the arbiter for the next turn
                st.session_state.next_speaker = None
                st.rerun()

    # --- MODE LOGIC FOR OTHER MODES ---
    else:
        # Note: Your full render_msg function with version controls should be used here.
        # This is a simplified call for demonstration.
        for node in conv_tree.path_to_leaf()[1:]:
            render_msg(node, mobile=IS_MOBILE)

        # pending AI response
        if st.session_state.pending_user_node_id:
            with st.spinner("Client is responding..."):
                ai_reply = get_ai_response(
                    conv_tree, st.session_state.pending_user_node_id, sel_persona, sel_scenario,
                    config['system_message_prompt'], config['system_message_details']
                )
            if ai_reply is not None:
                new_assist = conv_tree.add_node(st.session_state.pending_user_node_id, "assistant", ai_reply)
                conv_tree.current_leaf_id = new_assist
            st.session_state.pending_user_node_id = None
            st.rerun()

        # chat input
        user_text = st.chat_input("Type your message here…")
        if user_text:
            new_user = conv_tree.add_node(conv_tree.current_leaf_id, "user", user_text)
            conv_tree.current_leaf_id = new_user
            st.session_state.pending_user_node_id = new_user
            st.rerun()

# ---------------- TAB 3 · Session Evaluation ------------------
def print_bubble(msg: dict[str, str]) -> None:
    role_label = "Evaluation Assistant" if msg["role"] == "assistant" else "Trainee (You)"
    align = "flex-start" if msg["role"] == "assistant" else "flex-end"
    bubble_color = "#1b222a" if msg["role"] == "assistant" else "#0e2a47"

    st.markdown(f"""
    <div style='display:flex; justify-content:{align}; margin:8px 0;'>
      <div style='background-color:{bubble_color}; color:white; padding:12px 16px; border-radius:18px; max-width:{BUBBLE_MAX}; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                                                font-size:16px; line-height:1.5;'>  
        <strong>{role_label}:</strong><br>{markdown_to_html(msg['content'])}
      </div>
    </div>""", unsafe_allow_html=True)

with tab_eval:
    if sel_persona is None:
        st.info("Please generate or select a persona first in the *Persona Info* tab.")
    else:
        branch = st.session_state.conv_tree.path_to_leaf()[1:]
        if not branch:
            st.info("No conversation to evaluate yet. Have a chat first.")
        else:
            branch_key = st.session_state.conv_tree.current_leaf_id

            # ── First-time evaluation ───────────────────────────────────────────
            if branch_key not in st.session_state.evaluation_feedback:
                if st.button("Evaluate Session"):
                    with st.spinner("Evaluating session…"):
                        history = [{"role": n.role, "content": n.content} for n in branch if n.role != "system"]
                        feedback, transcript = initial_evaluation(API_KEY, history, EVALUATION_PROMPT) # MODIFIED
                    st.session_state.evaluation_feedback[branch_key] = feedback
                    st.session_state.evaluation_transcript[branch_key] = transcript
                    st.session_state.evaluation_assistant_conversation[branch_key] = [{"role": "assistant", "content": feedback}]
                    st.rerun()
            
            # ── Display & follow-up Q & A ────────────────────────────────────────
            if branch_key in st.session_state.evaluation_feedback:
                qa = st.session_state.evaluation_assistant_conversation[branch_key]
                for msg in qa:
                    print_bubble(msg)
                
                if follow_up := st.chat_input("Ask the evaluation assistant…"):
                    qa.append({"role": "user", "content": follow_up})
                    with st.spinner("Evaluation assistant is thinking…"):
                        answer = supervisor_chat(API_KEY, st.session_state.evaluation_transcript[branch_key], qa, EVALUATION_PROMPT) # MODIFIED
                    qa.append({"role": "assistant", "content": answer})
                    st.rerun()

                if st.button("Re-evaluate this branch from scratch"):
                    st.session_state.evaluation_feedback.pop(branch_key, None)
                    st.session_state.evaluation_assistant_conversation.pop(branch_key, None)
                    st.rerun()

            # ── Download full transcript (+eval +Q&A) ────────────────────────────
            lines = []
            role_map = {
                "user": "Trainee",
                "assistant": "Client",
                "assistant_teacher": "Teacher",
                "assistant_parent": "Parent"
            }
            for n in branch:
                if n.role in role_map:
                    speaker = role_map[n.role]
                    # For conflict mode, add the name
                    if n.role == "assistant_teacher":
                        speaker = f"Teacher ({sel_persona['teacher_persona']['name']})"
                    elif n.role == "assistant_parent":
                        speaker = f"Parent ({sel_persona['parent_persona']['name']})"
                    lines.append(f"{speaker}: {n.content}")
            
            transcript = "\n\n".join(lines)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download Full Session Transcript (with Evaluation & Q&A)",
                data=transcript,
                file_name=f"{config['download_filename']}_{ts}.txt", # MODIFIED
            )

# -----------------------------------------------------------------------------
# 4.  __main__ guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pass
