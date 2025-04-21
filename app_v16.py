from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from evaluate_session_v2 import evaluate_counselling_session

# -----------------------------------------------------------------------------
# 0.  Low‑level conversation tree classes
# -----------------------------------------------------------------------------

@dataclass
class MsgNode:
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    parent_id: Optional[str]
    created_at: datetime = field(default_factory=datetime.utcnow)


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
        parent_id = self.nodes[node_id].parent_id
        if parent_id is None:
            return []
        return self.children[parent_id]

    def sibling_index(self, node_id: str) -> int:
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
        return {
            "nodes": {nid: node.__dict__ for nid, node in self.nodes.items()},
            "children": self.children,
            "current_leaf_id": self.current_leaf_id,
        }

    # ------------------------------------------------------------------
    # Legacy migration (flat list ➜ linear tree)
    # ------------------------------------------------------------------
    @classmethod
    def from_flat_history(cls, history: List[Dict[str, str]]) -> "ConvTree":
        tree = cls()
        parent = "root"
        for msg in history:
            parent = tree.add_node(parent, msg["role"], msg["content"])
        tree.current_leaf_id = parent
        return tree


# -----------------------------------------------------------------------------
# 1.  Environment & OpenAI client
# -----------------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
PASSWORD = os.getenv("PASSWORD")
client = OpenAI(api_key=API_KEY)

# -----------------------------------------------------------------------------
# 2.  Utility functions (personas, system message, prompt)
# -----------------------------------------------------------------------------

def load_personas():
    with open("personas_v2.json", "r") as f:
        return json.load(f)


def build_system_message(persona, scenario=None):
    base_prompt = f"""
    You are a simulated client in a counselling session.
    Your name is {persona['name']}, you are {persona['age']} years old, and you work as a {persona['occupation']}.
    Your main issue: {persona['main_issue']}.
    Background: {persona['background']}.
    Cultural background: {persona['cultural_background']}.
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


def build_prompt(conv_tree: ConvTree, system_msg: str) -> List[Dict[str, str]]:
    msgs = [
        {"role": "developer", "content": system_msg},
    ]
    for node in conv_tree.path_to_leaf()[1:]:  # skip root
        if node.role in {"user", "assistant"}:
            msgs.append({"role": node.role, "content": node.content})
    return msgs


# -----------------------------------------------------------------------------
# 3.  OpenAI call wrapper
# -----------------------------------------------------------------------------

def get_ai_response(conv_tree: ConvTree, pending_user_node_id: str, persona, scenario):
    """Generate assistant reply for the path ending at *pending_user_node_id*."""
    system_msg = build_system_message(persona, scenario)
    conv_tree.current_leaf_id = pending_user_node_id  # set leaf before building prompt
    messages = build_prompt(conv_tree, system_msg)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
    )
    ai_content = response.choices[0].message.content
    return ai_content


# -----------------------------------------------------------------------------
# 4.  Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Counsellor Training Chatbot", layout="wide")

# ------------------ 4.1  Password gate ------------------

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Counsellor Training Chatbot")
        st.markdown(
            """
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
            """,
            unsafe_allow_html=True,
        )

        with st.form(key="password_form"):
            entered = st.text_input("Enter Password:", type="password")
            if st.form_submit_button("Submit"):
                if entered == PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Try again.")

    if not st.session_state.authenticated:
        st.stop()


check_password()

# ------------------ 4.2  Session‑state bootstrapping ------------------

if "conv_tree" not in st.session_state:
    st.session_state.conv_tree = ConvTree()

# legacy migration support (in case a previous flat history exists)
if "conversation_history" in st.session_state and st.session_state.conversation_history:
    st.session_state.conv_tree = ConvTree.from_flat_history(
        st.session_state.conversation_history
    )
    del st.session_state.conversation_history

conv_tree: ConvTree = st.session_state.conv_tree  # typed alias

if "editing_msg_id" not in st.session_state:
    st.session_state.editing_msg_id = None
if "editing_content" not in st.session_state:
    st.session_state.editing_content = ""
if "pending_user_node_id" not in st.session_state:
    st.session_state.pending_user_node_id = None

# ------------------ 4.3  Sidebar (persona & scenario) ------------------

all_personas = load_personas()
persona_keys = list(all_personas.keys())

st.sidebar.header("Session Setup")
sel_persona_key = st.sidebar.selectbox("Select a Persona:", persona_keys, index=0)
sel_persona = all_personas[sel_persona_key]

scenario_list = sel_persona.get("scenarios", [])
scenario_titles = [s["title"] for s in scenario_list]
sel_scenario = {}
if scenario_titles:
    sel_title = st.sidebar.selectbox("Select a Scenario:", scenario_titles)
    sel_scenario = next(s for s in scenario_list if s["title"] == sel_title)

if st.sidebar.button("Clear Conversation & Evaluation"):
    st.session_state.conv_tree = ConvTree()
    st.session_state.editing_msg_id = None
    st.session_state.editing_content = ""
    st.session_state.pending_user_node_id = None
    st.rerun()

# ------------------ 4.4  Tabs ------------------

tab_persona, tab_chat, tab_eval = st.tabs([
    "Persona Info",
    "Counselling Session",
    "Session Evaluation",
])

# --------------- TAB 1: Persona Info ---------------
with tab_persona:
    st.subheader("Persona Details")
    for k in ["name", "age", "gender", "occupation", "main_issue", "background", "cultural_background"]:
        st.markdown(f"**{k.replace('_', ' ').title()}:** {sel_persona[k]}")
    if sel_scenario:
        st.write("---")
        st.markdown(f"**Scenario: {sel_scenario['title']}**")
        st.markdown(f"- **Emotional State:** {sel_scenario.get('emotional_state', '')}")
        st.markdown(f"- **Context:** {sel_scenario.get('contextual_details', '')}")

# --------------- Helper: render one message bubble ---------------

def render_msg(node: MsgNode):
    role_label = (
        f"Client ({sel_persona['name']})" if node.role == "assistant" else "Trainee (You)"
    )
    # For user messages, align bubble to the right; for assistant, to the left.
    align = "flex-end" if node.role == "user" else "flex-start"
    bubble_color = "#0e2a47" if node.role == "user" else "#1b222a"
    text_color = "white"

    if node.role == "user":
        # If there are sibling messages, use a columns layout for the navigation arrows.
        sibs = conv_tree.siblings(node.id)
        if len(sibs) > 1:
            col_left, col_center, col_right, col_main = st.columns([1, 1, 1, 10])

            with col_left:
                # Center the "◀" button vertically.
                st.markdown(
                    """<div style="display:flex; align-items:center; height:100%;">""",
                    unsafe_allow_html=True
                )
                if st.button("◀", key=f"left_{node.id}"):
                    conv_tree.select_sibling(node.id, -1)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with col_center:
                # Center the "2/2" (or "idx/total") text vertically.
                current_idx = conv_tree.sibling_index(node.id) + 1
                total_sibs = len(conv_tree.siblings(node.id))
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; height:100%;">
                        {current_idx}/{total_sibs}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_right:
                # Center the "▶" button vertically.
                st.markdown(
                    """<div style="display:flex; align-items:center; height:100%;">""",
                    unsafe_allow_html=True
                )
                if st.button("▶", key=f"right_{node.id}"):
                    conv_tree.select_sibling(node.id, +1)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            # If no arrows are needed, use a full‐width container
            col_main = st.container()

        # When editing, show the text area (which will span full width below the message).
        if st.session_state.editing_msg_id == node.id:
            new_text = st.text_area(
                "Edit your message:",
                value=st.session_state.editing_content or node.content,
                key=f"textarea_{node.id}",
            )
            col_send, col_cancel = st.columns(2)
            with col_send:
                if st.button("Send", key=f"send_edit_{node.id}"):
                    parent_id = node.parent_id  # type: ignore
                    new_user_id = conv_tree.add_node(parent_id, "user", new_text)
                    with st.spinner("Client is responding..."):
                        ai_reply = get_ai_response(conv_tree, new_user_id, sel_persona, sel_scenario)
                    new_assist_id = conv_tree.add_node(new_user_id, "assistant", ai_reply)
                    conv_tree.current_leaf_id = new_assist_id
                    st.session_state.editing_msg_id = None
                    st.session_state.editing_content = ""
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key=f"cancel_edit_{node.id}"):
                    st.session_state.editing_msg_id = None
                    st.session_state.editing_content = ""
                    st.rerun()
        with col_main:
            # Two columns: one for the Edit button (centered), one for the bubble.
            col_edit, col_bubble = st.columns([1, 9])
            with col_edit:
                st.markdown(
                    """<div style="display:flex; align-items:center; height:100%;">""",
                    unsafe_allow_html=True
                )
                if st.button("✎ Edit", key=f"edit_{node.id}"):
                    st.session_state.editing_msg_id = node.id
                    st.session_state.editing_content = node.content
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with col_bubble:
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: {align}; margin: 8px 0;'>
                        <div style='background-color: {bubble_color}; color: {text_color}; 
                                    padding: 12px 16px; border-radius: 18px; 
                                    max-width: 75%; box-shadow: 1px 1px 6px rgba(0,0,0,0.2); 
                                    font-size: 16px; line-height: 1.5;'>
                            <strong>{role_label}:</strong><br>{node.content}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        # For assistant messages, just render the bubble in a full-width container.
        st.markdown(
            f"""
            <div style='display: flex; justify-content: {align}; margin: 8px 0;'>
                <div style='background-color: {bubble_color}; color: {text_color}; 
                            padding: 12px 16px; border-radius: 18px; 
                            max-width: 75%; box-shadow: 1px 1px 6px rgba(0,0,0,0.2); 
                            font-size: 16px; line-height: 1.5;'>
                    <strong>{role_label}:</strong><br>{node.content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------- TAB 2: Chat Session ---------------
with tab_chat:
    # Render current conversation branch
    for node in conv_tree.path_to_leaf()[1:]:  # skip root
        render_msg(node)

    # -------------- Handle pending AI response --------------
    if st.session_state.pending_user_node_id:
        with st.spinner("Client is responding..."):
            ai_reply = get_ai_response(
                conv_tree,
                st.session_state.pending_user_node_id,
                sel_persona,
                sel_scenario,
            )
        new_assist_id = conv_tree.add_node(
            st.session_state.pending_user_node_id, "assistant", ai_reply
        )
        conv_tree.current_leaf_id = new_assist_id
        st.session_state.pending_user_node_id = None
        st.rerun()

    # -------------- Chat input --------------
    user_text = st.chat_input("Type your message here…")
    if user_text:
        # Append user node under current leaf
        new_user_id = conv_tree.add_node(conv_tree.current_leaf_id, "user", user_text)
        conv_tree.current_leaf_id = new_user_id
        st.session_state.pending_user_node_id = new_user_id
        st.rerun()

# --------------- TAB 3: Evaluation ---------------
with tab_eval:
    branch = conv_tree.path_to_leaf()[1:]  # skip root
    if not branch:
        st.info("No conversation to evaluate yet. Have a chat first.")
    else:
        if "evaluation_feedback" not in st.session_state:
            st.session_state.evaluation_feedback = {}
        branch_key = conv_tree.current_leaf_id  # use leaf id as cache key

        if branch_key not in st.session_state.evaluation_feedback:
            if st.button("Evaluate Session"):
                with st.spinner("Evaluating session…"):
                    flat_history = [
                        {"role": n.role, "content": n.content} for n in branch if n.role != "system"
                    ]
                    feedback = evaluate_counselling_session(API_KEY, flat_history)
                    st.session_state.evaluation_feedback[branch_key] = feedback
                st.success("Evaluation complete!")
                st.rerun()
        else:
            st.markdown(st.session_state.evaluation_feedback[branch_key])
            if st.button("Re‑evaluate current conversation branch"):
                del st.session_state.evaluation_feedback[branch_key]
                st.rerun()

    # -------------- Transcript download --------------
    def build_transcript(nodes: List[MsgNode]):
        lines = []
        for n in nodes:
            if n.role == "assistant":
                lines.append(f"Client: {n.content}")
            elif n.role == "user":
                lines.append(f"Trainee: {n.content}")
        return "\n".join(lines)

    transcript_text = build_transcript(branch)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="Download current conversation branch transcript",
        data=transcript_text,
        file_name=f"counselling_session_{ts}.txt",
    )


# -----------------------------------------------------------------------------
# 5.  __main__ guard (not strictly necessary for Streamlit but kept for clarity)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
