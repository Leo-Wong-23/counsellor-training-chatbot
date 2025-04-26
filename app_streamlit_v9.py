from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional

import streamlit as st
from streamlit_js_eval import streamlit_js_eval

from dotenv import load_dotenv
from openai import OpenAI


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
# 2a.  Utility functions (personas, system message, prompt)
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
        {"role": "system", "content": system_msg},
    ]
    for node in conv_tree.path_to_leaf()[1:]:  # skip root
        if node.role in {"user", "assistant"}:
            msgs.append({"role": node.role, "content": node.content})
    return msgs


def get_ai_response(conv_tree: ConvTree, pending_user_node_id: str, persona, scenario):
    """Generate assistant reply for the path ending at *pending_user_node_id*."""
    system_msg = build_system_message(persona, scenario)
    conv_tree.current_leaf_id = pending_user_node_id  # set leaf before building prompt
    messages = build_prompt(conv_tree, system_msg)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        max_tokens=5000,
        temperature=1,
    )
    ai_content = response.choices[0].message.content
    return ai_content


# -------------------------------------------------------------------------
# 2 · Evaluation helpers  (NEW)
# -------------------------------------------------------------------------
SUPERVISOR_PROMPT = """
You are a clinical supervisor providing feedback to a trainee counsellor (role: user) based on a session transcript
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
""".strip()


def initial_evaluation(api_key: str, counselling_history: list[dict]) -> tuple[str, str]:
    transcript = "\n".join(
        f"{'Client' if m['role']=='assistant' else 'Trainee'}: {m['content']}"
        for m in counselling_history
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user",   "content": transcript},
        ],
        max_tokens=1500,
        temperature=0.7,
    )
    feedback = response.choices[0].message.content
    return feedback, transcript


def supervisor_chat(api_key: str,
                    transcript: str,
                    chat_history: list[dict]) -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user",   "content": transcript},   # 〈NEW〉 keeps context
            *chat_history,
        ],
        max_tokens=1000,
        temperature=0.7,
    )
    return response.choices[0].message.content


# -----------------------------------------------------------------------------
# 3.  Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Counsellor Training Chatbot", layout="wide")

# --- Mobile / desktop switch ---
SCREEN_W = streamlit_js_eval(js_expressions="screen.width", key="w")
IS_MOBILE = bool(SCREEN_W and int(SCREEN_W) < 768)  # Bootstrap’s “md” breakpoint

# --- Mobile-only CSS helper ---
if IS_MOBILE:
    st.markdown(
        """
        <style>
        /* Keep st.columns compact on screens ≤ 640 px */
        @media (max-width: 640px){

            /* 1 · remove the row-gap that forces wrapping */
            div[data-testid="horizontalBlock"]{
                row-gap:0 !important;
            }

            /* 2 · make each column shrink-to-fit */
            div[data-testid="horizontalBlock"] > div[data-testid="column"]{
                flex:0 0 auto !important;
                width:auto !important;
                padding-left:4px !important;
                padding-right:4px !important;
            }

            /* 3 · hide any empty row (prevents the blank band) */
            div[data-testid="horizontalBlock"]:not(:has(button,span)){
                display:none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
# ------------------ 3.1  Password gate ------------------

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
            - Engage in real-time conversations with realistic client personas experiencing diverse psychological challenges (in the Counselling Session tab).
            - Receive personalized feedback to enhance counselling techniques, with interactive discussions for deeper understanding (in the Session Evaluation tab).<br><br>

            **Features in Development:**
            - User voice input and dynamic model voice output, like sending and receiving voice messages.
            - Dynamic persona generation, letting the AI generate a persona on the fly. <br><br>

            ***Safety & Privacy Statement:***
            This app is currently in development and serves as a demonstration tool only—it is not intended for real-world counselling or professional use. 
            No chat history or personal data are stored beyond the active session; they are erased once you close or refresh the page.

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


# ------------------ 3.2  Session‑state bootstrapping ------------------

if "conv_tree" not in st.session_state:
    st.session_state.conv_tree = ConvTree()

# legacy migration support (in case a previous flat history exists)
if "conversation_history" in st.session_state and st.session_state.conversation_history:
    st.session_state.conv_tree = ConvTree.from_flat_history(
        st.session_state.conversation_history
    )
    del st.session_state.conversation_history

conv_tree: ConvTree = st.session_state.conv_tree  # typed alias

# Trace‑back editing state
if "editing_msg_id" not in st.session_state:
    st.session_state.editing_msg_id = None
if "editing_content" not in st.session_state:
    st.session_state.editing_content = ""
if "pending_user_node_id" not in st.session_state:
    st.session_state.pending_user_node_id = None

# Session Evaluation state
if "evaluation_feedback" not in st.session_state:
    st.session_state.evaluation_feedback = {}
if "evaluation_assistant_conversation" not in st.session_state:
    st.session_state.evaluation_assistant_conversation = {}
if "evaluation_transcript" not in st.session_state: st.session_state.evaluation_transcript = {}

# ------------------ 3.3  Sidebar (persona & scenario) ------------------

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
    st.session_state.evaluation_feedback = {}
    st.session_state.evaluation_assistant_conversation = {}
    st.rerun()


# ------------------ 3.4  Tabs ------------------

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


# ---------------------------------------------------------------------------
# Helper: render one message bubble  (desktop + new mobile layout)
# ---------------------------------------------------------------------------
def render_msg(node: MsgNode, mobile: bool = False):
    # ----- generic style helpers ------------------------------------------
    role_label = (
        f"Client ({sel_persona['name']})" if node.role == "assistant"
        else "Trainee (You)"
    )
    align        = "flex-start" if node.role == "assistant" else "flex-end"
    bubble_color = "#1b222a"     if node.role == "assistant" else "#0e2a47"
    text_color   = "white"

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
                    ai_reply = get_ai_response(conv_tree, new_user_id,
                                               sel_persona, sel_scenario)
                new_assist_id = conv_tree.add_node(new_user_id, "assistant", ai_reply)
                conv_tree.current_leaf_id = new_assist_id
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
                                  max-width:75%; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                                  font-size:16px; line-height:1.5;'>
                        <strong>{role_label}:</strong><br>{node.content}
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
                                  max-width:75%; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                                  font-size:16px; line-height:1.5;'>
                        <strong>{role_label}:</strong><br>{node.content}
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
                              max-width:75%; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                              font-size:16px; line-height:1.5;'>
                    <strong>{role_label}:</strong><br>{node.content}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return  # end “user w/o versions” path

    # ------------------------------------------------------------------
    # 3) Assistant (simulated client) bubble – unchanged
    # ------------------------------------------------------------------
    st.markdown(
        f"""
        <div style='display:flex; justify-content:{align}; margin:8px 0;'>
          <div style='background-color:{bubble_color}; color:{text_color};
                      padding:12px 16px; border-radius:18px;
                      max-width:75%; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                      font-size:16px; line-height:1.5;'>
            <strong>{role_label}:</strong><br>{node.content}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------- TAB 2: Chat Session ---------------
with tab_chat:
    for node in conv_tree.path_to_leaf()[1:]:
        render_msg(node, mobile=IS_MOBILE)

    # pending AI response
    if st.session_state.pending_user_node_id:
        with st.spinner("Client is responding..."):
            ai_reply = get_ai_response(
                conv_tree,
                st.session_state.pending_user_node_id,
                sel_persona,
                sel_scenario,
            )
        new_assist = conv_tree.add_node(
            st.session_state.pending_user_node_id, "assistant", ai_reply
        )
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


# ---------------- TAB 3 · Session Evaluation (REPLACED) ------------------
# -----------------------------------------------------------------
# Tiny helper to show one chat bubble            〈NEW〉
# -----------------------------------------------------------------
def print_bubble(msg: dict[str, str]) -> None:
    """
    msg = {"role": "assistant" | "user", "content": str}
    • assistant  → Evaluation Assistant (left-aligned, #1b222a)
    • user       → Trainee (right-aligned, #0e2a47)
    """
    role_label   = "Evaluation Assistant" if msg["role"] == "assistant" else "Trainee (You)"
    align        = "flex-start"           if msg["role"] == "assistant" else "flex-end"
    bubble_color = "#1b222a"              if msg["role"] == "assistant" else "#0e2a47"
    text_color   = "white"

    st.markdown(
        f"""
        <div style='display:flex; justify-content:{align}; margin:8px 0;'>
          <div style='background-color:{bubble_color}; color:{text_color};
                      padding:12px 16px; border-radius:18px;
                      max-width:75%; box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                      font-size:16px; line-height:1.5;'>  
            <strong>{role_label}:</strong><br>{msg['content']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_eval:
    branch = st.session_state.conv_tree.path_to_leaf()[1:]
    if not branch:
        st.info("No conversation to evaluate yet. Have a chat first.")
        st.stop()

    branch_key = st.session_state.conv_tree.current_leaf_id

    # ── First-time evaluation ───────────────────────────────────────────
    if branch_key not in st.session_state.evaluation_feedback:
        if st.button("Evaluate Session"):
            with st.spinner("Evaluating session…"):
                counselling_history = [
                    {"role": n.role, "content": n.content}
                    for n in branch
                    if n.role != "system"
                ]

                # ⤵︎ unpack BOTH values returned by initial_evaluation
                feedback, transcript = initial_evaluation(API_KEY, counselling_history)

            # save to session_state
            st.session_state.evaluation_feedback[branch_key]               = feedback
            st.session_state.evaluation_transcript[branch_key]             = transcript
            st.session_state.evaluation_assistant_conversation[branch_key] = [
                {"role": "assistant", "content": feedback}
            ]
            st.rerun()

    # ── Display & follow-up Q & A ────────────────────────────────────────
    if branch_key in st.session_state.evaluation_feedback:

        qa = st.session_state.evaluation_assistant_conversation[branch_key]

        # chat bubbles – use your existing bubble-rendering helper
        for msg in qa:
            print_bubble(msg)          #   <-- whatever you used before

        follow_up = st.chat_input("Ask your supervisor…")
        if follow_up:
            qa.append({"role": "user", "content": follow_up})

            with st.spinner("Supervisor is thinking…"):
                answer = supervisor_chat(
                    API_KEY,
                    st.session_state.evaluation_transcript[branch_key],   # 〈NEW〉
                    qa
                )

            qa.append({"role": "assistant", "content": answer})
            st.session_state.evaluation_assistant_conversation[branch_key] = qa
            st.rerun()

        # optional: let the trainee scrap the feedback and start over
        if st.button("Re-evaluate this branch from scratch"):
            st.session_state.evaluation_feedback.pop(branch_key, None)
            st.session_state.evaluation_assistant_conversation.pop(branch_key, None)
            st.rerun()

    # ── Download full transcript (+eval +Q&A) ────────────────────────────
    lines = [
        ("Client"   if n.role == "assistant" else "Trainee") + ": " + n.content
        for n in branch
        if n.role in {"user", "assistant"}
    ]
    if branch_key in st.session_state.evaluation_assistant_conversation:
        for m in st.session_state.evaluation_assistant_conversation[branch_key]:
            who = "Supervisor" if m["role"] == "assistant" else "Trainee"
            lines.append(f"{who}: {m['content']}")

    transcript = "\n".join(lines)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "Download Full Session Transcript (with Evaluation & Q&A)",
        data=transcript,
        file_name=f"counselling_session_{ts}.txt",
    )

# -----------------------------------------------------------------------------
# 4.  __main__ guard
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pass
