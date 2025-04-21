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
# 0.  Low‑level config – Streamlit page & password gate
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Counsellor Training Chatbot", layout="wide")

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
PASSWORD = os.getenv("PASSWORD")
client = OpenAI(api_key=API_KEY)

if PASSWORD:  # optional simple password gate for demos
    if "authenticated" not in st.session_state:
        pw = st.text_input("Password", type="password")
        if pw == PASSWORD:
            st.session_state.authenticated = True
            st.experimental_rerun()
        st.stop()
    elif not st.session_state.authenticated:
        st.stop()

# -----------------------------------------------------------------------------
# 1.  Helper dataclasses & branching‑conversation tree
# -----------------------------------------------------------------------------
@dataclass
class MsgNode:
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    parent_id: Optional[str]
    created_at: datetime = field(default_factory=datetime.utcnow)


class ConvTree:
    """Tree structure + helpers for trace‑back modification & branch navigation."""

    def __init__(self):
        root_id = "root"
        self.nodes: Dict[str, MsgNode] = {
            root_id: MsgNode(
                id=root_id,
                role="system",
                content="Root of conversation tree",
                parent_id=None,
            )
        }
        self.children: Dict[str, List[str]] = {root_id: []}
        self.current_leaf_id: str = root_id

    # ---------- CRUD ----------
    def add_node(self, parent_id: str, role: str, content: str) -> str:
        nid = str(uuid.uuid4())
        node = MsgNode(id=nid, role=role, content=content, parent_id=parent_id)
        self.nodes[nid] = node
        self.children.setdefault(nid, [])
        self.children.setdefault(parent_id, []).append(nid)
        return nid

    # ---------- Traversal ----------
    def path_to_leaf(self, leaf_id: Optional[str] = None) -> List[MsgNode]:
        leaf_id = leaf_id or self.current_leaf_id
        path = []
        while leaf_id:
            path.append(self.nodes[leaf_id])
            leaf_id = self.nodes[leaf_id].parent_id
        return list(reversed(path))

    def children_of(self, node_id: str) -> List[str]:
        return self.children.get(node_id, [])

    def siblings(self, node_id: str) -> List[str]:
        p = self.nodes[node_id].parent_id
        return self.children.get(p, []) if p else []

    def sibling_index(self, node_id: str) -> int:
        sibs = self.siblings(node_id)
        return sibs.index(node_id) if node_id in sibs else -1

    def deepest_descendant(self, node_id: str) -> str:
        cursor = node_id
        while self.children.get(cursor):
            cursor = self.children[cursor][0]
        return cursor

    def select_sibling(self, node_id: str, direction: int) -> None:
        sibs = self.siblings(node_id)
        if len(sibs) <= 1:
            return
        idx = self.sibling_index(node_id)
        new_id = sibs[(idx + direction) % len(sibs)]
        self.current_leaf_id = self.deepest_descendant(new_id)

# -----------------------------------------------------------------------------
# 2.  Persona & prompt helpers
# -----------------------------------------------------------------------------
def load_personas() -> List[Dict]:
    with open("personas_v2.json") as f:
        return json.load(f)


def load_scenarios() -> Dict[str, List[Dict]]:
    with open("scenarios_v2.json") as f:
        return json.load(f)


personas = load_personas()
scenarios = load_scenarios()

# -----------------------------------------------------------------------------
# 3.  Streamlit side‑bars (persona, scenario, settings)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Select a Persona & Scenario")
    persona_names = [p["name"] for p in personas]
    persona_idx = st.selectbox("Persona", range(len(personas)), format_func=lambda i: persona_names[i])
    sel_persona = personas[persona_idx]

    scenario_titles = [s["title"] for s in scenarios[sel_persona["name"]]]
    scenario_idx = st.selectbox("Scenario", range(len(scenario_titles)), format_func=lambda i: scenario_titles[i])
    sel_scenario = scenarios[sel_persona["name"]][scenario_idx]

    # New session button
    if st.button("🔄  Start New Session"):
        st.session_state.clear()
        st.experimental_rerun()

# -----------------------------------------------------------------------------
# 4.  Session‑level state initialisation
# -----------------------------------------------------------------------------
if "conv_tree" not in st.session_state:
    st.session_state.conv_tree = ConvTree()
if "pending_user_node_id" not in st.session_state:
    st.session_state.pending_user_node_id = None
if "evaluation_feedback" not in st.session_state:
    st.session_state.evaluation_feedback: Dict[str, str] = {}
if "evaluation_assistant_conversation" not in st.session_state:
    # maps leaf‑id → list[messages]
    st.session_state.evaluation_assistant_conversation: Dict[str, List[Dict]] = {}

conv_tree: ConvTree = st.session_state.conv_tree  # type: ignore

# -----------------------------------------------------------------------------
# 4.1  Display intro (only first render)
# -----------------------------------------------------------------------------
if st.session_state.get("show_intro", True):
    st.markdown(
        """
        **Hi, welcome to this Counsellor Training Chatbot!**

        *Interact with the simulated client in the **“Counselling Session”** tab, then
        request a **Session Evaluation** and ask follow‑up questions in the **“Session Evaluation”** tab.*  
        """,
        unsafe_allow_html=True,
    )
    st.session_state.show_intro = False
    st.stop()

# -----------------------------------------------------------------------------
# 4.2  Helper for AI completion
# -----------------------------------------------------------------------------
def build_system_prompt(persona: Dict, scenario: Dict) -> str:
    base = f"""You are **{persona['name']}**, {persona['age']}‑year‑old {persona['occupation']}"""
    if scenario:
        base += f". Current concern: **{scenario['title']}** – {scenario.get('contextual_details', '')}"
    return base


def get_ai_response(tree: ConvTree, user_node_id: str, persona: Dict, scenario: Dict) -> str:
    prompt = [{"role": "system", "content": build_system_prompt(persona, scenario)}]
    for n in tree.path_to_leaf(user_node_id)[1:]:
        prompt.append({"role": n.role, "content": n.content})
    chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=prompt, temperature=0.7)
    return chat.choices[0].message.content.strip()


# -----------------------------------------------------------------------------
# 4.3  Little renderer for nodes
# -----------------------------------------------------------------------------
def render_msg(node: MsgNode):
    if node.role == "system":
        return
    with st.chat_message("assistant" if node.role == "assistant" else "user"):
        st.markdown(node.content)


# -----------------------------------------------------------------------------
# 4.4  Tabs
# -----------------------------------------------------------------------------
tab_persona, tab_chat, tab_eval = st.tabs(
    ["Persona Info", "Counselling Session", "Session Evaluation"]
)

# ---------- TAB 1 : Persona Info ----------
with tab_persona:
    st.subheader("Persona Details")
    for k in ["name", "age", "gender", "occupation", "main_issue", "background", "cultural_background"]:
        st.markdown(f"**{k.replace('_', ' ').title()}:** {sel_persona[k]}")
    st.write("---")
    st.markdown(f"**Scenario: {sel_scenario['title']}**")
    st.markdown(f"- **Emotional State:** {sel_scenario.get('emotional_state','')}")
    st.markdown(f"- **Context:** {sel_scenario.get('contextual_details','')}")

# ---------- TAB 2 : Counselling Session ----------
with tab_chat:
    for n in conv_tree.path_to_leaf()[1:]:
        render_msg(n)

    # Handle pending AI reply
    if st.session_state.pending_user_node_id:
        with st.spinner("Client is responding…"):
            reply = get_ai_response(conv_tree, st.session_state.pending_user_node_id, sel_persona, sel_scenario)
        new_id = conv_tree.add_node(conv_tree.current_leaf_id, "assistant", reply)
        conv_tree.current_leaf_id = new_id
        st.session_state.pending_user_node_id = None
        st.experimental_rerun()

    # Chat input
    user_txt = st.chat_input("Type your message here…")
    if user_txt:
        uid = conv_tree.add_node(conv_tree.current_leaf_id, "user", user_txt)
        conv_tree.current_leaf_id = uid
        st.session_state.pending_user_node_id = uid
        st.experimental_rerun()

# ---------- TAB 3 : Session Evaluation ----------
with tab_eval:
    branch = conv_tree.path_to_leaf()[1:]  # skip system root

    if not branch:
        st.info("No conversation to evaluate yet. Have a chat first.")
    else:
        # --- Ensure required per‑branch dictionaries exist ---
        branch_key = conv_tree.current_leaf_id
        if branch_key not in st.session_state.evaluation_feedback:
            st.session_state.evaluation_feedback[branch_key] = None
        if branch_key not in st.session_state.evaluation_assistant_conversation:
            st.session_state.evaluation_assistant_conversation[branch_key] = []

        # ----------------------------------------------------
        # 3‑A  ‑‑ Evaluate (or show existing evaluation)
        # ----------------------------------------------------
        if st.session_state.evaluation_feedback[branch_key] is None:
            if st.button("Evaluate Session"):
                with st.spinner("Evaluating session…"):
                    flat = [{"role": n.role, "content": n.content} for n in branch if n.role != "system"]
                    fb = evaluate_counselling_session(API_KEY, flat)
                    st.session_state.evaluation_feedback[branch_key] = fb
                st.success("Evaluation complete!")
                st.experimental_rerun()
        else:
            feedback = st.session_state.evaluation_feedback[branch_key]
            st.markdown(feedback)

            # ------------------------------------------------
            # 3‑B  ‑‑ Evaluation‑Assistant Q & A
            # ------------------------------------------------
            conv = st.session_state.evaluation_assistant_conversation[branch_key]
            for m in conv:
                with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
                    st.markdown(m["content"])

            q = st.chat_input("Ask the Evaluation Assistant about your session…", key=f"eval_input_{branch_key}_{len(conv)}")
            if q:
                with st.spinner("Evaluation Assistant is thinking…"):
                    context = (
                        [{"role": "assistant", "content": feedback}] + conv + [{"role": "user", "content": q}]
                    )
                    ans = evaluate_counselling_session(API_KEY, context)
                conv.extend(
                    [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": ans},
                    ]
                )
                st.experimental_rerun()

            # ------------------------------------------------
            # 3‑C  ‑‑ Re‑evaluate (clears assistant convo)
            # ------------------------------------------------
            if st.button("Re‑evaluate current conversation branch"):
                st.session_state.evaluation_feedback.pop(branch_key, None)
                st.session_state.evaluation_assistant_conversation.pop(branch_key, None)
                st.experimental_rerun()

        # ----------------------------------------------------
        # 3‑D  ‑‑ Transcript (incl. evaluation & Q & A)
        # ----------------------------------------------------
        def build_transcript(nodes: List[MsgNode]) -> str:
            txt = []
            for n in nodes:
                prefix = "Client" if n.role == "assistant" else "Trainee"
                txt.append(f"{prefix}: {n.content}")
            return "\n".join(txt)

        transcript = build_transcript(branch)
        if st.session_state.evaluation_feedback.get(branch_key):
            transcript += "\n\n--- Session Evaluation ---\n"
            transcript += st.session_state.evaluation_feedback[branch_key] + "\n"
            for m in st.session_state.evaluation_assistant_conversation.get(branch_key, []):
                role = "Evaluation Assistant" if m["role"] == "assistant" else "Trainee"
                transcript += f"{role}: {m['content']}\n"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download full session transcript (with Evaluation)",
            data=transcript,
            file_name=f"counselling_session_{ts}.txt",
        )

# -----------------------------------------------------------------------------
# 5.  __main__ guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pass