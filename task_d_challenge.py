import streamlit as st
import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from PIL import Image

# ── 页面配置 ──
st.set_page_config(page_title="农业 AI 助手", page_icon="🌿", layout="wide")
st.title("🌿 农业 AI 知识问答系统")
st.caption("基于 RAG 技术，整合农业知识库与大语言模型")

# ── 侧边栏：配置 ──
st.sidebar.header("系统配置")

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    value=os.environ.get("AGICTO_API_KEY", "sk-enkOkhphKHVRFevBTMFER3RwDNlU6dwzBu8tqxNtniG6YHX9")
)

model_name = st.sidebar.selectbox("选择模型", ["qwen-plus", "gpt-4o-mini"])
top_k = st.sidebar.slider("检索文档数 (top_k)", 1, 5, 3)

base_url = "https://api.agicto.cn/v1"

# ── 初始化客户端 ──
@st.cache_resource
def init_clients(_api_key, _base_url):
    llm_client = OpenAI(api_key=_api_key, base_url=_base_url)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3"
    )

    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = chroma_client.get_collection(
            name="agri_knowledge",
            embedding_function=embedder
        )
    except Exception:
        collection = None

    return llm_client, collection

llm_client, collection = init_clients(api_key, base_url)

if collection is None:
    st.sidebar.warning("知识库未初始化，请先运行 Task B 构建知识库")

# ── 扩展功能 1：知识库管理 ──
st.sidebar.header("知识库管理")

uploaded_file = st.sidebar.file_uploader(
    "上传新的知识文档",
    type=["txt", "md"]
)

if uploaded_file is not None:
    os.makedirs("knowledge_base/user_uploads", exist_ok=True)

    save_path = os.path.join("knowledge_base/user_uploads", uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"已上传：{uploaded_file.name}")
    st.sidebar.info("上传文件已保存。若要参与检索，需要重新运行 Task B 建库流程。")

    try:
        preview_text = uploaded_file.getvalue().decode("utf-8")
        with st.sidebar.expander("预览上传内容"):
            st.text(preview_text[:500])
    except Exception:
        st.sidebar.warning("文件预览失败，但文件已保存。")

# ── 扩展功能 2：图像识别入口 ──
st.sidebar.header("番茄病害图像识别")

uploaded_image = st.sidebar.file_uploader(
    "上传番茄叶片图片",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.sidebar.image(image, caption="上传的图片", use_container_width=True)

    st.sidebar.info(
        "已完成图片上传入口。后续可在这里接入第4部分训练好的番茄病害识别模型。"
    )

# ── 检索函数 ──
def retrieve_knowledge(question, top_k=3):
    if collection is None:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    return list(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    )

# ── 问答函数：加入 RAG + 多轮对话 ──
def ask_with_rag(question, top_k=3):
    knowledge = retrieve_knowledge(question, top_k)

    # 取最近 6 条历史消息，避免上下文太长
    history_messages = st.session_state.messages[-6:]

    if not knowledge:
        system_prompt = "你是农业专家。请根据已有知识回答用户问题。如果不确定，请明确说明。"

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        messages.extend(history_messages)

        messages.append(
            {"role": "user", "content": question}
        )

        response = llm_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.5,
            max_tokens=1024
        )

        return response.choices[0].message.content, []

    # 构建 RAG 上下文
    context_parts = []
    for doc, meta, dist in knowledge:
        source = meta.get("source", "未知来源")
        context_parts.append(f"来源: {source}\n内容: {doc}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """你是农业植保专家。请基于以下参考资料回答用户的问题。

要求：
1. 回答必须优先依据参考资料
2. 如果资料不足以回答问题，请如实说明
3. 回答要实用、有针对性
4. 如果涉及农药、剂量或防治措施，应提醒用户结合当地植保部门建议和药剂标签使用
5. 允许参考最近几轮对话，但不能脱离参考资料随意编造"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    messages.extend(history_messages)

    messages.append({
        "role": "user",
        "content": f"参考资料:\n{context}\n\n当前问题: {question}"
    })

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=1024
    )

    return response.choices[0].message.content, knowledge

# ── 主界面：初始化历史消息 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── 清空对话按钮 ──
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# ── 显示历史消息 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 用户输入 ──
question = st.chat_input("请输入你的农业问题...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            if not api_key:
                answer = "请先在左侧输入 agicto API Key。"
                knowledge = []
            else:
                answer, knowledge = ask_with_rag(question, top_k)

            st.markdown(answer)

    # 显示参考来源
    if knowledge:
        with st.expander("📚 参考来源"):
            for i, (doc, meta, dist) in enumerate(knowledge):
                source = meta.get("source", "未知来源")
                relevance = 1 - dist

                st.markdown(
                    f"**来源 {i+1}**：{source} "
                    f"（相关度：{relevance:.2f}）"
                )
                st.text(doc[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── 知识库预览 ──
with st.sidebar.expander("📖 知识库文档列表"):
    if collection:
        count = collection.count()
        st.write(f"共 {count} 个文本块")
    else:
        st.write("当前未读取到 ChromaDB 知识库。")

# ── 说明 ──
with st.expander("系统说明"):
    st.markdown(
        """
        本系统整合了 Task B 中构建的 RAG 知识库，并通过 agicto 云端 API 调用大语言模型生成回答。

        已实现功能：
        - Streamlit Web 问答界面
        - ChromaDB 知识库检索
        - BGE-M3 向量模型
        - agicto 云端 API 调用
        - 参考来源展示
        - 多轮对话历史维护
        - 知识文档上传入口
        - 番茄病害图片上传入口

        注意：新上传的知识文档需要重新运行 Task B 建库流程后，才能被 RAG 系统检索到。
        """
    )