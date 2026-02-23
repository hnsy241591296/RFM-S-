import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import plotly.graph_objects as go
import time

# ================= 1. 页面全局配置 =================
st.set_page_config(page_title="社交电商智能营销系统", page_icon="🛒", layout="wide")

# ================= 2. 左侧边栏 (控制中心) =================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.title("⚙️ 系统控制中心")
    
    # 尝试从本地保险箱 (.streamlit/secrets.toml) 读取 Key
    default_key = st.secrets.get("GEMINI_API_KEY", "")
    
    # 如果本地有 Key 就会自动填入，绝不暴露在代码中
    api_key = st.text_input(
        "🔑 Gemini API Key：", 
        value=default_key,
        type="password", 
        help="系统已尝试自动读取本地密钥"
    )
    
    st.write("---")
    st.write("### 🎛️ AI 生成参数调节")
    # 让用户自己选择生成的语气
    tone_option = st.selectbox(
        "📝 主打文案语气",
        ("真诚关怀风", "限时利益诱惑风", "社交身份认同风", "幽默搞笑风", "紧迫催促风")
    )
    # 控制 AI 的创造力 (Temperature)
    ai_temperature = st.slider("🔥 AI 创造力 (温度值)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.write("---")
    st.caption("👨‍💻 Developed for Data Science & AI Marketing")

# ================= 3. 主页面头部 =================
st.title("🛒 RFM-S 社交电商智能营销系统")
st.write("基于 RFM-S 聚类模型与 Gemini 大模型的自动化营销闭环")

# ================= 4. 核心功能区 (使用标签页分离功能) =================
tab1, tab2 = st.tabs(["🎯 单用户精准诊断 (精细化)", "🚀 批量自动化生成 (全量提效)"])

# ----------------- Tab 1: 单用户精细化诊断 -----------------
with tab1:
    st.write("### 🔍 搜索目标用户")
    
    # 读取原始大表用于单人查询
    @st.cache_data
    def load_data():
        return pd.read_csv("data/processed_rfms_data_full.csv")
    
    try:
        df = load_data()
        
        # 动态获取列名提示用户
        columns_list = df.columns.tolist()
        st.caption(f"提示：你的数据表包含这些列: {', '.join(columns_list)}")
        
        target_user_id = st.text_input("🎯 请输入需要诊断的用户 ID：", "")
        
        if st.button("✨ 生成单人专属策略"):
            if not api_key:
                st.warning("⚠️ 请先在左侧边栏输入 API Key！")
            elif not target_user_id:
                st.warning("⚠️ 请输入用户 ID！")
            else:
                with st.spinner("正在检索数据并呼叫 Gemini 大模型..."):
                    # 假设第一列就是 ID 列
                    id_column_name = df.columns[0] 
                    user_data = df[df[id_column_name].astype(str) == str(target_user_id)]
                    
                    if user_data.empty:
                        st.error("🤷‍♂️ 查无此人，请检查 ID 是否正确。")
                    else:
                        user_dict = user_data.iloc[0].to_dict()
                        user_profile_str = "\n".join([f"- **{k}**: {v}" for k, v in user_dict.items()])
                        
                        with st.expander("✅ 成功提取到以下用户特征 (点击展开查看)"):
                            st.markdown(user_profile_str)
                        
                        # --- 绘制雷达图 ---
                        st.write("#### 🎯 用户 RFM-S 核心特征雷达图")
                        radar_categories = ['最近消费(R)', '消费频次(F)', '消费金额(M)', '社交裂变(S)']
                        
                        # 安全提取 RFMS 值 (请根据你的实际列名修改里面的字母)
                        radar_values = [
                            user_dict.get('R', 0), 
                            user_dict.get('F', 0), 
                            user_dict.get('M', 0), 
                            user_dict.get('S', 0)
                        ]
                        radar_values.append(radar_values[0])
                        radar_categories.append(radar_categories[0])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=radar_values, theta=radar_categories, fill='toself', name=f'用户 {target_user_id}', line_color='#FF4B4B'
                        ))
                        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, margin=dict(l=40, r=40, t=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # --- 呼叫大模型 (强制 JSON 输出) ---
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json", "temperature": ai_temperature})
                        
                        prompt = f"""
                        你是一位私域运营专家。基于以下用户的真实 RFM-S 特征数据生成一个JSON对象，必须包含以下三个字段：
                        1. "诊断分析"：(字符串) 简要分析该用户的消费潜力和社交裂变(S)价值，判断其痛点。
                        2. "触达策略"：(字符串) 针对该用户的特性，建议最佳的触达渠道（如私信、朋友圈、社群）和沟通重点。
                        3. "营销文案"：(数组) 提供3条【{tone_option}】的裂变文案，要求必须带有强烈的“分享给好友”动作引导。
                        
                        【用户数据】：
                        {user_profile_str}
                        """
                        try:
                            response = model.generate_content(prompt)
                            result_dict = json.loads(response.text)
                            
                            st.success("🎉 AI 深度诊断与策略生成完毕！")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info("🧠 **AI 深度诊断分析**")
                                st.write(result_dict.get("诊断分析", "暂无分析"))
                            with col2:
                                st.warning("🎯 **针对性触达策略**")
                                st.write(result_dict.get("触达策略", "暂无策略"))
                                
                            st.write("---")
                            st.write("### 💬 **专属 A/B 测试裂变文案库**")
                            copy_list = result_dict.get("营销文案", [])
                            for i, copy in enumerate(copy_list):
                                st.write(f"**策略文案 {i+1}：**")
                                st.code(copy, language="markdown")
                                
                        except Exception as e:
                            st.error(f"❌ AI 解析失败，请重试。报错详情：{e}")
                            
    except Exception as e:
        st.error(f"读取基础数据失败，请确认 data 文件夹下是否有 CSV 文件。报错详情：{e}")

# ----------------- Tab 2: 批量处理引擎 (带防崩溃保护) -----------------
with tab2:
    st.write("### 📁 批量用户名单上传")
    st.info("💡 提示：请上传包含用户数据特征的 CSV 文件。系统将为每位用户自动生成专属营销文案。")
    
    uploaded_file = st.file_uploader("选择一个 CSV 文件进行批量处理", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"✅ 成功读取文件，共包含 **{len(batch_df)}** 名用户数据。前3行预览：")
        st.dataframe(batch_df.head(3))
        
        st.write("---")
        # 截断保护机制：防止几十万条数据把免费 API 跑崩溃
        st.warning("⚠️ **生产环境安全锁**：考虑到大模型 API 的调用频率限制 (Rate Limit) 与前端网页超时问题，批量生成功能目前处于 Demo 演示模式。")
        process_limit = st.slider("请选择本次演示需要处理的用户数量 (安全阈值：1-20)：", min_value=1, max_value=20, value=3)
        
        if st.button("🚀 开始批量生成并导出"):
            if not api_key:
                st.error("⚠️ 请先在左侧边栏配置 API Key！")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"temperature": ai_temperature})
                
                # 关键：只截取用户选择的数量进行处理
                demo_df = batch_df.head(process_limit)
                total_rows = len(demo_df)
                
                for index, row in demo_df.iterrows():
                    user_str = ", ".join([f"{k}:{v}" for k, v in row.items()])
                    status_text.text(f"正在处理第 {index+1}/{total_rows} 个用户...")
                    
                    batch_prompt = f"针对这名电商用户[{user_str}]，写1条简短的微信私域裂变文案。语气要求：{tone_option}。直接输出文案内容，不要废话。"
                    
                    try:
                        resp = model.generate_content(batch_prompt)
                        copy_text = resp.text.strip()
                    except Exception as e:
                        copy_text = "⚠️ 触发 API 频率限制，生成失败"
                        
                    row_data = row.to_dict()
                    row_data['AI专属文案'] = copy_text
                    results.append(row_data)
                    
                    # 更新进度条
                    progress_bar.progress((index + 1) / total_rows)
                    # 强制休眠 4 秒，严格避免触发 Gemini 免费版的 RPM (每分钟请求数) 限制
                    time.sleep(4) 
                
                status_text.text("🎉 演示数据处理完成！")
                
                result_df = pd.DataFrame(results)
                st.write("### ✨ 批量生成结果预览")
                st.dataframe(result_df)
                
                # 将 dataframe 转为 csv 格式的二进制数据，供用户下载
                csv_data = result_df.to_csv(index=False).encode('utf-8-sig')
                
                st.download_button(
                    label="📥 下载带有 AI 文案的演示表格 (CSV)",
                    data=csv_data,
                    file_name="AI_Marketing_Demo_Result.csv",
                    mime="text/csv",
                )