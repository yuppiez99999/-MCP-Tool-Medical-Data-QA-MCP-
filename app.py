# -*- coding: utf-8 -*-
"""
ModelScope Space 部署入口 — 医疗数据质量评估 MCP Tool
=====================================================
小X宝医疗黑客松参赛作品

部署方式：
  1. 在 ModelScope 创建 Space (Gradio 类型)
  2. 上传本文件 + mcp_server.py
  3. Space 自动部署，生成公开访问URL

本地运行：
  python app.py
  # 访问 http://localhost:7860
"""
import json
import gradio as gr
from mcp_server import MedicalDataQAMCPServer

# 初始化 MCP Server
server = MedicalDataQAMCPServer()


# ============================================================
# Gradio 界面回调函数
# ============================================================
def ui_assess_quality(input_json: str, department: str):
    """质量评估界面回调"""
    try:
        records = json.loads(input_json) if isinstance(input_json, str) else input_json
        if not isinstance(records, list):
            return "错误: 请输入JSON数组格式", ""
        dept = department if department != "自动检测" else None
        result = server.assess_data_quality(records, dept)
        return json.dumps(result, ensure_ascii=False, indent=2), ""
    except json.JSONDecodeError as e:
        return f"JSON解析错误: {e}", ""
    except Exception as e:
        return f"错误: {e}", ""


def ui_classify_department(input_json: str):
    """科室分类界面回调（JSON记录）"""
    try:
        record = json.loads(input_json) if isinstance(input_json, str) else input_json
        result = server.classify_department(record)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"错误: {e}"


def ui_classify_department_text(text: str):
    """科室分类界面回调（文本输入）"""
    try:
        record = {"text": text, "data_type": "text"}
        result = server.classify_department(record)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"错误: {e}"


def ui_grade_level(score: float):
    """等级评定界面回调"""
    result = server.grade_data_level(score)
    return json.dumps(result, ensure_ascii=False, indent=2)


def ui_generate_report(input_json: str, dataset_name: str):
    """报告生成界面回调"""
    try:
        records = json.loads(input_json) if isinstance(input_json, str) else input_json
        if not isinstance(records, list):
            return "错误: 请输入JSON数组格式"
        result = server.generate_quality_report(records, dataset_name)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"错误: {e}"


def ui_search_evidence(query: str, source: str, max_results: int):
    """医学循证检索界面回调"""
    try:
        if not query.strip():
            return "错误: 请输入检索关键词"
        source_map = {
            "英文论文": "paper_en",
            "中文论文": "paper_cn",
            "会议论文": "meeting",
            "临床指南": "guide",
            "临床试验": "trial",
            "药品说明书": "package_insert",
        }
        src = source_map.get(source, "paper_en")
        result = server.search_medical_evidence(query, src, max_results)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"错误: {e}"


def ui_assess_with_evidence(input_json: str, query: str, source: str, evidence_count: int, department: str):
    """质量评估+文献检索联动界面回调"""
    try:
        records = json.loads(input_json) if isinstance(input_json, str) else input_json
        if not isinstance(records, list):
            return "错误: 请输入JSON数组格式"
        source_map = {
            "英文论文": "paper_en",
            "中文论文": "paper_cn",
            "会议论文": "meeting",
            "临床指南": "guide",
            "临床试验": "trial",
            "药品说明书": "package_insert",
        }
        src = source_map.get(source, "paper_en")
        dept = department if department != "自动检测" else None
        result = server.assess_with_evidence(records, query, src, dept, evidence_count)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        return f"JSON解析错误: {e}"
    except Exception as e:
        return f"错误: {e}"


def ui_evidence_report(input_json: str, dataset_name: str, source: str, evidence_per_dim: int, department: str):
    """带文献引用的质量报告界面回调"""
    try:
        records = json.loads(input_json) if isinstance(input_json, str) else input_json
        if not isinstance(records, list):
            return "错误: 请输入JSON数组格式"
        source_map = {
            "英文论文": "paper_en",
            "中文论文": "paper_cn",
            "临床指南": "guide",
            "临床试验": "trial",
        }
        src = source_map.get(source, "paper_en")
        dept = department if department != "自动检测" else None
        result = server.generate_evidence_based_report(
            records, dataset_name, dept, evidence_per_dim, src
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        return f"JSON解析错误: {e}"
    except Exception as e:
        return f"错误: {e}"


# ============================================================
# 示例数据
# ============================================================
SAMPLE_RECORDS = json.dumps([
    {"completeness": 95, "accuracy": 92, "timeliness": 88, "compliance": 96,
     "data_type": "image", "department": "radiology"},
    {"completeness": 78, "accuracy": 85, "timeliness": 70, "compliance": 82,
     "data_type": "lab", "department": "laboratory"},
    {"completeness": 60, "accuracy": 65, "timeliness": 55, "compliance": 70,
     "data_type": "text", "department": "pediatrics"},
], ensure_ascii=False, indent=2)

SAMPLE_SINGLE = json.dumps({
    "completeness": 88, "accuracy": 90, "timeliness": 75, "compliance": 85,
    "data_type": "ecg", "department": "cardiology",
}, ensure_ascii=False, indent=2)


# ============================================================
# Gradio 界面
# ============================================================
def create_ui():
    """创建 Gradio 界面"""
    with gr.Blocks(
        title="医疗数据质量评估 MCP Tool",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🏥 医疗数据质量评估 MCP Tool

        > 小X宝医疗黑客松参赛作品 | 基于390万条医疗Token数据 | KnowS医学循证检索

        **功能**: 评估医疗数据质量、自动科室分类、数据等级评定、循证质量报告、医学循证文献检索

        **MCP工具**: 8个工具可被任何AI Agent调用，支持ModelScope部署
        """)

        with gr.Tab("📊 质量评估"):
            with gr.Row():
                with gr.Column():
                    dept_choice = gr.Dropdown(
                        ["自动检测", "放射科", "病理科", "神经内科", "心血管科",
                         "检验科", "骨科", "儿科", "急诊科"],
                        value="自动检测", label="指定科室（可选）",
                    )
                    input_json = gr.Textbox(
                        label="医疗数据记录 (JSON数组)",
                        value=SAMPLE_RECORDS, lines=12,
                    )
                    btn = gr.Button("🔍 评估质量", variant="primary")
                with gr.Column():
                    output = gr.Textbox(label="评估结果", lines=20)
            btn.click(ui_assess_quality, [input_json, dept_choice], [output, output])

        with gr.Tab("🔬 科室分类"):
            gr.Markdown(
                "### 🔬 科室自动分类\n"
                "支持两种模式：**ML模型**（基于文本内容，KnowS文献训练）"
                " 和 **规则引擎**（基于数据类型映射）"
            )
            with gr.Row():
                with gr.Column():
                    with gr.Tabs():
                        with gr.Tab("文本分类（ML模型）"):
                            text_input = gr.Textbox(
                                label="数据描述文本",
                                value="CT scan of brain showing acute ischemic stroke",
                                lines=4, placeholder="输入数据描述或元信息...",
                            )
                            btn2a = gr.Button("🏷️ ML模型分类", variant="primary")
                        with gr.Tab("记录分类（JSON）"):
                            single_input = gr.Textbox(
                                label="单条记录 (JSON)", value=SAMPLE_SINGLE, lines=6,
                            )
                            btn2 = gr.Button("🏷️ 规则引擎分类", variant="secondary")
                with gr.Column():
                    output2 = gr.Textbox(label="分类结果", lines=18)
            btn2.click(ui_classify_department, [single_input], [output2])
            btn2a.click(ui_classify_department_text, [text_input], [output2])

        with gr.Tab("📈 等级评定"):
            with gr.Row():
                with gr.Column():
                    score_input = gr.Slider(0, 100, value=85, step=0.5,
                                             label="综合质量分")
                    btn3 = gr.Button("🏅 评定等级", variant="primary")
                with gr.Column():
                    output3 = gr.Textbox(label="等级结果", lines=10)
            btn3.click(ui_grade_level, [score_input], [output3])

        with gr.Tab("📋 完整报告"):
            with gr.Row():
                with gr.Column():
                    ds_name = gr.Textbox(label="数据集名称", value="我的医疗数据集")
                    report_input = gr.Textbox(
                        label="数据记录 (JSON数组)", value=SAMPLE_RECORDS, lines=12,
                    )
                    btn4 = gr.Button("📄 生成报告", variant="primary")
                with gr.Column():
                    output4 = gr.Textbox(label="质量报告", lines=25)
            btn4.click(ui_generate_report, [report_input, ds_name], [output4])

        with gr.Tab("📚 医学循证检索"):
            with gr.Row():
                with gr.Column():
                    ev_search_query = gr.Textbox(
                        label="检索关键词",
                        value="MDS NPM1 mutation",
                        placeholder="输入疾病名、基因、药物等医学关键词",
                        lines=2,
                    )
                    with gr.Row():
                        ev_search_source = gr.Dropdown(
                            ["英文论文", "中文论文", "会议论文", "临床指南", "临床试验", "药品说明书"],
                            value="英文论文", label="数据源",
                        )
                        ev_search_max = gr.Slider(1, 40, value=10, step=1, label="返回数量")
                    btn5 = gr.Button("🔍 检索文献", variant="primary")
                with gr.Column():
                    output5 = gr.Textbox(label="检索结果", lines=20)
            btn5.click(ui_search_evidence, [ev_search_query, ev_search_source, ev_search_max], [output5])

        with gr.Tab("🔗 质量+文献联动"):
            with gr.Row():
                with gr.Column():
                    ev_dept = gr.Dropdown(
                        ["自动检测", "放射科", "病理科", "神经内科", "心血管科",
                         "检验科", "骨科", "儿科", "急诊科"],
                        value="自动检测", label="指定科室（可选）",
                    )
                    ev_input = gr.Textbox(
                        label="医疗数据记录 (JSON数组)",
                        value=SAMPLE_RECORDS, lines=8,
                    )
                    ev_query = gr.Textbox(
                        label="检索词（留空自动生成）",
                        value="", placeholder="不填则根据科室和质量自动生成",
                    )
                    with gr.Row():
                        ev_source = gr.Dropdown(
                            ["英文论文", "中文论文", "会议论文", "临床指南", "临床试验", "药品说明书"],
                            value="英文论文", label="文献数据源",
                        )
                        ev_count = gr.Slider(1, 20, value=5, step=1, label="返回文献数")
                    btn6 = gr.Button("🔍 评估+检索文献", variant="primary")
                with gr.Column():
                    output6 = gr.Textbox(label="评估+文献结果", lines=25)
            btn6.click(ui_assess_with_evidence,
                       [ev_input, ev_query, ev_source, ev_count, ev_dept],
                       [output6])

        with gr.Tab("📑 循证质量报告"):
            gr.Markdown(
                "### 📑 带循证文献引用的质量报告\n"
                "自动识别质量薄弱维度 → 针对每个维度检索医学文献 → 生成带引用的改进建议"
            )
            with gr.Row():
                with gr.Column():
                    erp_dataset = gr.Textbox(
                        label="数据集名称",
                        value="测试数据集",
                        placeholder="输入数据集名称",
                    )
                    erp_dept = gr.Dropdown(
                        ["自动检测", "放射科", "病理科", "神经内科", "心血管科",
                         "检验科", "骨科", "儿科", "急诊科"],
                        value="自动检测", label="指定科室（可选）",
                    )
                    erp_input = gr.Textbox(
                        label="医疗数据记录 (JSON数组)",
                        value=SAMPLE_RECORDS, lines=6,
                    )
                    with gr.Row():
                        erp_source = gr.Dropdown(
                            ["英文论文", "中文论文", "临床指南", "临床试验"],
                            value="英文论文", label="文献数据源",
                        )
                        erp_per_dim = gr.Slider(1, 5, value=2, step=1, label="每维度文献数")
                    btn7 = gr.Button("📊 生成循证质量报告", variant="primary")
                with gr.Column():
                    output7 = gr.Textbox(label="循证质量报告结果", lines=28)
            btn7.click(ui_evidence_report,
                       [erp_input, erp_dataset, erp_source, erp_per_dim, erp_dept],
                       [output7])

        with gr.Tab("🔧 MCP工具列表"):
            tools = server.list_tools()
            tool_info = "\n\n".join([
                f"### {t['name']}\n{t['description']}\n\n"
                f"**参数**:\n```json\n{json.dumps(t['inputSchema'], ensure_ascii=False, indent=2)}\n```"
                for t in tools
            ])
            gr.Markdown(f"## 可用MCP工具 ({len(tools)}个)\n\n{tool_info}")

    return demo


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
