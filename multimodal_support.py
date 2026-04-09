#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态支持模块 - 企业智能客服Agent第三阶段优化

功能：
1. 图像理解（模拟OCR和图像描述）
2. 文档处理（PDF、Word、Excel模拟）
3. 多模态知识库检索
4. 多模态响应生成

注意：这是一个演示模块，实际部署时需要安装相应的依赖包

作者：gcddxd12
版本：1.0.0
创建日期：2026-04-09
"""

import os
import mimetypes
import base64
import json
from typing import Dict, Any, List, Optional, Union, BinaryIO
from dataclasses import dataclass
from enum import Enum

# ========== 多模态数据类型 ==========
class MediaType(Enum):
    """媒体类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    DOCX = "docx"
    EXCEL = "excel"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"

@dataclass
class MediaContent:
    """媒体内容数据类"""
    content_type: MediaType
    file_path: Optional[str] = None
    raw_data: Optional[bytes] = None
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# ========== 媒体类型检测 ==========
class MediaDetector:
    """媒体类型检测器"""

    @staticmethod
    def detect_media_type(file_path: str) -> MediaType:
        """检测文件类型"""
        if not os.path.exists(file_path):
            return MediaType.UNKNOWN

        # 获取MIME类型
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            # 通过扩展名判断
            ext = os.path.splitext(file_path)[1].lower()
            return MediaDetector._detect_by_extension(ext)

        # 根据MIME类型分类
        if mime_type.startswith('image/'):
            return MediaType.IMAGE
        elif mime_type == 'application/pdf':
            return MediaType.PDF
        elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                          'application/msword']:
            return MediaType.DOCX
        elif mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                          'application/vnd.ms-excel']:
            return MediaType.EXCEL
        elif mime_type.startswith('audio/'):
            return MediaType.AUDIO
        elif mime_type.startswith('video/'):
            return MediaType.VIDEO
        elif mime_type.startswith('text/'):
            return MediaType.TEXT
        else:
            return MediaType.UNKNOWN

    @staticmethod
    def _detect_by_extension(ext: str) -> MediaType:
        """通过扩展名检测类型"""
        image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
        doc_exts = ['.pdf', '.doc', '.docx', '.txt', '.rtf']
        excel_exts = ['.xls', '.xlsx', '.csv']

        if ext in image_exts:
            return MediaType.IMAGE
        elif ext in doc_exts:
            if ext == '.pdf':
                return MediaType.PDF
            elif ext in ['.doc', '.docx']:
                return MediaType.DOCX
            else:
                return MediaType.TEXT
        elif ext in excel_exts:
            return MediaType.EXCEL
        else:
            return MediaType.UNKNOWN

# ========== 图像处理（模拟） ==========
class ImageProcessor:
    """图像处理器（模拟版本）"""

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def extract_text_from_image(self, image_path: str) -> str:
        """从图像中提取文字（模拟OCR）"""
        if not os.path.exists(image_path):
            return f"错误：图像文件不存在 - {image_path}"

        # 模拟OCR结果
        mock_ocr_results = {
            "screenshot.png": "这是一张截图，包含以下信息：用户名：张三，登录时间：2026-04-09 14:30，IP地址：192.168.1.100",
            "receipt.jpg": "购物小票\n商家：XX超市\n日期：2026-04-09\n商品：苹果 5.00元，香蕉 3.50元，总计：8.50元",
            "document_scan.jpg": "重要文档扫描件\n标题：企业合作协议\n甲方：XX科技有限公司\n乙方：YY有限公司\n签署日期：2026-04-01",
            "default": "图像内容识别：这是一张图片，包含文字和图形信息。建议使用专业OCR工具进行精确识别。"
        }

        filename = os.path.basename(image_path)
        for key in mock_ocr_results:
            if key in filename:
                return mock_ocr_results[key]

        return mock_ocr_results["default"]

    def describe_image(self, image_path: str) -> str:
        """描述图像内容（模拟图像理解）"""
        if not os.path.exists(image_path):
            return f"错误：图像文件不存在 - {image_path}"

        # 模拟图像描述
        mock_descriptions = {
            "product_photo.png": "产品照片：展示企业版软件界面，包含仪表板、数据分析图表和用户管理模块。",
            "architecture_diagram.jpg": "系统架构图：展示微服务架构，包含API网关、认证服务、用户服务、数据服务等组件。",
            "team_photo.jpg": "团队合影：显示10名员工在办公室环境中，背景有白板和电脑设备。",
            "default": "图像描述：这是一张图片，包含视觉元素。建议使用计算机视觉模型进行详细分析。"
        }

        filename = os.path.basename(image_path)
        for key in mock_descriptions:
            if key in filename:
                return mock_descriptions[key]

        return mock_descriptions["default"]

    def extract_metadata(self, image_path: str) -> Dict[str, Any]:
        """提取图像元数据（模拟）"""
        import datetime

        return {
            "filename": os.path.basename(image_path),
            "size": os.path.getsize(image_path),
            "last_modified": datetime.datetime.fromtimestamp(
                os.path.getmtime(image_path)
            ).isoformat(),
            "format": os.path.splitext(image_path)[1][1:].upper(),
            "dimensions": "1920x1080",  # 模拟尺寸
            "color_space": "RGB",
            "dpi": "72"
        }

# ========== 文档处理（模拟） ==========
class DocumentProcessor:
    """文档处理器（模拟版本）"""

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF中提取文字（模拟）"""
        if not os.path.exists(pdf_path):
            return f"错误：PDF文件不存在 - {pdf_path}"

        # 模拟PDF内容
        mock_pdf_content = {
            "user_manual.pdf": """用户手册
产品名称：企业智能客服系统
版本：2.1.0
发布日期：2026-04-01

第一章：系统介绍
1.1 产品概述
企业智能客服系统是基于AI技术的智能客服解决方案...

第二章：安装部署
2.1 系统要求
操作系统：Windows Server 2016+ 或 Linux CentOS 7+...""",
            "api_documentation.pdf": """API文档
版本：v1.2
基础URL：https://api.company.com/v1

认证方式：
使用API密钥进行认证，在请求头中添加：
Authorization: Bearer {api_key}

接口列表：
1. GET /users - 获取用户列表
2. POST /tickets - 创建工单
3. PUT /tickets/{id} - 更新工单状态...""",
            "default": f"""PDF文档内容：{os.path.basename(pdf_path)}
这是一份PDF文档，包含文本、图片和格式信息。
实际部署时请使用pdfplumber或PyPDF2库进行精确文本提取。"""
        }

        filename = os.path.basename(pdf_path)
        for key in mock_pdf_content:
            if key in filename:
                return mock_pdf_content[key]

        return mock_pdf_content["default"]

    def extract_text_from_docx(self, docx_path: str) -> str:
        """从Word文档中提取文字（模拟）"""
        if not os.path.exists(docx_path):
            return f"错误：Word文档不存在 - {docx_path}"

        # 模拟Word内容
        mock_docx_content = {
            "business_plan.docx": """商业计划书
项目名称：企业智能客服系统

一、项目概述
1.1 项目背景
随着人工智能技术的发展，智能客服系统成为企业数字化转型的重要工具...

二、市场分析
2.1 市场规模
全球智能客服市场规模预计到2026年将达到200亿美元...""",
            "requirements_spec.docx": """需求规格说明书
项目：企业智能客服系统 V2.0

1. 功能需求
1.1 用户管理
- 支持多角色权限管理
- 支持单点登录(SSO)
- 支持用户行为日志记录...

2. 非功能需求
2.1 性能要求
- 系统响应时间 < 2秒
- 支持并发用户数 > 1000
- 可用性 > 99.9%...""",
            "default": f"""Word文档内容：{os.path.basename(docx_path)}
这是一份Microsoft Word文档，包含格式化文本、表格和图像。
实际部署时请使用python-docx库进行精确文本提取。"""
        }

        filename = os.path.basename(docx_path)
        for key in mock_docx_content:
            if key in filename:
                return mock_docx_content[key]

        return mock_docx_content["default"]

    def extract_text_from_excel(self, excel_path: str) -> str:
        """从Excel中提取文字（模拟）"""
        if not os.path.exists(excel_path):
            return f"错误：Excel文件不存在 - {excel_path}"

        # 模拟Excel内容
        mock_excel_content = {
            "sales_report.xlsx": """销售报表 - 2026年第一季度

| 产品名称 | 销售额(万元) | 同比增长 | 市场份额 |
|----------|--------------|----------|----------|
| 企业版   | 1500         | 25%      | 35%      |
| 专业版   | 900          | 15%      | 25%      |
| 基础版   | 600          | 10%      | 15%      |

总计：3000万元，同比增长18%""",
            "user_data.xlsx": """用户数据统计表

| 用户ID | 用户名 | 注册日期   | 最后登录   | 消费金额 |
|--------|--------|------------|------------|----------|
| 1001   | 张三   | 2026-01-15 | 2026-04-08 | 5000     |
| 1002   | 李四   | 2026-02-20 | 2026-04-09 | 3000     |
| 1003   | 王五   | 2026-03-10 | 2026-04-07 | 8000     |

总计用户数：1500，平均消费：4500""",
            "default": f"""Excel表格内容：{os.path.basename(excel_path)}
这是一份Microsoft Excel文件，包含工作表、公式和图表。
实际部署时请使用pandas或openpyxl库进行数据处理。"""
        }

        filename = os.path.basename(excel_path)
        for key in mock_excel_content:
            if key in filename:
                return mock_excel_content[key]

        return mock_excel_content["default"]

# ========== 多模态知识库 ==========
class MultimodalKnowledgeBase:
    """多模态知识库管理器"""

    def __init__(self, storage_dir: str = "./multimodal_kb"):
        self.storage_dir = storage_dir
        self.media_index = {}  # 文件路径 -> 媒体内容映射
        os.makedirs(storage_dir, exist_ok=True)

        # 初始化处理器
        self.image_processor = ImageProcessor(use_mock=True)
        self.doc_processor = DocumentProcessor(use_mock=True)
        self.media_detector = MediaDetector()

    def add_media_file(self, file_path: str) -> Optional[MediaContent]:
        """添加媒体文件到知识库"""
        if not os.path.exists(file_path):
            print(f"[ERROR] 文件不存在: {file_path}")
            return None

        try:
            # 检测媒体类型
            media_type = self.media_detector.detect_media_type(file_path)

            # 创建媒体内容对象
            media_content = MediaContent(
                content_type=media_type,
                file_path=file_path,
                metadata={
                    "added_time": self._get_current_timestamp(),
                    "file_size": os.path.getsize(file_path)
                }
            )

            # 提取文本内容
            text_content = self._extract_text_content(media_content)
            media_content.text_content = text_content

            # 添加到索引
            self.media_index[file_path] = media_content

            # 复制文件到存储目录
            dest_path = os.path.join(
                self.storage_dir,
                os.path.basename(file_path)
            )
            import shutil
            shutil.copy2(file_path, dest_path)

            print(f"[INFO] 媒体文件已添加到知识库: {file_path} ({media_type.value})")
            return media_content

        except Exception as e:
            print(f"[ERROR] 添加媒体文件失败: {e}")
            return None

    def _extract_text_content(self, media_content: MediaContent) -> str:
        """根据媒体类型提取文本内容"""
        if not media_content.file_path:
            return ""

        file_path = media_content.file_path

        if media_content.content_type == MediaType.IMAGE:
            # 图像：OCR + 描述
            ocr_text = self.image_processor.extract_text_from_image(file_path)
            description = self.image_processor.describe_image(file_path)
            return f"图像内容：\nOCR文字：{ocr_text}\n图像描述：{description}"

        elif media_content.content_type == MediaType.PDF:
            return self.doc_processor.extract_text_from_pdf(file_path)

        elif media_content.content_type == MediaType.DOCX:
            return self.doc_processor.extract_text_from_docx(file_path)

        elif media_content.content_type == MediaType.EXCEL:
            return self.doc_processor.extract_text_from_excel(file_path)

        elif media_content.content_type == MediaType.TEXT:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return "无法读取文本文件"

        else:
            return f"不支持的文件类型：{media_content.content_type.value}"

    def search_media(self, query: str, media_type: Optional[MediaType] = None) -> List[MediaContent]:
        """搜索媒体内容"""
        results = []

        for file_path, media_content in self.media_index.items():
            # 类型过滤
            if media_type and media_content.content_type != media_type:
                continue

            # 简单文本匹配
            if (query.lower() in media_content.text_content.lower() or
                query.lower() in os.path.basename(file_path).lower()):
                results.append(media_content)

        # 按相关性排序（简单实现）
        results.sort(key=lambda x: self._calculate_relevance(x, query), reverse=True)
        return results

    def _calculate_relevance(self, media_content: MediaContent, query: str) -> float:
        """计算相关性分数（简化版）"""
        text = media_content.text_content.lower()
        filename = os.path.basename(media_content.file_path or "").lower()
        query = query.lower()

        score = 0.0

        # 文件名匹配
        if query in filename:
            score += 2.0

        # 文本内容匹配
        if query in text:
            score += 1.0

        # 匹配次数
        score += text.count(query) * 0.1

        return score

    def get_media_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取媒体文件信息"""
        if file_path not in self.media_index:
            return None

        media_content = self.media_index[file_path]
        return {
            "file_path": media_content.file_path,
            "content_type": media_content.content_type.value,
            "metadata": media_content.metadata,
            "text_preview": media_content.text_content[:200] + "..." if media_content.text_content else ""
        }

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()

# ========== 多模态工具 ==========
class MultimodalTools:
    """多模态工具集合"""

    def __init__(self):
        self.knowledge_base = MultimodalKnowledgeBase()
        self.image_processor = ImageProcessor(use_mock=True)

    def process_uploaded_file(self, file_path: str) -> str:
        """处理上传的文件"""
        result = self.knowledge_base.add_media_file(file_path)
        if result:
            return f"文件已成功处理并添加到知识库：\n" \
                   f"文件名：{os.path.basename(file_path)}\n" \
                   f"类型：{result.content_type.value}\n" \
                   f"内容预览：{result.text_content[:100]}..."
        else:
            return "文件处理失败"

    def search_multimodal_content(self, query: str) -> str:
        """搜索多模态内容"""
        results = self.knowledge_base.search_media(query)

        if not results:
            return f"未找到与 '{query}' 相关的多媒体内容"

        response = f"找到 {len(results)} 个相关结果：\n\n"
        for i, media in enumerate(results[:3]):  # 最多显示3个结果
            response += f"{i+1}. 【{media.content_type.value}】 {os.path.basename(media.file_path or '')}\n"
            response += f"   内容：{media.text_content[:80]}...\n\n"

        return response

    def analyze_image(self, image_path: str) -> str:
        """分析图像内容"""
        if not os.path.exists(image_path):
            return f"错误：图像文件不存在 - {image_path}"

        # OCR文字提取
        ocr_text = self.image_processor.extract_text_from_image(image_path)

        # 图像描述
        description = self.image_processor.describe_image(image_path)

        # 元数据
        metadata = self.image_processor.extract_metadata(image_path)

        response = f"图像分析结果：\n"
        response += f"文件名：{os.path.basename(image_path)}\n"
        response += f"尺寸：{metadata['dimensions']}\n"
        response += f"格式：{metadata['format']}\n\n"
        response += f"文字识别(OCR)：\n{ocr_text}\n\n"
        response += f"图像描述：\n{description}"

        return response

    def extract_document_content(self, doc_path: str) -> str:
        """提取文档内容"""
        if not os.path.exists(doc_path):
            return f"错误：文档文件不存在 - {doc_path}"

        # 检测文档类型
        detector = MediaDetector()
        doc_type = detector.detect_media_type(doc_path)

        processor = DocumentProcessor(use_mock=True)

        if doc_type == MediaType.PDF:
            content = processor.extract_text_from_pdf(doc_path)
            doc_type_str = "PDF文档"
        elif doc_type == MediaType.DOCX:
            content = processor.extract_text_from_docx(doc_path)
            doc_type_str = "Word文档"
        elif doc_type == MediaType.EXCEL:
            content = processor.extract_text_from_excel(doc_path)
            doc_type_str = "Excel表格"
        elif doc_type == MediaType.TEXT:
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_type_str = "文本文件"
            except:
                content = "无法读取文本文件"
                doc_type_str = "文本文件"
        else:
            content = f"不支持的文件类型：{doc_type.value}"
            doc_type_str = "未知文件"

        response = f"{doc_type_str}内容提取：\n"
        response += f"文件名：{os.path.basename(doc_path)}\n"
        response += f"类型：{doc_type_str}\n\n"
        response += f"内容：\n{content[:500]}{'...' if len(content) > 500 else ''}"

        return response

# ========== 测试函数 ==========
def test_multimodal_support():
    """测试多模态支持功能"""
    print("=== 测试多模态支持系统 ===")

    try:
        # 创建测试文件
        test_dir = "./test_multimodal_files"
        os.makedirs(test_dir, exist_ok=True)

        # 创建测试文件
        test_files = {
            "test_image.png": "这是一张测试图片",
            "test_document.pdf": "这是一个测试PDF文档",
            "test_report.docx": "这是一个测试Word报告",
            "test_data.xlsx": "这是一个测试Excel数据表",
        }

        for filename, content in test_files.items():
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[INFO] 创建测试文件: {filepath}")

        # 测试多模态工具
        tools = MultimodalTools()

        print("\n1. 测试图像分析:")
        result1 = tools.analyze_image(os.path.join(test_dir, "test_image.png"))
        print(f"结果: {result1[:100]}...")

        print("\n2. 测试文档处理:")
        result2 = tools.extract_document_content(os.path.join(test_dir, "test_document.pdf"))
        print(f"结果: {result2[:100]}...")

        print("\n3. 测试文件上传处理:")
        result3 = tools.process_uploaded_file(os.path.join(test_dir, "test_report.docx"))
        print(f"结果: {result3}")

        print("\n4. 测试多模态搜索:")
        result4 = tools.search_multimodal_content("测试")
        print(f"结果: {result4}")

        # 清理测试文件
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

        print("\n[SUCCESS] 多模态支持系统测试完成")
        return True

    except Exception as e:
        print(f"[FAILED] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== 主入口 ==========
if __name__ == "__main__":
    print("多模态支持模块")
    print("功能: 图像理解、文档处理、多模态知识库")

    # 运行测试
    test_multimodal_support()