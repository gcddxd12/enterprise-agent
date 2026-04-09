# 多模态支持

## 概述
多模态支持是企业智能客服Agent的第三阶段优化成果，扩展了系统的输入类型，使其能够处理图像、文档（PDF、Word、Excel）等多种媒体格式。该模块通过模拟和真实技术结合，实现了图像理解、文档处理、多模态知识库等功能。

## 1. 多模态架构设计

### 1.1 媒体类型系统

**文件**: [multimodal_support.py](e:\my_multi_agent\multimodal_support.py) 第27-51行

**代码示例**:
```python
# multimodal_support.py 第27-51行：媒体类型定义
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
```

**设计要点**:
1. **枚举类型**: 明确定义支持的媒体类型
2. **数据类**: 统一的数据结构，便于序列化和处理
3. **元数据支持**: 扩展的元数据字段，支持灵活的信息存储

### 1.2 媒体类型检测器

**代码示例**:
```python
# multimodal_support.py 第53-109行：媒体类型检测器
class MediaDetector:
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
        # ... 其他类型判断
```

**检测策略**:
1. **MIME类型优先**: 使用标准MIME类型检测
2. **扩展名备用**: 扩展名作为后备检测方法
3. **逐步细化**: 从大类到具体类型的层级判断

## 2. 图像处理功能

### 2.1 图像处理器类

**文件**: [multimodal_support.py](e:\my_multi_agent\multimodal_support.py) 第111-172行

**代码示例**:
```python
# multimodal_support.py 第111-172行：图像处理器
class ImageProcessor:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def extract_text_from_image(self, image_path: str) -> str:
        """从图像中提取文字（模拟OCR）"""
        if not os.path.exists(image_path):
            return f"错误：图像文件不存在 - {image_path}"

        # 模拟OCR结果
        mock_ocr_results = {
            "screenshot.png": "这是一张截图，包含以下信息：用户名：张三，登录时间：2026-04-09 14:30...",
            "receipt.jpg": "购物小票\n商家：XX超市\n日期：2026-04-09...",
            "default": "图像内容识别：这是一张图片，包含文字和图形信息..."
        }
        # ... 匹配逻辑
```

**功能特点**:
1. **模拟OCR**: 针对常见场景的预定义文本提取
2. **图像描述**: 基于文件名的图像内容描述
3. **元数据提取**: 文件大小、格式、修改时间等基础信息

### 2.2 模拟与实际技术对比

**当前模拟实现**:
```python
# 模拟OCR返回预定义内容
def extract_text_from_image(self, image_path: str) -> str:
    filename = os.path.basename(image_path)
    for key in mock_ocr_results:
        if key in filename:
            return mock_ocr_results[key]
    return mock_ocr_results["default"]
```

**实际部署建议**:
```python
# 实际部署时可替换为以下技术
def real_ocr_extraction(image_path: str) -> str:
    # 使用Tesseract OCR
    import pytesseract
    from PIL import Image
    
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='chi_sim+eng')
    return text

def real_image_description(image_path: str) -> str:
    # 使用CLIP或类似模型
    from transformers import CLIPProcessor, CLIPModel
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # ... 处理逻辑
```

## 3. 文档处理功能

### 3.1 文档处理器类

**文件**: [multimodal_support.py](e:\my_multi_agent\multimodal_support.py) 第174-302行

**代码示例**:
```python
# multimodal_support.py 第174-302行：文档处理器
class DocumentProcessor:
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
版本：2.1.0...""",
            "api_documentation.pdf": """API文档
版本：v1.2...""",
            "default": f"""PDF文档内容：{os.path.basename(pdf_path)}
这是一份PDF文档，包含文本、图片和格式信息..."""
        }
        # ... 匹配逻辑
```

**支持格式**:
1. **PDF文档**: 用户手册、API文档等
2. **Word文档**: 商业计划书、需求规格说明书等
3. **Excel表格**: 销售报表、用户数据统计等

### 3.2 文档提取策略

**模拟实现**:
```python
# 基于文件名的内容匹配
filename = os.path.basename(pdf_path)
for key in mock_pdf_content:
    if key in filename:
        return mock_pdf_content[key]
return mock_pdf_content["default"]
```

**实际技术栈**:
```python
# 实际部署的文档处理库
def real_pdf_extraction(pdf_path: str) -> str:
    # 使用PyPDF2或pdfplumber
    import pdfplumber
    
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def real_docx_extraction(docx_path: str) -> str:
    # 使用python-docx
    from docx import Document
    
    doc = Document(docx_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def real_excel_extraction(excel_path: str) -> str:
    # 使用pandas或openpyxl
    import pandas as pd
    
    df = pd.read_excel(excel_path)
    return df.to_string()
```

## 4. 多模态知识库

### 4.1 知识库管理器

**文件**: [multimodal_support.py](e:\my_multi_agent\multimodal_support.py) 第304-447行

**代码示例**:
```python
# multimodal_support.py 第304-447行：多模态知识库管理器
class MultimodalKnowledgeBase:
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
        # 1. 检测媒体类型
        media_type = self.media_detector.detect_media_type(file_path)
        
        # 2. 创建媒体内容对象
        media_content = MediaContent(
            content_type=media_type,
            file_path=file_path,
            metadata={
                "added_time": self._get_current_timestamp(),
                "file_size": os.path.getsize(file_path)
            }
        )
        
        # 3. 提取文本内容
        text_content = self._extract_text_content(media_content)
        media_content.text_content = text_content
        
        # 4. 添加到索引
        self.media_index[file_path] = media_content
        
        # 5. 复制文件到存储目录
        dest_path = os.path.join(self.storage_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dest_path)
        
        return media_content
```

**知识库功能**:
1. **文件管理**: 媒体文件的存储、索引、检索
2. **内容提取**: 自动提取各种格式的文本内容
3. **元数据管理**: 时间戳、文件大小、格式等信息
4. **搜索能力**: 基于文本内容和文件名的搜索

### 4.2 多模态搜索算法

**代码示例**:
```python
# multimodal_support.py 第391-428行：多模态搜索
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
    
    # 文件名匹配（权重更高）
    if query in filename:
        score += 2.0
    
    # 文本内容匹配
    if query in text:
        score += 1.0
    
    # 匹配次数
    score += text.count(query) * 0.1
    
    return score
```

**搜索策略**:
1. **类型过滤**: 按媒体类型筛选结果
2. **文本匹配**: 在内容和文件名中搜索关键词
3. **相关性排序**: 基于匹配位置和次数的简单评分
4. **结果限制**: 可配置返回结果数量

## 5. 多模态工具集成

### 5.1 工具集合类

**文件**: [multimodal_support.py](e:\my_multi_agent\multimodal_support.py) 第449-542行

**代码示例**:
```python
# multimodal_support.py 第449-542行：多模态工具集合
class MultimodalTools:
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
```

**工具功能**:
1. **文件上传处理**: 统一的多模态文件处理入口
2. **内容搜索**: 跨媒体类型的统一搜索
3. **图像分析**: 集成OCR、描述、元数据提取
4. **文档处理**: 多种格式的文档内容提取

### 5.2 与LangGraph工作流集成

**演示文件**: [demo_multimodal_features.py](e:\my_multi_agent\demo_multimodal_features.py)

**代码示例**:
```python
# demo_multimodal_features.py 第19-126行：多模态功能演示
from langgraph_agent_with_memory import run_langgraph_agent_with_memory, get_memory_manager

def demo_multimodal_features():
    """演示多模态功能"""
    print("企业智能客服Agent - 多模态功能演示")
    print("第三阶段优化：多模态支持（图像理解、文档处理、文件上传）")
    
    # 演示场景1：图像分析
    result1 = run_langgraph_agent_with_memory("分析这张产品图片的内容", max_iterations=2)
    print(f"最终答案: {result1['final_answer'][:200]}...")
    print(f"使用的工具: {result1['plan']}")
    
    # 演示场景2：PDF文档处理
    result2 = run_langgraph_agent_with_memory("帮我提取这个PDF文档的内容", max_iterations=2)
    print(f"最终答案: {result2['final_answer'][:200]}...")
    print(f"使用的工具: {result2['plan']}")
    
    # 演示场景3：与现有功能集成
    result3 = run_langgraph_agent_with_memory("先分析这张截图，然后告诉我今天的日期", max_iterations=2)
    print(f"最终答案: {result3['final_answer'][:200]}...")
    print(f"使用的工具: {result3['plan']}")
```

**集成方式**:
1. **工具注册**: 多模态工具通过@tool装饰器注册到LangChain工具系统
2. **规划识别**: 规划节点自动识别多模态查询类型
3. **执行调用**: 执行节点调用相应的多模态工具
4. **记忆集成**: 多模态交互记录到对话历史和用户偏好

## 6. 测试与验证

### 6.1 模块自测试

**文件**: [multimodal_support.py](e:\my_multi_agent\multimodal_support.py) 第544-605行

**代码示例**:
```python
# multimodal_support.py 第544-605行：测试函数
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
        
        print("\n[SUCCESS] 多模态支持系统测试完成")
        return True
        
    except Exception as e:
        print(f"[FAILED] 测试失败: {e}")
        return False
```

**测试范围**:
1. **功能测试**: 各个多模态工具的基本功能
2. **集成测试**: 多模态工具与知识库的集成
3. **错误处理**: 文件不存在、格式不支持等异常情况
4. **性能测试**: 大文件处理、并发访问等（模拟）

### 6.2 集成测试

**文件**: [test_multimodal_integration.py](e:\my_multi_agent\test_multimodal_integration.py)

**代码示例**:
```python
# test_multimodal_integration.py 第13-86行：集成测试
def test_multimodal_queries():
    """测试多模态查询"""
    test_cases = [
        ("分析这张图片内容", "image_analysis"),
        ("我想处理一个PDF文档", "document_processing"),
        ("上传一个Word文件并分析", "file_upload_processing"),
        ("处理Excel表格", "document_processing"),
        ("查看截图内容", "image_analysis"),
    ]
    
    all_passed = True
    for query, expected_tool in test_cases:
        print(f"\n测试查询: '{query}'")
        print(f"期望工具: {expected_tool}")
        
        result = run_langgraph_agent_with_memory(query, max_iterations=2)
        
        # 检查是否使用了期望的工具
        plan = result['plan']
        if plan:
            tool_used = False
            for task in plan:
                if expected_tool in task:
                    tool_used = True
                    break
            
            if tool_used:
                print(f"[SUCCESS] 成功使用 {expected_tool} 工具")
            else:
                print(f"[FAILED] 未使用期望的 {expected_tool} 工具")
                all_passed = False
    
    return all_passed
```

**测试验证**:
1. **查询识别**: 测试系统是否能正确识别多模态查询
2. **工具调用**: 验证是否调用正确的多模态工具
3. **结果质量**: 检查返回结果的准确性和完整性
4. **记忆记录**: 验证多模态交互是否被正确记录

## 7. 技术架构总结

### 7.1 模块化设计优势

**分层架构**:
1. **基础层**: 媒体类型定义、检测器
2. **处理层**: 图像处理器、文档处理器
3. **存储层**: 多模态知识库
4. **工具层**: 多模态工具集合
5. **集成层**: LangGraph工作流集成

**可扩展性**:
1. **新格式支持**: 添加新的MediaType和处理类
2. **实际部署**: 替换模拟实现为真实技术栈
3. **功能扩展**: 添加新的多模态工具和功能

### 7.2 实际部署建议

**依赖包安装**:
```bash
# 图像处理
pip install Pillow pytesseract opencv-python
pip install transformers  # CLIP模型

# 文档处理
pip install pypdf2 pdfplumber python-docx openpyxl pandas

# 多模态模型
pip install torch torchvision
pip install sentence-transformers
```

**生产环境配置**:
```python
# 实际部署配置
class ProductionImageProcessor(ImageProcessor):
    def __init__(self):
        super().__init__(use_mock=False)
        # 初始化实际OCR和CV模型
        self.ocr_engine = pytesseract
        self.cv_model = load_clip_model()
    
    def extract_text_from_image(self, image_path: str) -> str:
        # 使用真实OCR引擎
        return self.ocr_engine.image_to_string(image_path)
    
    def describe_image(self, image_path: str) -> str:
        # 使用真实CV模型
        return self.cv_model.describe(image_path)
```

## 8. 学习总结

### 关键技术要点
1. **媒体类型系统**: 统一的类型定义和检测机制
2. **模拟实现策略**: 快速原型开发，支持平滑迁移到实际技术
3. **知识库设计**: 统一的多模态内容存储和检索
4. **工具集成模式**: 与现有LangChain/LangGraph生态的无缝集成

### 最佳实践
1. **渐进式实现**: 从模拟到实际，降低开发风险
2. **模块化设计**: 清晰的职责分离，便于维护和扩展
3. **全面测试**: 覆盖功能、集成、错误处理等多个维度
4. **文档完善**: 详细的API文档和使用示例

### 进阶方向
1. **真实技术集成**: 集成Tesseract OCR、CLIP模型等
2. **多模态嵌入**: 跨模态的向量表示和检索
3. **流式处理**: 支持大文件和实时处理
4. **分布式存储**: 海量多模态数据的分布式存储和管理

---

**相关文件**:
- [multimodal_support.py](e:\my_multi_agent\multimodal_support.py) - 多模态支持核心实现
- [demo_multimodal_features.py](e:\my_multi_agent\demo_multimodal_features.py) - 多模态功能演示
- [test_multimodal_integration.py](e:\my_multi_agent\test_multimodal_integration.py) - 多模态集成测试
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) - 工作流集成点

**下一步学习**: 工程化实践 →