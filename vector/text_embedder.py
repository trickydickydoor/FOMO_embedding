import json
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from .text_splitter import TextSplitter

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("警告: Google Generative AI客户端未安装，请运行: pip install google-generativeai")

class TextEmbedder:
    """处理文本embedding的生成 - 使用Google Gemini API"""
    
    def __init__(self, config_file: str = 'embedding_config.json', log_callback: Optional[callable] = None):
        """
        初始化文本embedding处理器
        
        Args:
            config_file: 配置文件路径
            log_callback: 日志回调函数
        """
        self.config_file = config_file
        self.log_callback = log_callback or print
        self.client = None
        self.model_name = "models/embedding-001"
        self.batch_size = 100
        
        # 文本处理设置
        self.max_content_length = 2000
        self.enable_text_splitting = True
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # 初始化文本分割器
        if self.enable_text_splitting:
            self.text_splitter = TextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载配置并初始化客户端
        self._load_config()
    
    def _load_config(self) -> bool:
        """加载embedding配置"""
        if not GEMINI_AVAILABLE:
            self.log_callback("错误: Google Generative AI客户端未安装")
            return False
            
        try:
            # 尝试多个可能的配置文件路径
            possible_paths = [
                self.config_file,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config_file),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.config_file),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vector', self.config_file)
            ]
            
            config_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            # 如果没有找到配置文件，尝试从环境变量获取
            if not config_path:
                return self._load_from_env()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 获取Gemini配置
            gemini_config = config.get('gemini', {})
            api_key = gemini_config.get('api_key', '').strip()
            self.model_name = gemini_config.get('model_name', 'models/embedding-001')
            self.batch_size = gemini_config.get('batch_size', 100)
            
            # 获取文本处理设置
            embedding_settings = config.get('embedding_settings', {})
            text_prep = embedding_settings.get('text_preparation', {})
            self.max_content_length = text_prep.get('max_content_length', 2000)
            self.enable_text_splitting = text_prep.get('enable_text_splitting', True)
            self.chunk_size = text_prep.get('chunk_size', 1000)
            self.chunk_overlap = text_prep.get('chunk_overlap', 200)
            
            # 重新初始化文本分割器
            if self.enable_text_splitting:
                self.text_splitter = TextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            
            if not api_key:
                self.log_callback("警告: Gemini API Key 为空")
                return False
            
            # 初始化Gemini客户端
            self.client = genai.Client(api_key=api_key)
            
            self.log_callback(f"Gemini API 已连接，使用模型: {self.model_name}")
            return True
            
        except json.JSONDecodeError as e:
            self.log_callback(f"警告: 配置文件格式错误: {e}")
            return False
        except Exception as e:
            self.log_callback(f"警告: 加载 Gemini 配置失败: {e}")
            return False
    
    def _load_from_env(self) -> bool:
        """从环境变量加载配置"""
        try:
            api_key = os.getenv('GEMINI_API_KEY', '').strip()
            self.model_name = os.getenv('GEMINI_MODEL', 'models/embedding-001')
            self.batch_size = int(os.getenv('GEMINI_BATCH_SIZE', '100'))
            
            # 从环境变量获取文本处理设置
            self.enable_text_splitting = os.getenv('ENABLE_TEXT_SPLITTING', 'true').lower() == 'true'
            self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
            self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
            
            # 重新初始化文本分割器
            if self.enable_text_splitting:
                self.text_splitter = TextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            
            if not api_key:
                self.log_callback("警告: 未找到 GEMINI_API_KEY 环境变量")
                return False
            
            # 初始化Gemini客户端
            self.client = genai.Client(api_key=api_key)
            
            self.log_callback(f"Gemini API 已从环境变量连接，使用模型: {self.model_name}")
            return True
            
        except Exception as e:
            self.log_callback(f"从环境变量加载配置失败: {e}")
            return False
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本，移除不必要的字符和格式
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def _prepare_text_for_embedding(self, news_item: Dict[str, Any]) -> str:
        """
        为embedding准备文本内容 - 只使用content字段
        
        Args:
            news_item: 新闻项数据
            
        Returns:
            准备好的文本
        """
        content = self._clean_text(news_item.get('content', ''))
        
        # 如果内容为空，返回空字符串
        if not content:
            return ""
        
        # 限制内容长度
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        return content
    
    def _split_into_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """
        将列表分割成批次
        
        Args:
            items: 要分割的列表
            batch_size: 批次大小
            
        Returns:
            批次列表
        """
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches
    
    def generate_embeddings(self, news_items: List[Dict[str, Any]], 
                          retry_attempts: int = 3) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        为新闻项生成embeddings（支持文本分割）
        
        Args:
            news_items: 新闻项列表
            retry_attempts: 重试次数
            
        Returns:
            (embeddings列表, 对应的chunk信息列表)
        """
        if not self.client or not news_items:
            return [], []
        
        # 准备文本块
        all_chunks = []
        for news_idx, item in enumerate(news_items):
            content = self._clean_text(item.get('content', ''))
            
            if not content:
                continue
                
            if self.enable_text_splitting:
                # 分割文本
                chunks = self.text_splitter.split_text(content)
                
                for chunk_idx, chunk_info in enumerate(chunks):
                    all_chunks.append({
                        'text': chunk_info['text'],
                        'news_item': item,
                        'news_index': news_idx,
                        'chunk_index': chunk_idx,
                        'chunk_id': f"{item.get('id', 'unknown')}_{chunk_idx}",
                        'line_start': chunk_info['line_start'],
                        'line_end': chunk_info['line_end'],
                        'char_start': chunk_info['char_start'],
                        'char_end': chunk_info['char_end']
                    })
            else:
                # 不分割，整篇处理
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "..."
                
                all_chunks.append({
                    'text': content,
                    'news_item': item,
                    'news_index': news_idx,
                    'chunk_index': 0,
                    'chunk_id': f"{item.get('id', 'unknown')}_full",
                    'line_start': 1,
                    'line_end': content.count('\n') + 1,
                    'char_start': 0,
                    'char_end': len(content)
                })
        
        if not all_chunks:
            self.log_callback("没有可处理的文本块")
            return [], []
        
        # 提取要向量化的文本
        texts_to_embed = [chunk['text'] for chunk in all_chunks]
        
        # 分批处理
        batches = self._split_into_batches(texts_to_embed, self.batch_size)
        all_embeddings = []
        successful_chunks = []
        
        chunk_idx = 0
        for batch_idx, batch_texts in enumerate(batches):
            self.log_callback(f"处理批次 {batch_idx + 1}/{len(batches)} ({len(batch_texts)} 个文本块)")
            
            for attempt in range(retry_attempts):
                try:
                    # 调用Gemini API
                    embeddings_batch = []
                    for text in batch_texts:
                        result = self.client.models.embed_content(
                            model=self.model_name,
                            contents=text
                        )
                        embeddings_batch.append(result.embeddings[0].values)
                    
                    # 记录成功的结果
                    for i, embedding in enumerate(embeddings_batch):
                        all_embeddings.append(embedding)
                        successful_chunks.append(all_chunks[chunk_idx + i])
                    
                    chunk_idx += len(batch_texts)
                    
                    self.log_callback(f"  ✅ 批次 {batch_idx + 1} 成功生成 {len(embeddings_batch)} 个embeddings (768维)")
                    break
                    
                except Exception as e:
                    self.log_callback(f"  ❌ 批次 {batch_idx + 1} 失败 (尝试 {attempt + 1}/{retry_attempts}): {e}")
                    
                    if attempt < retry_attempts - 1:
                        time.sleep(5 + (2 ** attempt))
                    else:
                        self.log_callback(f"  💀 批次 {batch_idx + 1} 最终失败，跳过")
                        chunk_idx += len(batch_texts)  # 跳过失败的块
        
        total_chunks = len(all_chunks)
        successful_count = len(all_embeddings)
        self.log_callback(f"Embedding生成完成: {successful_count}/{total_chunks} 个文本块成功")
        
        return all_embeddings, successful_chunks
    
    def generate_single_embedding(self, text: str, retry_attempts: int = 3) -> Optional[List[float]]:
        """
        为单个文本生成embedding
        
        Args:
            text: 输入文本
            retry_attempts: 重试次数
            
        Returns:
            embedding向量或None
        """
        if not self.client or not text:
            return None
        
        cleaned_text = self._clean_text(text)
        
        for attempt in range(retry_attempts):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=cleaned_text
                )
                
                return result.embeddings[0].values
                
            except Exception as e:
                self.log_callback(f"单个embedding生成失败 (尝试 {attempt + 1}/{retry_attempts}): {e}")
                
                if attempt < retry_attempts - 1:
                    time.sleep(5 + (2 ** attempt))
        
        return None
    
    def check_connection(self) -> bool:
        """
        检查Gemini API连接状态
        
        Returns:
            连接是否正常
        """
        try:
            if not self.client:
                return False
            
            # 测试生成一个简单的embedding
            test_embedding = self.generate_single_embedding("测试连接", retry_attempts=1)
            return test_embedding is not None and len(test_embedding) == 768
            
        except Exception as e:
            self.log_callback(f"连接检查失败: {e}")
            return False
    
    def estimate_cost(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        估算embedding的成本
        
        Args:
            news_items: 新闻项列表
            
        Returns:
            成本估算信息
        """
        if not news_items:
            return {'total_tokens': 0, 'estimated_cost_usd': 0}
        
        # 估算token数量（1个中文字符约等于1.5个token）
        total_chars = 0
        for item in news_items:
            text = self._prepare_text_for_embedding(item)
            total_chars += len(text)
        
        # 估算token数量
        estimated_tokens = int(total_chars * 1.5)
        
        # Gemini embedding的价格 (需要确认实际价格)
        price_per_1k_tokens = 0.00001
        estimated_cost = (estimated_tokens / 1000) * price_per_1k_tokens
        
        return {
            'total_items': len(news_items),
            'total_characters': total_chars,
            'estimated_tokens': estimated_tokens,
            'estimated_cost_usd': round(estimated_cost, 6),
            'model': self.model_name,
            'dimensions': 768,
            'provider': 'Google Gemini API',
            'note': '每分钟15个请求免费额度'
        }