import json
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("警告: Pinecone客户端未安装，请运行: pip install pinecone-client")

class PineconeHandler:
    """处理与 Pinecone 向量数据库的所有交互"""
    
    def __init__(self, config_file: str = 'embedding_config.json', log_callback: Optional[callable] = None):
        """
        初始化 Pinecone 处理器
        
        Args:
            config_file: 配置文件路径
            log_callback: 日志回调函数
        """
        self.config_file = config_file
        self.log_callback = log_callback or print
        self.client: Optional[Pinecone] = None
        self.index = None
        self.index_name: Optional[str] = None
        self.dimension: int = 1536  # OpenAI text-embedding-3-small的维度
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载配置并初始化客户端
        self._load_config()
    
    def _load_config(self) -> bool:
        """加载 Pinecone 配置"""
        if not PINECONE_AVAILABLE:
            self.log_callback("错误: Pinecone客户端未安装")
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
            
            # 获取Pinecone配置
            pinecone_config = config.get('pinecone', {})
            api_key = pinecone_config.get('api_key', '').strip()
            self.index_name = pinecone_config.get('index_name', 'fomo-news').strip()
            self.dimension = pinecone_config.get('dimension', 1536)
            
            if not api_key:
                self.log_callback("警告: Pinecone API Key 为空")
                return False
            
            # 初始化Pinecone客户端
            self.client = Pinecone(api_key=api_key)
            
            # 检查或创建索引
            self._ensure_index_exists()
            
            self.log_callback(f"Pinecone 已连接到索引: {self.index_name}")
            return True
            
        except json.JSONDecodeError as e:
            self.log_callback(f"警告: 配置文件格式错误: {e}")
            return False
        except Exception as e:
            self.log_callback(f"警告: 加载 Pinecone 配置失败: {e}")
            return False
    
    def _load_from_env(self) -> bool:
        """从环境变量加载配置"""
        try:
            api_key = os.getenv('PINECONE_API_KEY', '').strip()
            self.index_name = os.getenv('PINECONE_INDEX_NAME', 'fomo-news').strip()
            self.dimension = int(os.getenv('PINECONE_DIMENSION', '1536'))
            
            if not api_key:
                self.log_callback("警告: 未找到 PINECONE_API_KEY 环境变量")
                return False
            
            # 初始化Pinecone客户端
            self.client = Pinecone(api_key=api_key)
            
            # 检查或创建索引
            self._ensure_index_exists()
            
            self.log_callback(f"Pinecone 已从环境变量连接到索引: {self.index_name}")
            return True
            
        except Exception as e:
            self.log_callback(f"从环境变量加载配置失败: {e}")
            return False
    
    def _ensure_index_exists(self):
        """确保索引存在，如果不存在则创建"""
        try:
            # 获取现有索引列表
            existing_indexes = [index.name for index in self.client.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.log_callback(f"创建新的Pinecone索引: {self.index_name}")
                
                # 创建索引（使用Serverless规格）
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # 等待索引创建完成
                time.sleep(10)
            
            # 连接到索引
            self.index = self.client.Index(self.index_name)
            
        except Exception as e:
            self.log_callback(f"处理索引时出错: {e}")
            raise
    
    def _generate_vector_id(self, news_item: Dict[str, Any]) -> str:
        """
        为新闻项生成唯一的向量ID
        
        Args:
            news_item: 新闻项数据
            
        Returns:
            唯一的向量ID
        """
        # 使用URL和标题的组合生成稳定的ID
        content = f"{news_item.get('url', '')}-{news_item.get('title', '')}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def upsert_vectors(self, chunks: List[Dict[str, Any]], 
                      embeddings: List[List[float]]) -> List[str]:
        """
        向Pinecone插入或更新向量（支持文本块）
        
        Args:
            chunks: 文本块信息列表
            embeddings: 对应的embedding向量列表
            
        Returns:
            成功插入的向量ID列表
        """
        if not self.index or len(chunks) != len(embeddings):
            return []
        
        try:
            vectors = []
            vector_ids = []
            
            for chunk, embedding in zip(chunks, embeddings):
                news_item = chunk['news_item']
                vector_id = chunk['chunk_id']
                
                # 准备RAG所需的元数据（类似n8n格式）
                metadata = {
                    # 原文章信息
                    'article_title': news_item.get('title', ''),
                    'article_url': news_item.get('url', ''),
                    'article_published_time': news_item.get('published_at', ''),
                    'news_id': str(news_item.get('id', '')),
                    
                    # 文本块信息
                    'text': chunk['text'][:40000],  # Pinecone限制40KB，截断长文本
                    'chunk_index': chunk['chunk_index'],
                    
                    # 位置信息（类似n8n的loc.lines）
                    'loc.lines.from': chunk['line_start'],
                    'loc.lines.to': chunk['line_end'],
                    'loc.chars.from': chunk['char_start'],
                    'loc.chars.to': chunk['char_end'],
                    
                    # 其他有用信息
                    'chunk_length': len(chunk['text']),
                    'source': news_item.get('source', '36kr')
                }
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
                vector_ids.append(vector_id)
            
            # 批量插入向量
            batch_size = 100
            successful_ids = []
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    self.index.upsert(vectors=batch)
                    successful_ids.extend([v['id'] for v in batch])
                    self.log_callback(f"成功插入 {len(batch)} 个文本块向量到Pinecone")
                except Exception as e:
                    self.log_callback(f"批量插入失败: {e}")
            
            return successful_ids
            
        except Exception as e:
            self.log_callback(f"插入向量时出错: {e}")
            return []
    
    def query_similar(self, query_embedding: List[float], 
                     top_k: int = 10, 
                     filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        查询相似向量
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的最相似结果数量
            filter_dict: 过滤条件
            
        Returns:
            相似结果列表
        """
        if not self.index:
            return []
        
        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            results = []
            for match in response.matches:
                results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            return results
            
        except Exception as e:
            self.log_callback(f"查询向量时出错: {e}")
            return []
    
    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        删除指定的向量
        
        Args:
            vector_ids: 要删除的向量ID列表
            
        Returns:
            是否删除成功
        """
        if not self.index or not vector_ids:
            return False
        
        try:
            self.index.delete(ids=vector_ids)
            self.log_callback(f"成功删除 {len(vector_ids)} 个向量")
            return True
            
        except Exception as e:
            self.log_callback(f"删除向量时出错: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            索引统计信息
        """
        if not self.index:
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
            
        except Exception as e:
            self.log_callback(f"获取索引统计时出错: {e}")
            return {}
    
    def check_connection(self) -> bool:
        """
        检查Pinecone连接状态
        
        Returns:
            连接是否正常
        """
        try:
            if not self.client or not self.index:
                return False
            
            # 尝试获取索引统计信息来测试连接
            stats = self.get_index_stats()
            return len(stats) > 0
            
        except Exception as e:
            self.log_callback(f"连接检查失败: {e}")
            return False