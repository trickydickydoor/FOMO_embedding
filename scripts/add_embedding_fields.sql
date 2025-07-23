-- 添加新闻Embedding相关字段到news_items表
-- 执行时间：请在Supabase Dashboard中运行此脚本

BEGIN;

-- 添加embedding状态字段
ALTER TABLE news_items ADD COLUMN IF NOT EXISTS embedding_status TEXT DEFAULT 'pending';

-- 添加Pinecone向量ID字段  
ALTER TABLE news_items ADD COLUMN IF NOT EXISTS embedding_vector_id TEXT;

-- 添加embedding完成时间
ALTER TABLE news_items ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ;

-- 添加embedding模型信息
ALTER TABLE news_items ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'models/embedding-001';

-- 创建索引提高查询性能
CREATE INDEX IF NOT EXISTS idx_news_items_embedding_status ON news_items(embedding_status);
CREATE INDEX IF NOT EXISTS idx_news_items_embedding_vector_id ON news_items(embedding_vector_id);
CREATE INDEX IF NOT EXISTS idx_news_items_embedded_at ON news_items(embedded_at);

-- 添加注释说明字段用途
COMMENT ON COLUMN news_items.embedding_status IS 'Embedding状态: pending(待处理), processing(处理中), completed(已完成), failed(失败)';
COMMENT ON COLUMN news_items.embedding_vector_id IS 'Pinecone中的向量ID，用于关联向量数据';
COMMENT ON COLUMN news_items.embedded_at IS 'Embedding完成的时间戳';
COMMENT ON COLUMN news_items.embedding_model IS '使用的Embedding模型名称';

COMMIT;

-- 查看新表结构
\d news_items;