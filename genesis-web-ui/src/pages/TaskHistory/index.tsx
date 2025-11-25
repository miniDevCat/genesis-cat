import React from 'react'
import { Card, Empty, List, Image, Tag, Space } from 'antd'
import { ClockCircleOutlined } from '@ant-design/icons'
import { useGenerationStore } from '@/store/useGenerationStore'

const TaskHistory: React.FC = () => {
  const { generatedImages } = useGenerationStore()

  return (
    <div style={{ padding: 24 }}>
      <Card title="历史记录" bordered={false}>
        {generatedImages.length === 0 ? (
          <Empty description="暂无生成记录" />
        ) : (
          <List
            grid={{ gutter: 16, xs: 1, sm: 2, md: 3, lg: 4, xl: 4, xxl: 6 }}
            dataSource={generatedImages}
            renderItem={(item) => (
              <List.Item>
                <Card
                  hoverable
                  cover={
                    <Image
                      alt={item.prompt}
                      src={item.url}
                      style={{ height: 200, objectFit: 'cover' }}
                    />
                  }
                >
                  <Card.Meta
                    title={
                      <div style={{ fontSize: 12, height: 40, overflow: 'hidden' }}>
                        {item.prompt}
                      </div>
                    }
                    description={
                      <Space direction="vertical" size="small">
                        <div>
                          <Tag>{item.params.width}×{item.params.height}</Tag>
                          <Tag>Steps: {item.params.steps}</Tag>
                        </div>
                        <div style={{ fontSize: 12, color: '#999' }}>
                          <ClockCircleOutlined /> {new Date(item.timestamp).toLocaleString()}
                        </div>
                      </Space>
                    }
                  />
                </Card>
              </List.Item>
            )}
          />
        )}
      </Card>
    </div>
  )
}

export default TaskHistory
