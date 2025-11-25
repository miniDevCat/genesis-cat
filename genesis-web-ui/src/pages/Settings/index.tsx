import React, { useEffect, useState } from 'react'
import { Card, Descriptions, Tag, Spin, message } from 'antd'
import { genesisApi } from '@/services/api'

const Settings: React.FC = () => {
  const [deviceInfo, setDeviceInfo] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDeviceInfo()
  }, [])

  const loadDeviceInfo = async () => {
    try {
      const response = await genesisApi.getDeviceInfo()
      setDeviceInfo(response.device)
    } catch (error) {
      message.error('加载设备信息失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Card title="系统设置" bordered={false}>
        {loading ? (
          <Spin />
        ) : (
          <Descriptions column={1} bordered>
            <Descriptions.Item label="设备">
              <Tag color={deviceInfo?.device === 'cuda' ? 'green' : 'orange'}>
                {deviceInfo?.device?.toUpperCase() || 'CPU'}
              </Tag>
            </Descriptions.Item>
            {deviceInfo?.device_name && (
              <Descriptions.Item label="GPU 名称">
                {deviceInfo.device_name}
              </Descriptions.Item>
            )}
            <Descriptions.Item label="API 地址">
              http://localhost:5000/api
            </Descriptions.Item>
          </Descriptions>
        )}
      </Card>
    </div>
  )
}

export default Settings
