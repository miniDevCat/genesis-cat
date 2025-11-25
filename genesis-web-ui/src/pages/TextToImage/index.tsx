import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Input,
  Button,
  Slider,
  InputNumber,
  Select,
  Space,
  message,
  Progress,
  Image,
  Spin,
  Tag,
  Divider,
  Tooltip,
} from 'antd'
import {
  ThunderboltOutlined,
  ReloadOutlined,
  DownloadOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import { genesisApi, type GenerateParams } from '@/services/api'
import { useGenerationStore } from '@/store/useGenerationStore'

const { TextArea } = Input

const TextToImage: React.FC = () => {
  const store = useGenerationStore()
  const [progress, setProgress] = useState(0)
  const [statusText, setStatusText] = useState('')
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [taskPolling, setTaskPolling] = useState<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    return () => {
      if (taskPolling) {
        clearInterval(taskPolling)
      }
    }
  }, [taskPolling])

  const pollTaskStatus = (taskId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await genesisApi.getTaskStatus(taskId)
        const task = response.task

        setProgress(task.progress || 0)
        setStatusText(task.status)

        if (task.status === 'completed') {
          clearInterval(interval)
          setTaskPolling(null)
          store.setIsGenerating(false)
          
          if (task.result && task.result.image) {
            setCurrentImage(task.result.image)
            store.addGeneratedImage({
              id: taskId,
              url: task.result.image,
              prompt: store.prompt,
              params: {
                width: store.width,
                height: store.height,
                steps: store.steps,
                cfg_scale: store.cfgScale,
                seed: task.result.seed,
              },
              timestamp: Date.now(),
            })
            message.success('图像生成成功！')
          }
        } else if (task.status === 'failed') {
          clearInterval(interval)
          setTaskPolling(null)
          store.setIsGenerating(false)
          message.error(`生成失败: ${task.error || '未知错误'}`)
        }
      } catch (error) {
        console.error('Poll error:', error)
      }
    }, 1000)

    setTaskPolling(interval)
  }

  const handleGenerate = async () => {
    if (!store.prompt.trim()) {
      message.warning('请输入提示词')
      return
    }

    try {
      store.setIsGenerating(true)
      setProgress(0)
      setStatusText('提交任务...')

      const params: GenerateParams = {
        prompt: store.prompt,
        negative_prompt: store.negativePrompt,
        width: store.width,
        height: store.height,
        steps: store.steps,
        cfg_scale: store.cfgScale,
        seed: store.seed,
        sampler: store.sampler,
        scheduler: store.scheduler,
      }

      const response = await genesisApi.submitTask({
        task_type: 'text_to_image',
        params,
      })
      
      if (response.success) {
        store.setCurrentTaskId(response.task_id)
        message.info('任务已提交，开始生成...')
        pollTaskStatus(response.task_id)
      } else {
        throw new Error('任务提交失败')
      }
    } catch (error: any) {
      store.setIsGenerating(false)
      message.error(`生成失败: ${error.message || '未知错误'}`)
    }
  }

  const handleRandomSeed = () => {
    store.setSeed(Math.floor(Math.random() * 2147483647))
  }

  const presetSizes = [
    { label: '512×512', width: 512, height: 512 },
    { label: '768×768', width: 768, height: 768 },
    { label: '1024×1024', width: 1024, height: 1024 },
    { label: '512×768', width: 512, height: 768 },
    { label: '768×512', width: 768, height: 512 },
  ]

  const examplePrompts = [
    'a beautiful landscape with mountains and lake, sunset, 4k, highly detailed',
    'a cute cat sitting on a windowsill, soft lighting, detailed fur',
    'cyberpunk city at night, neon lights, futuristic, highly detailed',
    'portrait of a young woman, natural lighting, photorealistic',
  ]

  return (
    <div style={{ padding: 24 }}>
      <Row gutter={24}>
        {/* Left Panel - Controls */}
        <Col xs={24} lg={10}>
          <Card title="生成设置">
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              {/* Prompt */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  正向提示词 <span style={{ color: 'red' }}>*</span>
                </div>
                <TextArea
                  value={store.prompt}
                  onChange={(e) => store.setPrompt(e.target.value)}
                  placeholder="描述你想生成的图像..."
                  rows={4}
                  disabled={store.isGenerating}
                />
                <div style={{ marginTop: 8 }}>
                  <Space wrap>
                    {examplePrompts.map((prompt, index) => (
                      <Tag
                        key={index}
                        style={{ cursor: 'pointer' }}
                        onClick={() => !store.isGenerating && store.setPrompt(prompt)}
                      >
                        示例 {index + 1}
                      </Tag>
                    ))}
                  </Space>
                </div>
              </div>

              {/* Negative Prompt */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>负向提示词</div>
                <TextArea
                  value={store.negativePrompt}
                  onChange={(e) => store.setNegativePrompt(e.target.value)}
                  placeholder="要避免的内容..."
                  rows={2}
                  disabled={store.isGenerating}
                />
              </div>

              <Divider />

              {/* Size Presets */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>尺寸预设</div>
                <Space wrap>
                  {presetSizes.map((preset) => (
                    <Button
                      key={preset.label}
                      size="small"
                      onClick={() => {
                        store.setWidth(preset.width)
                        store.setHeight(preset.height)
                      }}
                      disabled={store.isGenerating}
                      type={
                        store.width === preset.width && store.height === preset.height
                          ? 'primary'
                          : 'default'
                      }
                    >
                      {preset.label}
                    </Button>
                  ))}
                </Space>
              </div>

              {/* Width & Height */}
              <Row gutter={16}>
                <Col span={12}>
                  <div style={{ marginBottom: 8, fontWeight: 500 }}>宽度</div>
                  <InputNumber
                    value={store.width}
                    onChange={(value) => store.setWidth(value || 512)}
                    min={64}
                    max={2048}
                    step={64}
                    style={{ width: '100%' }}
                    disabled={store.isGenerating}
                  />
                </Col>
                <Col span={12}>
                  <div style={{ marginBottom: 8, fontWeight: 500 }}>高度</div>
                  <InputNumber
                    value={store.height}
                    onChange={(value) => store.setHeight(value || 512)}
                    min={64}
                    max={2048}
                    step={64}
                    style={{ width: '100%' }}
                    disabled={store.isGenerating}
                  />
                </Col>
              </Row>

              {/* Steps */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  采样步数: {store.steps}
                </div>
                <Slider
                  value={store.steps}
                  onChange={(value) => store.setSteps(value)}
                  min={1}
                  max={100}
                  disabled={store.isGenerating}
                />
              </div>

              {/* CFG Scale */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  CFG Scale: {store.cfgScale.toFixed(1)}
                </div>
                <Slider
                  value={store.cfgScale}
                  onChange={(value) => store.setCfgScale(value)}
                  min={1}
                  max={20}
                  step={0.5}
                  disabled={store.isGenerating}
                />
              </div>

              {/* Seed */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>种子</div>
                <Space.Compact style={{ width: '100%' }}>
                  <InputNumber
                    value={store.seed || -1}
                    onChange={(value) => store.setSeed(value === -1 ? null : value)}
                    placeholder="随机"
                    style={{ width: '100%' }}
                    disabled={store.isGenerating}
                  />
                  <Tooltip title="随机种子">
                    <Button
                      icon={<ReloadOutlined />}
                      onClick={handleRandomSeed}
                      disabled={store.isGenerating}
                    />
                  </Tooltip>
                </Space.Compact>
              </div>

              {/* Sampler */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>采样器</div>
                <Select
                  value={store.sampler}
                  onChange={(value) => store.setSampler(value)}
                  style={{ width: '100%' }}
                  disabled={store.isGenerating}
                  options={[
                    { label: 'Euler', value: 'euler' },
                    { label: 'Euler A', value: 'euler_a' },
                    { label: 'DPM++ 2M', value: 'dpmpp_2m' },
                    { label: 'DPM++ SDE', value: 'dpmpp_sde' },
                    { label: 'DDIM', value: 'ddim' },
                  ]}
                />
              </div>

              {/* Generate Button */}
              <Button
                type="primary"
                size="large"
                icon={<ThunderboltOutlined />}
                onClick={handleGenerate}
                loading={store.isGenerating}
                block
              >
                {store.isGenerating ? '生成中...' : '开始生成'}
              </Button>

              {/* Progress */}
              {store.isGenerating && (
                <div>
                  <Progress percent={progress} status="active" />
                  <div style={{ textAlign: 'center', marginTop: 8, color: '#666' }}>
                    {statusText}
                  </div>
                </div>
              )}
            </Space>
          </Card>
        </Col>

        {/* Right Panel - Preview */}
        <Col xs={24} lg={14}>
          <Card
            title="生成结果"
            extra={
              currentImage && (
                <Space>
                  <Button
                    icon={<DownloadOutlined />}
                    onClick={() => {
                      const link = document.createElement('a')
                      link.href = currentImage
                      link.download = `genesis_${Date.now()}.png`
                      link.click()
                    }}
                  >
                    下载
                  </Button>
                  <Button icon={<SaveOutlined />}>保存</Button>
                </Space>
              )
            }
          >
            <div
              style={{
                minHeight: 600,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#f5f5f5',
                borderRadius: 8,
              }}
            >
              {store.isGenerating ? (
                <div style={{ textAlign: 'center' }}>
                  <Spin size="large" />
                  <div style={{ marginTop: 16, color: '#666' }}>正在生成图像...</div>
                </div>
              ) : currentImage ? (
                <Image
                  src={currentImage}
                  alt="Generated"
                  style={{ maxWidth: '100%', maxHeight: 600 }}
                  preview={{
                    mask: '查看大图',
                  }}
                />
              ) : (
                <div style={{ textAlign: 'center', color: '#999' }}>
                  <ThunderboltOutlined style={{ fontSize: 64, marginBottom: 16 }} />
                  <div>点击「开始生成」按钮创建图像</div>
                </div>
              )}
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default TextToImage
