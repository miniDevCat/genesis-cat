import React, { useState } from 'react'
import {
  Card,
  Input,
  Button,
  Space,
  Slider,
  Select,
  InputNumber,
  Row,
  Col,
  Progress,
  message,
  Divider,
  Tag,
  Tooltip,
  Alert,
  Switch,
} from 'antd'
import {
  PlayCircleOutlined,
  StopOutlined,
  DownloadOutlined,
  ReloadOutlined,
  VideoCameraOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import { genesisApi } from '../../services/api'

const { TextArea } = Input

interface VideoParams {
  prompt: string
  negative_prompt: string
  width: number
  height: number
  frames: number
  fps: number
  steps: number
  cfg_scale: number
  seed: number | null
  motion_strength: number
  shift?: number
  model_id?: string
  scheduler?: string
  loras?: Array<{ name: string; strength: number }>
  lora_low_mem_load?: boolean
  lora_merge_loras?: boolean
}

interface VideoModel {
  id: string
  name: string
  path: string
  type: string
  description: string
}

const TextToVideo: React.FC = () => {
  // 可用模型
  const [videoModels, setVideoModels] = useState<VideoModel[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  
  // LoRA 列表和选择
  const [availableLoras, setAvailableLoras] = useState<string[]>([])
  const [selectedLoras, setSelectedLoras] = useState<Array<{ name: string; strength: number }>>([])
  
  // 生成参数
  const [params, setParams] = useState<VideoParams>({
    prompt: '',
    negative_prompt: '',
    width: 512,
    height: 512,
    frames: 16,
    fps: 8,
    steps: 20,
    cfg_scale: 7.5,
    seed: null,
    motion_strength: 0.5,
    shift: 1.0,
    scheduler: 'unipc',
    loras: [],
    lora_low_mem_load: false,
    lora_merge_loras: false,
  })

  // 生成状态
  const [isGenerating, setIsGenerating] = useState(false)
  const [progress, setProgress] = useState(0)
  const [statusText, setStatusText] = useState('')
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null)
  const [generatedVideo, setGeneratedVideo] = useState<string | null>(null)
  const [videoInfo, setVideoInfo] = useState<any>(null)
  
  // 加载可用模型和配置
  React.useEffect(() => {
    const loadData = async () => {
      try {
        // 1. 先加载模型列表
        console.log('[DEBUG] Loading models...')
        const modelsResponse = await genesisApi.getModels() as any
        if (modelsResponse.success && modelsResponse.models) {
          setVideoModels(modelsResponse.models.video_models || [])
          console.log('[DEBUG] Models loaded:', modelsResponse.models.video_models)
          
          // 加载 LoRA 列表
          if (modelsResponse.models.loras) {
            setAvailableLoras(modelsResponse.models.loras)
            console.log('[DEBUG] LoRAs loaded:', modelsResponse.models.loras)
          }
        }
        
        // 2. 再加载配置（确保模型列表已加载）
        console.log('[DEBUG] Loading config...')
        const configResponse = await genesisApi.getConfigParams() as any
        console.log('[DEBUG] Config response:', configResponse)
        
        if (configResponse.success && configResponse.config) {
          const lastUsed = configResponse.config.last_used || {}
          console.log('[DEBUG] Last used params:', lastUsed)
          
          // 加载上次使用的参数
          if (Object.keys(lastUsed).length > 0) {
            // 提取 model_id、loras 和视频信息
            const { model_id, loras, last_video_url, last_video_info, ...otherParams } = lastUsed
            console.log('[DEBUG] Extracted model_id:', model_id)
            console.log('[DEBUG] Extracted loras:', loras)
            console.log('[DEBUG] Last video URL:', last_video_url)
            console.log('[DEBUG] Other params:', otherParams)
            
            // 设置参数
            setParams(prev => ({
              ...prev,
              ...otherParams,
              loras: loras || []
            }))
            
            // 设置 LoRA 选择
            if (loras && Array.isArray(loras) && loras.length > 0) {
              console.log('[DEBUG] Restoring LoRA selection:', loras)
              setSelectedLoras(loras)
            }
            
            // 设置模型选择（模型列表已经加载完成）
            if (model_id) {
              console.log('[DEBUG] Setting selected model to:', model_id)
              setSelectedModel(model_id)
            } else if (modelsResponse.models?.default_video_model) {
              // 如果没有保存的模型，使用默认模型
              console.log('[DEBUG] Using default model:', modelsResponse.models.default_video_model)
              setSelectedModel(modelsResponse.models.default_video_model)
            }
            
            // 恢复上次生成的视频
            if (last_video_url) {
              console.log('[DEBUG] Restoring last video:', last_video_url)
              setGeneratedVideo(last_video_url)
              
              // 解析视频信息
              if (last_video_info) {
                try {
                  const videoInfo = JSON.parse(last_video_info)
                  setVideoInfo(videoInfo)
                  console.log('[DEBUG] Restored video info:', videoInfo)
                } catch (e) {
                  console.error('[ERROR] Failed to parse video info:', e)
                }
              }
              
              // message.success(`已加载参数和上次生成的视频`)
            } else {
              message.success(`已加载参数 (模型: ${model_id || '默认'})`)
            }
          } else if (modelsResponse.models?.default_video_model) {
            // 如果没有保存的配置，使用默认模型
            setSelectedModel(modelsResponse.models.default_video_model)
          }
        }
      } catch (error) {
        console.error('Failed to load data:', error)
      }
    }
    
    loadData()
  }, [])

  // 预设提示词
  const examplePrompts = [
    '一只可爱的小猫在草地上玩耍，阳光明媚',
    '城市街道上车水马龙，延时摄影效果',
    '海浪拍打着沙滩，日落时分',
    '樱花飘落，微风吹拂',
    '宇航员在太空中漂浮',
  ]

  // 尺寸预设
  const sizePresets = [
    { label: '方形 (512×512)', width: 512, height: 512 },
    { label: '横屏 (768×512)', width: 768, height: 512 },
    { label: '竖屏 (512×768)', width: 512, height: 768 },
    { label: '宽屏 (1024×576)', width: 1024, height: 576 },
  ]

  // 轮询任务状态
  const pollTaskStatus = async (taskId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await genesisApi.getTaskStatus(taskId)
        
        if (response.success && response.task) {
          const task = response.task
          setProgress(task.progress || 0)
          setStatusText(task.status)

          if (task.status === 'completed') {
            clearInterval(pollInterval)
            setIsGenerating(false)
            setProgress(100)
            setStatusText('生成完成！')
            
            console.log('[DEBUG] Task completed, result:', task.result)
            
            if (task.result && task.result.video) {
              console.log('[DEBUG] Video data received, length:', task.result.video.length)
              setGeneratedVideo(task.result.video)
              setVideoInfo(task.result)
              message.success('视频生成成功！')
            } else {
              console.error('[ERROR] No video data in result:', task.result)
              message.error('视频数据为空，请检查后端')
            }
          } else if (task.status === 'failed') {
            clearInterval(pollInterval)
            setIsGenerating(false)
            message.error(`生成失败: ${task.error || '未知错误'}`)
            setStatusText('生成失败')
          }
        }
      } catch (error: any) {
        console.error('Poll error:', error)
      }
    }, 1000)

    // 超时保护
    setTimeout(() => {
      clearInterval(pollInterval)
      if (isGenerating) {
        setIsGenerating(false)
        message.warning('任务超时，请检查后端状态')
      }
    }, 300000) // 5分钟超时
  }

  // 保存当前参数
  const handleSaveParams = async () => {
    try {
      // 保存参数，包括选中的模型
      const paramsToSave = {
        ...params,
        model_id: selectedModel
      }
      console.log('[DEBUG] Saving params:', paramsToSave)
      console.log('[DEBUG] Selected model:', selectedModel)
      await genesisApi.saveConfigParams(paramsToSave)
      message.success(`参数已保存 (模型: ${selectedModel})`)
    } catch (error: any) {
      message.error(`保存失败: ${error.message}`)
      console.error('Save params error:', error)
    }
  }

  // 生成视频
  const handleGenerate = async () => {
    if (!params.prompt.trim()) {
      message.warning('请输入提示词')
      return
    }

    // 自动保存参数
    try {
      const paramsToSave = {
        ...params,
        model_id: selectedModel
      }
      await genesisApi.saveConfigParams(paramsToSave)
    } catch (error) {
      console.error('Auto-save params failed:', error)
    }

    setIsGenerating(true)
    setProgress(0)
    setStatusText('提交任务...')
    setGeneratedVideo(null)
    setVideoInfo(null)

    try {
      console.log('[DEBUG] Submitting task with model_id:', selectedModel)
      console.log('[DEBUG] Submitting task with loras:', params.loras)
      const response = await genesisApi.submitTask({
        task_type: 'text_to_video',
        params: {
          ...params,
          seed: params.seed === null ? -1 : params.seed,
          model_id: selectedModel,
        },
      } as any)

      if (response.success && response.task_id) {
        setCurrentTaskId(response.task_id)
        setStatusText('生成中...')
        message.info('任务已提交，开始生成视频...')
        pollTaskStatus(response.task_id)
      } else {
        throw new Error('任务提交失败')
      }
    } catch (error: any) {
      setIsGenerating(false)
      message.error(`生成失败: ${error.message}`)
      console.error('Generate error:', error)
    }
  }

  // 停止生成
  const handleStop = async () => {
    if (currentTaskId) {
      try {
        await genesisApi.cancelTask(currentTaskId)
        setIsGenerating(false)
        message.info('已取消生成')
      } catch (error) {
        console.error('Cancel error:', error)
      }
    }
  }

  // 下载视频
  const handleDownload = async () => {
    if (!generatedVideo) return
    
    try {
      console.log('[DEBUG] Starting download...')
      console.log('[DEBUG] Video URL:', generatedVideo)
      
      message.loading('正在下载视频...', 0)
      
      // 构建完整 URL
      const fullUrl = generatedVideo.startsWith('http') 
        ? generatedVideo 
        : `http://localhost:5000${generatedVideo}`
      
      console.log('[DEBUG] Full URL:', fullUrl)
      
      // 使用 fetch 获取视频文件
      const response = await fetch(fullUrl)
      console.log('[DEBUG] Response status:', response.status)
      console.log('[DEBUG] Response headers:', Object.fromEntries(response.headers.entries()))
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      
      // 转换为 Blob
      const blob = await response.blob()
      console.log('[DEBUG] Downloaded blob size:', blob.size, 'bytes')
      console.log('[DEBUG] Downloaded blob type:', blob.type)
      
      if (blob.size < 1000) {
        console.error('[ERROR] Blob too small! Might be an error page.')
        throw new Error(`文件太小 (${blob.size} bytes)，可能下载失败`)
      }
      
      // 创建下载链接
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `genesis_video_${Date.now()}.mp4`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      // 释放 URL
      window.URL.revokeObjectURL(url)
      
      message.destroy()
      message.success(`视频下载成功！(${(blob.size / 1024 / 1024).toFixed(2)} MB)`)
    } catch (error: any) {
      message.destroy()
      message.error(`下载失败: ${error.message}`)
      console.error('[ERROR] Download error:', error)
    }
  }

  // 使用示例提示词
  const useExamplePrompt = (prompt: string) => {
    setParams({ ...params, prompt })
  }

  // 应用尺寸预设
  const applySizePreset = (preset: { width: number; height: number }) => {
    setParams({ ...params, width: preset.width, height: preset.height })
  }

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={24}>
        {/* 左侧：参数控制 */}
        <Col xs={24} lg={10}>
          <Card 
            title={<><VideoCameraOutlined /> 文生视频参数</>}
            extra={
              <Space size="middle">
                <Button
                  icon={<SaveOutlined />}
                  onClick={handleSaveParams}
                  disabled={isGenerating}
                >
                  保存设置
                </Button>
                <Button
                  type={isGenerating ? 'default' : 'primary'}
                  danger={isGenerating}
                  icon={isGenerating ? <StopOutlined /> : <PlayCircleOutlined />}
                  onClick={isGenerating ? handleStop : handleGenerate}
                >
                  {isGenerating ? '停止运行' : '运行应用'}
                </Button>
              </Space>
            }
            style={{ marginBottom: 24 }} 
            variant="outlined"
          >
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              {/* 提示词 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  正向提示词 <Tag color="blue">必填</Tag>
                </div>
                <TextArea
                  value={params.prompt}
                  onChange={(e) => setParams({ ...params, prompt: e.target.value })}
                  placeholder="描述你想要生成的视频内容..."
                  rows={4}
                  disabled={isGenerating}
                />
                <div style={{ marginTop: 8 }}>
                  <span style={{ fontSize: 12, color: '#666' }}>示例：</span>
                  <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {examplePrompts.map((prompt, index) => (
                      <Tag
                        key={index}
                        style={{ cursor: 'pointer' }}
                        onClick={() => useExamplePrompt(prompt)}
                      >
                        {prompt}
                      </Tag>
                    ))}
                  </div>
                </div>
              </div>

              {/* 负向提示词 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>负向提示词</div>
                <TextArea
                  value={params.negative_prompt}
                  onChange={(e) => setParams({ ...params, negative_prompt: e.target.value })}
                  placeholder="描述不想要的内容..."
                  rows={2}
                  disabled={isGenerating}
                />
              </div>

              <Divider />

              {/* 模型选择 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  视频生成模型 <Tag color="green">可选</Tag>
                  <Tag color="blue" style={{ marginLeft: 8 }}>当前: {selectedModel || '未选择'}</Tag>
                </div>
                <Select
                  value={selectedModel}
                  onChange={(value) => {
                    console.log('[DEBUG] Model changed to:', value)
                    setSelectedModel(value)
                  }}
                  style={{ width: '100%' }}
                  disabled={isGenerating}
                  placeholder="选择视频生成模型"
                >
                  {videoModels.map((model) => (
                    <Select.Option key={model.id} value={model.id}>
                      {model.name}
                    </Select.Option>
                  ))}
                </Select>
                {selectedModel && (
                  <div style={{ marginTop: 8, fontSize: 12, color: '#999' }}>
                    <div style={{ marginBottom: 4 }}>
                      {videoModels.find(m => m.id === selectedModel)?.description}
                    </div>
                    <div style={{ color: '#bbb', wordBreak: 'break-all' }}>
                      {videoModels.find(m => m.id === selectedModel)?.path}
                    </div>
                  </div>
                )}
              </div>

              {/* LoRA 选择 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  LoRA 模型 <Tag color="purple">多选</Tag>
                  {selectedLoras.length > 0 && (
                    <Tag color="blue" style={{ marginLeft: 8 }}>已选: {selectedLoras.length}</Tag>
                  )}
                </div>
                <Select
                  mode="multiple"
                  value={selectedLoras.map(l => l.name)}
                  onChange={(values) => {
                    // 保留已有的强度设置，新增的默认为 1.0
                    const newLoras = values.map(name => {
                      const existing = selectedLoras.find(l => l.name === name)
                      return existing || { name, strength: 1.0 }
                    })
                    setSelectedLoras(newLoras)
                    setParams({ ...params, loras: newLoras })
                  }}
                  style={{ width: '100%' }}
                  disabled={isGenerating}
                  placeholder="选择 LoRA 模型（可多选）"
                  maxTagCount="responsive"
                >
                  {availableLoras.map((lora) => (
                    <Select.Option key={lora} value={lora}>
                      {lora}
                    </Select.Option>
                  ))}
                </Select>
                
                {/* LoRA 强度调整 */}
                {selectedLoras.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    {selectedLoras.map((lora, index) => (
                      <div key={lora.name} style={{ marginBottom: 8 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <span style={{ fontSize: 12, color: '#666' }}>{lora.name}</span>
                          <span style={{ fontSize: 12, color: '#1890ff' }}>强度: {lora.strength.toFixed(2)}</span>
                        </div>
                        <Slider
                          value={lora.strength}
                          onChange={(value) => {
                            const newLoras = [...selectedLoras]
                            newLoras[index].strength = value
                            setSelectedLoras(newLoras)
                            setParams({ ...params, loras: newLoras })
                          }}
                          min={0}
                          max={2}
                          step={0.1}
                          disabled={isGenerating}
                        />
                      </div>
                    ))}
                    
                    {/* LoRA 高级选项 */}
                    <div style={{ marginTop: 16, padding: '12px', background: '#ffffff', border: '0px solid #d9d9d9', borderRadius: '4px' }}>
                      <div style={{ marginBottom: 8, fontWeight: 500, fontSize: 12 }}>LoRA 高级选项</div>
                      <div style={{ marginBottom: 8 }}>
                        <Tooltip title="使用较少的显存加载 LoRA，但加载速度会变慢">
                          <Space>
                            <Switch
                              checked={params.lora_low_mem_load}
                              onChange={(checked) => setParams({ ...params, lora_low_mem_load: checked })}
                              disabled={isGenerating}
                              size="small"
                            />
                            <span style={{ fontSize: 12 }}>低显存模式 (Low Mem Load)</span>
                          </Space>
                        </Tooltip>
                      </div>
                      <div>
                        <Tooltip title="将 LoRA 合并到模型中（推荐关闭以获得更好的兼容性）">
                          <Space>
                            <Switch
                              checked={params.lora_merge_loras}
                              onChange={(checked) => setParams({ ...params, lora_merge_loras: checked })}
                              disabled={isGenerating}
                              size="small"
                            />
                            <span style={{ fontSize: 12 }}>合并 LoRA (Merge LoRAs)</span>
                          </Space>
                        </Tooltip>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <Divider />

              {/* 视频尺寸 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>视频尺寸</div>
                <Space wrap>
                  {sizePresets.map((preset, index) => (
                    <Button
                      key={index}
                      size="small"
                      onClick={() => applySizePreset(preset)}
                      type={params.width === preset.width && params.height === preset.height ? 'primary' : 'default'}
                    >
                      {preset.label}
                    </Button>
                  ))}
                </Space>
                <Row gutter={16} style={{ marginTop: 12 }}>
                  <Col span={12}>
                    <div style={{ marginBottom: 4, fontSize: 12 }}>宽度</div>
                    <InputNumber
                      value={params.width}
                      onChange={(value) => setParams({ ...params, width: value || 512 })}
                      min={256}
                      max={1024}
                      step={64}
                      style={{ width: '100%' }}
                      disabled={isGenerating}
                    />
                  </Col>
                  <Col span={12}>
                    <div style={{ marginBottom: 4, fontSize: 12 }}>高度</div>
                    <InputNumber
                      value={params.height}
                      onChange={(value) => setParams({ ...params, height: value || 512 })}
                      min={256}
                      max={1024}
                      step={64}
                      style={{ width: '100%' }}
                      disabled={isGenerating}
                    />
                  </Col>
                </Row>
              </div>

              {/* 视频参数 */}
              <Row gutter={16}>
                <Col span={12}>
                  <div style={{ marginBottom: 8, fontWeight: 500 }}>
                    <Tooltip title="视频总帧数，越多越长">
                      帧数
                    </Tooltip>
                  </div>
                  <InputNumber
                    value={params.frames}
                    onChange={(value) => setParams({ ...params, frames: value || 16 })}
                    min={8}
                    max={64}
                    style={{ width: '100%' }}
                    disabled={isGenerating}
                  />
                </Col>
                <Col span={12}>
                  <div style={{ marginBottom: 8, fontWeight: 500 }}>
                    <Tooltip title="每秒帧数，影响播放速度">
                      FPS
                    </Tooltip>
                  </div>
                  <InputNumber
                    value={params.fps}
                    onChange={(value) => setParams({ ...params, fps: value || 8 })}
                    min={4}
                    max={30}
                    style={{ width: '100%' }}
                    disabled={isGenerating}
                  />
                </Col>
              </Row>

              {/* 采样步数 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  采样步数: {params.steps}
                </div>
                <Slider
                  value={params.steps}
                  onChange={(value) => setParams({ ...params, steps: value })}
                  min={1}
                  max={50}
                  disabled={isGenerating}
                />
              </div>

              {/* 调度器选择 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  <Tooltip title="采样调度器，影响生成质量和速度">
                    调度器 (Scheduler)
                  </Tooltip>
                </div>
                <Select
                  value={params.scheduler}
                  onChange={(value) => setParams({ ...params, scheduler: value })}
                  style={{ width: '100%' }}
                  disabled={isGenerating}
                  showSearch
                  optionFilterProp="children"
                >
                  <Select.OptGroup label="推荐调度器">
                    <Select.Option value="unipc">UniPC (推荐)</Select.Option>
                    <Select.Option value="unipc/beta">UniPC Beta</Select.Option>
                    <Select.Option value="rcm">RCM (4步快速)</Select.Option>
                    <Select.Option value="euler">Euler (快速)</Select.Option>
                  </Select.OptGroup>
                  
                  <Select.OptGroup label="DPM 系列">
                    <Select.Option value="dpm++">DPM++ (高质量)</Select.Option>
                    <Select.Option value="dpm++/beta">DPM++ Beta</Select.Option>
                    <Select.Option value="dpm++_sde">DPM++ SDE</Select.Option>
                    <Select.Option value="dpm++_sde/beta">DPM++ SDE Beta</Select.Option>
                  </Select.OptGroup>
                  
                  <Select.OptGroup label="Euler 系列">
                    <Select.Option value="euler/beta">Euler Beta</Select.Option>
                  </Select.OptGroup>
                  
                  <Select.OptGroup label="快速调度器">
                    <Select.Option value="lcm">LCM (超快)</Select.Option>
                    <Select.Option value="lcm/beta">LCM Beta</Select.Option>
                    <Select.Option value="humo_lcm">Humo LCM</Select.Option>
                  </Select.OptGroup>
                  
                  <Select.OptGroup label="FlowMatch 系列">
                    <Select.Option value="flowmatch_causvid">FlowMatch CausVid</Select.Option>
                    <Select.Option value="flowmatch_distill">FlowMatch Distill</Select.Option>
                    <Select.Option value="flowmatch_pusa">FlowMatch Pusa</Select.Option>
                    <Select.Option value="flowmatch_frame_euler_d">FlowMatch Frame Euler D</Select.Option>
                    <Select.Option value="flowmatch_sa_ode_stable">FlowMatch SA ODE Stable</Select.Option>
                  </Select.OptGroup>
                  
                  <Select.OptGroup label="特殊调度器">
                    <Select.Option value="deis">DEIS</Select.Option>
                    <Select.Option value="res_multistep">Res MultiStep</Select.Option>
                    <Select.Option value="sa_ode_stable/lowstep">SA ODE Stable (低步数)</Select.Option>
                    <Select.Option value="multitalk">MultiTalk</Select.Option>
                  </Select.OptGroup>
                  
                  <Select.OptGroup label="易经五行系列">
                    <Select.Option value="iching/wuxing">易经五行 (标准)</Select.Option>
                    <Select.Option value="iching/wuxing-strong">易经五行 (强力)</Select.Option>
                    <Select.Option value="iching/wuxing-stable">易经五行 (稳定)</Select.Option>
                    <Select.Option value="iching/wuxing-smooth">易经五行 (平滑)</Select.Option>
                    <Select.Option value="iching/wuxing-clean">易经五行 (清晰)</Select.Option>
                    <Select.Option value="iching/wuxing-sharp">易经五行 (锐利)</Select.Option>
                    <Select.Option value="iching/wuxing-lowstep">易经五行 (低步数)</Select.Option>
                  </Select.OptGroup>
                </Select>
                <div style={{ marginTop: 4, fontSize: 12, color: '#999' }}>
                  {params.scheduler === 'unipc' && '⭐ 平衡质量和速度，推荐日常使用'}
                  {params.scheduler === 'rcm' && '⚡ 4步快速生成，适合RCM模型'}
                  {params.scheduler === 'dpm++' && '🎨 高质量输出，速度较慢'}
                  {params.scheduler === 'euler' && '⚡ 快速生成，质量良好'}
                  {params.scheduler === 'lcm' && '⚡⚡ 超快速生成，需要LCM模型'}
                  {params.scheduler?.includes('iching') && '🔮 易经五行调度器，中国传统智慧'}
                  {params.scheduler?.includes('flowmatch') && '🌊 FlowMatch技术，实验性调度器'}
                </div>
              </div>

              {/* CFG Scale */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  CFG Scale: {params.cfg_scale}
                </div>
                <Slider
                  value={params.cfg_scale}
                  onChange={(value) => setParams({ ...params, cfg_scale: value })}
                  min={1}
                  max={20}
                  step={0.5}
                  disabled={isGenerating}
                />
              </div>

              {/* Shift 值 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  <Tooltip title="时间步偏移值，影响生成质量和风格">
                    Shift: {params.shift}
                  </Tooltip>
                </div>
                <Slider
                  value={params.shift}
                  onChange={(value) => setParams({ ...params, shift: value })}
                  min={0}
                  max={10}
                  step={0.1}
                  disabled={isGenerating}
                />
              </div>

              {/* 运动强度 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>
                  <Tooltip title="控制视频中的运动幅度">
                    运动强度: {params.motion_strength}
                  </Tooltip>
                </div>
                <Slider
                  value={params.motion_strength}
                  onChange={(value) => setParams({ ...params, motion_strength: value })}
                  min={0}
                  max={1}
                  step={0.1}
                  disabled={isGenerating}
                />
              </div>

              {/* 随机种子 */}
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>随机种子</div>
                <InputNumber
                  value={params.seed}
                  onChange={(value) => setParams({ ...params, seed: value })}
                  placeholder="-1 表示随机"
                  style={{ width: '100%' }}
                  disabled={isGenerating}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 右侧：预览和结果 */}
        <Col xs={24} lg={14}>
          <Card title="视频预览" variant="outlined">
            {/* 进度显示 */}
            {isGenerating && (
              <div style={{ marginBottom: 24 }}>
                <Progress percent={progress} status="active" />
                <div style={{ textAlign: 'center', marginTop: 8, color: '#666' }}>
                  {statusText}
                </div>
              </div>
            )}

            {/* 提示信息 */}
            {!generatedVideo && !isGenerating && (
              <Alert
                message="提示"
                description={
                  <div>
                    <p>• 文生视频功能需要较长时间（1-5分钟）</p>
                    <p>• 首次使用会下载视频生成模型（约 10GB）</p>
                    <p>• 建议使用 GPU 加速，CPU 生成会非常慢</p>
                    <p>• 帧数越多，生成时间越长</p>
                  </div>
                }
                type="info"
                showIcon
              />
            )}

            {/* 视频播放器 */}
            {generatedVideo && (
              <div>
                <div
                  style={{
                    background: '#f0f0f0',
                    borderRadius: 8,
                    padding: 16,
                    marginBottom: 16,
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                  }}
                >
                  <video
                    src={generatedVideo.startsWith('http') ? generatedVideo : `http://localhost:5000${generatedVideo}`}
                    controls
                    loop
                    autoPlay
                    style={{
                      maxWidth: '100%',
                      maxHeight: '500px',
                      borderRadius: 4,
                    }}
                    onError={(e) => {
                      console.error('[ERROR] Video playback error:', e)
                      console.error('[ERROR] Video src:', generatedVideo)
                      message.error('视频播放失败，请检查视频格式')
                    }}
                    onLoadedData={() => {
                      console.log('[DEBUG] Video loaded successfully')
                      console.log('[DEBUG] Video src:', generatedVideo)
                    }}
                  />
                </div>

                {/* 操作按钮 */}
                <Space>
                  <Button
                    icon={<DownloadOutlined />}
                    onClick={handleDownload}
                  >
                    下载视频
                  </Button>
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={() => {
                      setGeneratedVideo(null)
                      setVideoInfo(null)
                    }}
                  >
                    清除
                  </Button>
                </Space>

                {/* 视频信息 */}
                {videoInfo && (
                  <Card
                    size="small"
                    title="视频信息"
                    style={{ marginTop: 16 }}
                  >
                    <Space direction="vertical" size="small">
                      <div><strong>提示词:</strong> {videoInfo.prompt}</div>
                      <div><strong>尺寸:</strong> {videoInfo.width} × {videoInfo.height}</div>
                      <div><strong>帧数:</strong> {videoInfo.frames} 帧</div>
                      <div><strong>FPS:</strong> {videoInfo.fps}</div>
                      <div><strong>时长:</strong> {(videoInfo.frames / videoInfo.fps).toFixed(2)} 秒</div>
                      {videoInfo.seed && <div><strong>种子:</strong> {videoInfo.seed}</div>}
                    </Space>
                  </Card>
                )}
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default TextToVideo
