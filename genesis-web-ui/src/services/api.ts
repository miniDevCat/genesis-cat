import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5 minutes for image generation
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

export interface GenerateParams {
  prompt: string
  negative_prompt?: string
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  seed?: number | null
  sampler?: string
  scheduler?: string
  task_type?: string
  [key: string]: any
}

export interface Task {
  task_id: string
  task_type: string
  status: string
  progress: number
  result?: any
  error?: string
  created_at?: string
  started_at?: string
  completed_at?: string
}

export interface TaskSubmitResponse {
  success: boolean
  task_id: string
  task: Task
}

export interface TaskStatusResponse {
  success: boolean
  task: Task
}

export interface ModelsResponse {
  success: boolean
  models: {
    checkpoints: string[]
    loras: string[]
    vae: string[]
  }
}

export interface DeviceInfo {
  device: string
  device_name?: string
  memory_total?: number
  memory_allocated?: number
  memory_reserved?: number
}

export interface DeviceResponse {
  success: boolean
  device: DeviceInfo
}

// API methods
export const genesisApi = {
  // Health check
  health: () => api.get('/health'),

  // Submit generation task
  submitTask: (data: any): Promise<TaskSubmitResponse> =>
    api.post('/task/submit', data),

  // Get task status
  getTaskStatus: (taskId: string): Promise<TaskStatusResponse> =>
    api.get(`/task/${taskId}`),

  // Cancel task
  cancelTask: (taskId: string) => api.post(`/task/${taskId}/cancel`),

  // List all tasks
  listTasks: () => api.get('/tasks'),

  // Get available models
  getModels: (): Promise<ModelsResponse> => api.get('/models'),

  // Get device info
  getDeviceInfo: (): Promise<DeviceResponse> => api.get('/device'),

  // Create session
  createSession: (clientType: string = 'web') =>
    api.post('/session/create', { client_type: clientType }),

  // Get config params
  getConfigParams: () => api.get('/config/params'),

  // Save config params
  saveConfigParams: (params: any) => api.post('/config/params', { params }),

  // Update default config
  updateDefaultConfig: (config: any) => api.post('/config/defaults', config),
}

export default api
