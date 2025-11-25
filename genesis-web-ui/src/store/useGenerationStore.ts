import { create } from 'zustand'

interface GenerationState {
  prompt: string
  negativePrompt: string
  width: number
  height: number
  steps: number
  cfgScale: number
  seed: number | null
  sampler: string
  scheduler: string
  isGenerating: boolean
  currentTaskId: string | null
  generatedImages: Array<{
    id: string
    url: string
    prompt: string
    params: any
    timestamp: number
  }>
  
  setPrompt: (prompt: string) => void
  setNegativePrompt: (negativePrompt: string) => void
  setWidth: (width: number) => void
  setHeight: (height: number) => void
  setSteps: (steps: number) => void
  setCfgScale: (cfgScale: number) => void
  setSeed: (seed: number | null) => void
  setSampler: (sampler: string) => void
  setScheduler: (scheduler: string) => void
  setIsGenerating: (isGenerating: boolean) => void
  setCurrentTaskId: (taskId: string | null) => void
  addGeneratedImage: (image: any) => void
  clearHistory: () => void
}

export const useGenerationStore = create<GenerationState>((set) => ({
  prompt: 'a beautiful landscape with mountains and lake, sunset, 4k, highly detailed',
  negativePrompt: 'ugly, blurry, low quality, distorted, bad anatomy',
  width: 512,
  height: 512,
  steps: 20,
  cfgScale: 7.0,
  seed: null,
  sampler: 'euler',
  scheduler: 'normal',
  isGenerating: false,
  currentTaskId: null,
  generatedImages: [],
  
  setPrompt: (prompt) => set({ prompt }),
  setNegativePrompt: (negativePrompt) => set({ negativePrompt }),
  setWidth: (width) => set({ width }),
  setHeight: (height) => set({ height }),
  setSteps: (steps) => set({ steps }),
  setCfgScale: (cfgScale) => set({ cfgScale }),
  setSeed: (seed) => set({ seed }),
  setSampler: (sampler) => set({ sampler }),
  setScheduler: (scheduler) => set({ scheduler }),
  setIsGenerating: (isGenerating) => set({ isGenerating }),
  setCurrentTaskId: (taskId) => set({ currentTaskId: taskId }),
  addGeneratedImage: (image) =>
    set((state) => ({
      generatedImages: [image, ...state.generatedImages],
    })),
  clearHistory: () => set({ generatedImages: [] }),
}))
