import React, { useState, useEffect } from 'react'
import '../styles/themes.css'

interface Theme {
  id: string
  name: string
  description: string
  preview: string
}

const themes: Theme[] = [
  {
    id: 'infinite-talk',
    name: 'InfiniteTalk',
    description: '紫色渐变，优雅现代',
    preview: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
  },
  {
    id: 'wan-animate',
    name: 'WanAnimate',
    description: '清新蓝绿，活力十足',
    preview: 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)'
  },
  {
    id: 'standard',
    name: 'Standard I2V',
    description: '经典灰色，专业稳重',
    preview: 'linear-gradient(135deg, #64748b 0%, #475569 100%)'
  },
  {
    id: 'cyberpunk',
    name: 'Cyberpunk',
    description: '赛博朋克，炫酷科技',
    preview: 'linear-gradient(135deg, #ff0080 0%, #7928ca 100%)'
  },
  {
    id: 'sunset',
    name: 'Sunset',
    description: '日落橙红，温暖热情',
    preview: 'linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)'
  },
  {
    id: 'forest',
    name: 'Forest',
    description: '森林绿色，自然清新',
    preview: 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
  }
]

export const ThemeSwitcher: React.FC = () => {
  const [currentTheme, setCurrentTheme] = useState('infinite-talk')

  useEffect(() => {
    // 从 localStorage 读取主题
    const savedTheme = localStorage.getItem('genesis-theme') || 'infinite-talk'
    setCurrentTheme(savedTheme)
    applyTheme(savedTheme)
  }, [])

  const applyTheme = (themeId: string) => {
    // 移除所有主题类
    document.body.classList.remove(
      ...themes.map(t => `theme-${t.id}`)
    )
    // 添加新主题类
    document.body.classList.add(`theme-${themeId}`)
    // 保存到 localStorage
    localStorage.setItem('genesis-theme', themeId)
  }

  const handleThemeChange = (themeId: string) => {
    setCurrentTheme(themeId)
    applyTheme(themeId)
  }

  return (
    <div style={{ padding: '24px' }}>
      <h3 style={{ marginBottom: '16px', fontSize: '18px', fontWeight: 600 }}>
        🎨 选择主题皮肤
      </h3>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
        gap: '16px'
      }}>
        {themes.map(theme => (
          <div
            key={theme.id}
            onClick={() => handleThemeChange(theme.id)}
            style={{
              padding: '16px',
              border: currentTheme === theme.id 
                ? '3px solid var(--primary-color, #667eea)' 
                : '2px solid #e2e8f0',
              borderRadius: '12px',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              background: 'white'
            }}
            className={currentTheme === theme.id ? 'theme-card-active' : ''}
          >
            {/* 预览色块 */}
            <div style={{
              height: '60px',
              background: theme.preview,
              borderRadius: '8px',
              marginBottom: '12px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
            }} />
            
            {/* 主题名称 */}
            <div style={{
              fontWeight: 600,
              fontSize: '16px',
              marginBottom: '4px',
              color: currentTheme === theme.id ? 'var(--primary-color, #667eea)' : '#1e293b'
            }}>
              {theme.name}
            </div>
            
            {/* 主题描述 */}
            <div style={{
              fontSize: '12px',
              color: '#64748b'
            }}>
              {theme.description}
            </div>
            
            {/* 选中标记 */}
            {currentTheme === theme.id && (
              <div style={{
                marginTop: '8px',
                padding: '4px 8px',
                background: theme.preview,
                color: 'white',
                borderRadius: '4px',
                fontSize: '12px',
                fontWeight: 600,
                textAlign: 'center'
              }}>
                ✓ 当前主题
              </div>
            )}
          </div>
        ))}
      </div>

      {/* 主题预览 */}
      <div style={{ marginTop: '32px' }}>
        <h4 style={{ marginBottom: '16px', fontSize: '16px', fontWeight: 600 }}>
          主题预览
        </h4>
        
        {/* 标签页预览 */}
        <div className="theme-tabs">
          <button className="theme-tab active">文生图</button>
          <button className="theme-tab">历史记录</button>
          <button className="theme-tab">设置</button>
        </div>

        {/* 按钮预览 */}
        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <button className="theme-button">生成图像</button>
          <button className="theme-button" style={{ opacity: 0.7 }}>取消</button>
        </div>

        {/* 进度条预览 */}
        <div className="theme-progress">
          <div className="theme-progress-bar" style={{ width: '60%' }} />
        </div>
      </div>
    </div>
  )
}

export default ThemeSwitcher
