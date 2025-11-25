import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { App as AntdApp } from 'antd'
import BasicLayout from './layouts/BasicLayout'
import TextToImage from './pages/TextToImage'
import TextToVideo from './pages/TextToVideo'
import TaskHistory from './pages/TaskHistory'
import Settings from './pages/Settings'
import ThemeSettings from './pages/ThemeSettings'

const App: React.FC = () => {
  return (
    <AntdApp>
      <BrowserRouter>
        <BasicLayout>
          <Routes>
            <Route path="/" element={<Navigate to="/text-to-image" replace />} />
            <Route path="/text-to-image" element={<TextToImage />} />
            <Route path="/text-to-video" element={<TextToVideo />} />
            <Route path="/history" element={<TaskHistory />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/themes" element={<ThemeSettings />} />
          </Routes>
        </BasicLayout>
      </BrowserRouter>
    </AntdApp>
  )
}

export default App
