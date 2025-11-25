import React from 'react'
import { Card } from 'antd'
import ThemeSwitcher from '../../components/ThemeSwitcher'

const ThemeSettings: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Card>
        <ThemeSwitcher />
      </Card>
    </div>
  )
}

export default ThemeSettings
